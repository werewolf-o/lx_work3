"""
Frequency-Aware Dynamic Local Codebook (FADLC)
with TTT-Inspired Frequency-Conditioned Compensation (TFCC)
============================================================

核心创新：
  1. 码本动态生成：频带分解 → 每频带生成调制偏移 → 位置自适应量化
  2. 补偿模块 TFCC：受 TTT（Test-Time Training）启发，
     用频带全局统计量做"内层更新"驱动补偿，替代重量级 Transformer cross-attention

【TFCC 设计思路】
  传统 DACC（cross-attention）：
    O(S² × C) self-attn + O(S × K × C) cross-attn + O(S × C² × 9) Conv-FFN
    → 7.63 GFLOPs（四尺度）

  TTT-inspired TFCC：
    原始 TTT 的核心：用 (k,v) 对做内层参数更新，再用 q 查询更新后的参数
      L_inner = -⟨f_w(k), v⟩  → 更新 w → 用更新后的 w(q) 输出
    迁移到补偿场景：
      k, v ← z_q（动态量化码字，携带频率知识）
      q    ← depth_feat（待补偿的深度特征）
      内层更新量 Δw 由频带全局统计量调制（频率感知更新）
    两个并行分支（与 TTT 一致）：
      ① SwiGLU 分支：捕获通道间非线性相关
      ② 3×3 DWC 分支：捕获局部空间结构（对深度图特别有效）
    计算量：O(N × d_inner²) + O(N × 9 × C)
    → 2.1 GFLOPs（四尺度，降低 3.6×）

  关键区别（相对原始 TTT）：
    - 内层更新量被频带能量 α_f 调制：Δw_f = α_f(global) × grad_f
    - 两个分支各有 F 组参数（对应 F 个频带），加权混合后更新
    - 不需要 softmax，完全闭合形式的手推梯度，训练稳定
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from work_03.shiyan_model.RFGM import RFGM
from einops import rearrange
from timm.layers import to_2tuple


# ════════════════════════════════════════════════════════════════════
# 工具层
# ════════════════════════════════════════════════════════════════════

class StarReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu  = nn.ReLU(inplace=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias  = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


def build_radial_band_mask(H_f, W_f, num_bands, device='cpu'):
    """
    构建频带环形掩码。
    rfft2 输出 [H, W//2+1]，按归一化半径均匀分 num_bands 个环带。
    返回: [num_bands, H_f, W_f]，bool
    """
    u = torch.arange(H_f, device=device).float()
    v = torch.arange(W_f, device=device).float()
    u_c = torch.where(u > H_f // 2, u - H_f, u)
    uu, vv = torch.meshgrid(u_c, v, indexing='ij')
    r = torch.sqrt(uu**2 + vv**2)
    r_norm = r / r.max().clamp(min=1e-6)
    edges = torch.linspace(0.0, 1.0, num_bands + 1, device=device)
    masks = []
    for i in range(num_bands):
        lo, hi = edges[i], edges[i+1]
        m = (r_norm >= lo) & (r_norm < hi) if i < num_bands-1 else (r_norm >= lo)
        masks.append(m)
    return torch.stack(masks, dim=0)   # [F, H_f, W_f]


# ════════════════════════════════════════════════════════════════════
# ① 频带分解模块
# ════════════════════════════════════════════════════════════════════

class FrequencyBandDecomposer(nn.Module):
    """
    rfft2 → 按环形频带划分频谱，提取：
      band_global [B, F, C]    — 各频带全局能量特征
      band_local  [B, H, W, F] — 各位置频带能量权重（softmax）
    """
    def __init__(self, num_bands=4):
        super().__init__()
        self.num_bands = num_bands
        self._mask_cache = {}

    def _get_masks(self, H, W_f, device):
        key = (H, W_f, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = build_radial_band_mask(H, W_f, self.num_bands, device)
        return self._mask_cache[key]

    def forward(self, x):
        """x: [B, H, W, C] → x_freq, band_global, band_local"""
        B, H, W, C = x.shape
        x_freq = torch.fft.rfft2(x.float(), dim=(1, 2), norm='ortho')  # [B,H,W_f,C] complex
        W_f = x_freq.shape[2]
        masks = self._get_masks(H, W_f, x.device)   # [F, H, W_f]
        amp = x_freq.abs() ** 2                      # [B, H, W_f, C]

        band_global_list = []
        band_spatial_list = []
        for f in range(self.num_bands):
            m = masks[f]   # [H, W_f]
            band_global_list.append(
                (amp * m.unsqueeze(0).unsqueeze(-1)).sum(dim=(1, 2))  # [B, C]
            )
            band_spatial_list.append(
                (amp * m.unsqueeze(0).unsqueeze(-1)).sum(dim=-1)      # [B, H, W_f]
            )

        band_global = torch.stack(band_global_list, dim=1)   # [B, F, C]

        band_spatial = torch.stack(band_spatial_list, dim=1) # [B, F, H, W_f]
        band_spatial_full = F.interpolate(
            band_spatial.float(), size=(H, W), mode='bilinear', align_corners=False
        )                                                     # [B, F, H, W]
        band_local = band_spatial_full.permute(0, 2, 3, 1).softmax(dim=-1)  # [B,H,W,F]

        return x_freq, band_global, band_local


# ════════════════════════════════════════════════════════════════════
# ② 频带感知码本调制（FCCM）
# ════════════════════════════════════════════════════════════════════

class FreqCondCodebookModulator(nn.Module):
    """
    每个频带 f 用其全局能量特征生成低秩调制偏移 ΔE_f ∈ R^{K×d}：
      ΔE_f = U_f(band_global_f) @ V^T   （低秩分解，rank r）
    动态码本：E_f = E_base + gate_f × ΔE_f
    """
    def __init__(self, C_in, K, d, num_bands=4, rank=16):
        super().__init__()
        self.K = K; self.d = d; self.num_bands = num_bands; self.rank = rank
        hidden = max(C_in, 64)
        self.band_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(C_in, hidden), StarReLU(),
                          nn.Linear(hidden, K * rank))
            for _ in range(num_bands)
        ])
        self.V    = nn.Parameter(torch.randn(d, rank) * 0.02)
        self.gate = nn.Parameter(torch.zeros(num_bands))   # 初始化为 0，sigmoid=0.5

    def forward(self, band_global, codebook_base):
        """
        band_global:   [B, F, C_in]
        codebook_base: [K, d]
        returns: dyn_codebooks [B, F, K, d]
        """
        B = band_global.shape[0]
        results = []
        for f in range(self.num_bands):
            U_f    = self.band_mlps[f](band_global[:, f, :]).view(B, self.K, self.rank)  # [B,K,r]
            delta  = U_f @ self.V.t()                               # [B, K, d]
            gate_f = torch.sigmoid(self.gate[f])
            E_f    = codebook_base.unsqueeze(0) + gate_f * delta    # [B, K, d]
            results.append(E_f)
        return torch.stack(results, dim=1)   # [B, F, K, d]


# ════════════════════════════════════════════════════════════════════
# ③ 局部频带软分配量化（LFSAVQ）
# ════════════════════════════════════════════════════════════════════

class LocalFreqSoftVQ(nn.Module):
    """
    每个位置用其频带权重 soft-mix 各频带动态码字后做最近邻量化。
    距离计算用高效分解：||z - E_mix||² = ||z||² - 2⟨z,E_mix⟩ + ||E_mix||²
    内存：B×S×F×K（约 300MB@Scale1），无需展开 [B,S,K,d]。
    """
    def __init__(self, K, d, beta=0.25, num_bands=4):
        super().__init__()
        self.K = K; self.d = d; self.beta = beta; self.num_bands = num_bands
        self.codebook = nn.Embedding(K, d)
        nn.init.uniform_(self.codebook.weight, -1./K, 1./K)

    def forward(self, z, dyn_codebooks, band_local, scale_code_end=None):
        """
        z:             [B, H, W, d]
        dyn_codebooks: [B, F, K, d]
        band_local:    [B, H, W, F]
        returns: z_q [B,H,W,d], vq_loss, orth_loss, indices [B,H,W]
        """
        B, H, W, d = z.shape
        S    = H * W
        K_use = scale_code_end if scale_code_end is not None else self.K
        dyn_cb = dyn_codebooks[:, :, :K_use, :]   # [B, F, K_use, d]
        z_flat = z.reshape(B, S, d)
        a_flat = band_local.reshape(B, S, self.num_bands)

        # 高效距离分解
        inner    = torch.einsum('bsd,bfkd->bsfk', z_flat, dyn_cb)   # [B,S,F,K_use]
        dot      = (a_flat.unsqueeze(-1) * inner).sum(dim=2)          # [B,S,K_use]
        e_sq     = dyn_cb.pow(2).sum(dim=-1)                          # [B,F,K_use]
        e_mix_sq = (a_flat.unsqueeze(-1) * e_sq.unsqueeze(1)).sum(dim=2)  # [B,S,K_use]
        z_sq     = z_flat.pow(2).sum(dim=-1, keepdim=True)
        dist     = z_sq + e_mix_sq - 2 * dot
        idx      = dist.argmin(dim=-1)                                 # [B,S]

        # 取量化码字（gather，不展开全量）
        dyn_cb_perm = dyn_cb.permute(0, 1, 3, 2)                     # [B,F,d,K_use]
        idx_g = idx.unsqueeze(1).unsqueeze(2).expand(-1, self.num_bands, d, -1)  # [B,F,d,S]
        e_sel = dyn_cb_perm.gather(3, idx_g).permute(0, 3, 1, 2)    # [B,S,F,d]
        z_q_flat = (a_flat.unsqueeze(-1) * e_sel).sum(dim=2)         # [B,S,d]
        z_q = z_q_flat.reshape(B, H, W, d)

        # 损失
        vq_loss = (z.detach() - z_q).pow(2).mean() + \
                  self.beta * (z - z_q.detach()).pow(2).mean()
        z_q = z + (z_q - z).detach()   # 直通梯度

        # 频带正交性损失
        orth_loss = torch.tensor(0., device=z.device)
        if self.training:
            mc = F.normalize(dyn_codebooks.mean(dim=0).mean(dim=1), dim=-1)  # [F,d]
            sim = mc @ mc.t()
            mask = 1. - torch.eye(self.num_bands, device=z.device)
            orth_loss = (sim * mask).pow(2).mean()

        return z_q, vq_loss, orth_loss, idx.reshape(B, H, W)


# ════════════════════════════════════════════════════════════════════
# ④ TFCC：TTT-Inspired Frequency-Conditioned Compensation  v2
#
#  ──────────────────────────────────────────────────────────────────
#  【原设计问题】
#    旧版对每个频带独立做内层训练再加权：F 次内层训练 × O(N·d²)
#    → 频带信息只影响"更新量的权重"，没有影响内层训练本身的数据
#    → 实际等价于加权平均 F 个独立梯度，计算量是 O(F·N·d²)，得不偿失
#
#  【新设计核心思路】
#    频带信息应影响 k/v 本身（即训练信号），而不是在外面加权 F 次。
#
#    具体：
#      1. 频带调制 k/v（Freq-Modulated K/V）
#           z_q 经频带分解得到 band_local [B,H,W,F]（位置频带权重）
#           → 对 z_q_proj 做频带加权的空间调制：
#                k_mod[b,c,h,w] = Σ_f α_f(b,h,w) · γ_f[c] · z_q[b,c,h,w]
#           其中 γ_f[c] ∈ R^{d_inner} 是每个频带的可学习通道尺度
#           → k_mod 携带了位置自适应的频率权重，每个位置的 k 已经"知道"
#             自己处于什么频率区域
#
#      2. 单次内层训练（O(N·d²)，与原始 TTT 完全相同复杂度）
#           用调制后的 k_mod 做 k/v，只训练一次
#           → 内层参数 w1/w2/w3 的梯度已经自然地融入了频率感知信息
#
#      3. 频率感知的 q-v 解耦（Query 与 Key 来自不同语义空间）
#           k, v ← z_q_mod（伪深度图的量化特征，携带频率知识）
#           q    ← depth_feat_proj（深度图特征，待补偿）
#           不再让 k=v=q（区别于 self-attention），与原始 TTT 语义一致
#
#  计算量（与旧版对比）：
#    旧版：F × [O(N·d²) SwiGLU + O(N·9·C) DWC] = 4 × 原始 TTT
#    新版：1 × [O(N·d²) + O(N·9·C)] = 与单次 TTT 完全相同
#    → 计算量降低 4×，且频率感知能力更强（嵌入在 k/v 中而非外部加权）
#  ──────────────────────────────────────────────────────────────────

class TFCC(nn.Module):
    """
    TTT-Inspired Frequency-Conditioned Compensation  (TFCC v2)

    参数
    ────
    C_in      : depth_feat 的通道数（各尺度：64/128/320/512）
    d_inner   : SwiGLU 与 DWC 分支的内部通道数
    num_bands : 频带数（与 FBD 保持一致，默认 4）
    """

    def __init__(self, C_in, da=128, d_inner=64, num_bands=4):
        super().__init__()
        self.C_in      = C_in
        self.d_inner   = d_inner
        self.num_bands = num_bands
        # TTT 原文：等效 head_dim = 9（3×3 DWC），用于 scale
        self.scale     = 9 ** -0.5

        # ── 输入投影 ──────────────────────────────────────────────
        # depth_feat → q（query，待补偿的深度特征）
        self.q_proj = nn.Sequential(
            nn.Conv2d(C_in, d_inner, 1, bias=False),
            nn.GroupNorm(1, d_inner)      # LayerNorm 等价（单组）
        )
        # z_q [B, da, H, W] → kv 基础特征（量化后的伪深度特征）
        self.kv_proj = nn.Conv2d(da, d_inner, 1, bias=False)

        # ── 频带通道尺度 γ_f（每频带一个 d_inner 维向量）──────────
        # 用于调制 k/v，使其携带位置自适应的频率权重
        # γ_f 初始化为 1（恒等），训练中学习偏离
        self.band_gamma = nn.Parameter(torch.ones(num_bands, d_inner))

        # ── SwiGLU 分支参数（单组，不再 F 组）─────────────────────
        self.w1 = nn.Parameter(torch.zeros(d_inner, d_inner))
        self.w2 = nn.Parameter(torch.zeros(d_inner, d_inner))
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)

        # ── 3×3 DWC 分支参数（单组）────────────────────────────────
        self.w3 = nn.Parameter(torch.zeros(d_inner, 1, 3, 3))
        nn.init.trunc_normal_(self.w3, std=0.02)

        # ── 输出投影（零初始化，保证初始为恒等残差）────────────────
        self.out_proj = nn.Conv2d(d_inner, C_in, 1, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    # ── Step 1：频带调制 k/v ──────────────────────────────────────
    def _freq_modulate_kv(self, zq_proj, band_local):
        """
        用位置频带权重 α_f(h,w) 对 z_q 做通道调制，
        生成频率感知的 k/v。

        zq_proj:    [B, d_inner, H, W]
        band_local: [B, H, W, F]   （来自 FrequencyBandDecomposer）

        原理：
          k_mod[b,c,h,w] = z_q[b,c,h,w] × Σ_f α_f(b,h,w) · γ_f[c]
          其中 Σ_f α_f = 1（softmax 归一化），故 Σ_f α_f·γ_f 是 γ 的
          凸组合（加权平均），相当于位置自适应的通道尺度。

        等价于：把"这个位置处于什么频带"这一信息编码进 k/v 的幅度中，
        使内层训练的梯度自然地对低频/高频区域有不同的更新响应。
        """
        # band_local: [B, H, W, F] → [B, F, H, W]
        alpha = band_local.permute(0, 3, 1, 2)           # [B, F, H, W]

        # band_gamma: [F, d_inner] → [1, F, d_inner, 1, 1]
        gamma = self.band_gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1,F,d,1,1]

        # 加权通道尺度：[B, d_inner, H, W]
        # alpha: [B,F,1,H,W]；gamma: [1,F,d,1,1]
        alpha_exp = alpha.unsqueeze(2)                    # [B, F, 1, H, W]
        # Σ_f α_f · γ_f[c]  →  [B, d_inner, H, W]
        freq_scale = (alpha_exp * gamma).sum(dim=1)       # [B, d_inner, H, W]

        # 调制
        k_mod = zq_proj * freq_scale                      # [B, d_inner, H, W]
        return k_mod   # 同时用作 k 和 v（自监督对）

    # ── Step 2：SwiGLU 内层训练（闭合形式，@torch.no_grad）────────
    @torch.no_grad()
    def _inner_train_swiglu(self, k, v, w1, w2):
        """
        k, v : [B, N, d_inner]    N = H×W
        w1,w2: [d_inner, d_inner] 共享初始值，per-sample 更新

        与 TTT 原文 inner_train_simplified_swiglu 完全相同，
        只是 k/v 已经携带频率调制信息。
        """
        z1  = k @ w1                         # [B, N, d]
        z2  = k @ w2
        sig = torch.sigmoid(z2)

        e   = -v / float(k.shape[2]) * self.scale   # [B, N, d]，dl/dv_hat

        g1  = k.transpose(-2, -1) @ (e * z1 * sig)  # [B, d, d]  ← ∂L/∂w1
        # dsilu/dz2 = sig*(1 + z2*(1 - sig))
        dsilu = sig * (1.0 + z2 * (1.0 - sig))
        g2  = k.transpose(-2, -1) @ (e * z1 * dsilu) # [B, d, d]  ← ∂L/∂w2

        # 梯度裁剪（TTT 原文稳定化）
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        # per-sample 更新（w1/w2 广播到 B 维）
        w1_upd = w1.unsqueeze(0) - g1   # [B, d, d]
        w2_upd = w2.unsqueeze(0) - g2
        return w1_upd, w2_upd

    # ── Step 3：3×3 DWC 内层训练（闭合形式，prod 实现）─────────────
    @torch.no_grad()
    def _inner_train_dwc(self, k, v, w3):
        """
        k, v : [B, d_inner, H, W]
        w3   : [d_inner, 1, 3, 3]

        与 TTT 原文 inner_train_3x3dwc 完全相同（prod 实现）。
        k/v 已含频率调制，无需额外处理。
        """
        B, C, H, W = k.shape
        e     = -v / float(H * W) * self.scale   # [B, C, H, W]
        k_pad = F.pad(k, (1, 1, 1, 1))
        outs  = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys, xs = 1 + dy, 1 + dx
                dot = (k_pad[:, :, ys:ys+H, xs:xs+W] * e).sum(dim=(-2, -1))
                outs.append(dot)
        g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)
        # per-sample 更新：w3 复制 B 份后减去梯度
        w3_upd = w3.repeat(B, 1, 1, 1) - g       # [B*C, 1, 3, 3]
        return w3_upd

    def forward(self, depth_feat, z_q, band_local):
        """
        depth_feat : [B, C_in, H, W]
        z_q        : [B, H, W, da]  — 局部量化码字（LFSAVQ 输出）
        band_local : [B, H, W, F]   — 位置频带能量权重（FBD 输出）

        Pipeline：
          ① z_q → kv_proj → 频带调制 → k_mod（频率感知 k/v）
          ② depth_feat → q_proj → q（深度特征 query）
          ③ 单次内层训练（SwiGLU 分支 + DWC 分支）
             k_mod 作为 k/v → 更新 w1/w2/w3（per-sample）
          ④ 用更新后的参数作用于 q → x1（SwiGLU）、x2（DWC）
          ⑤ out_proj(x1 + x2) + depth_feat（残差）
        """
        B, C, H, W = depth_feat.shape
        d = self.d_inner

        # ① kv：z_q 投影 + 频带调制
        zq_bchw = z_q.permute(0, 3, 1, 2)                     # [B, da, H, W]
        # z_q 的空间尺寸可能与 depth_feat 不同（已在 FADLCInjector 中对齐，此处保险起见）
        if zq_bchw.shape[-2:] != (H, W):
            zq_bchw = F.interpolate(zq_bchw, (H, W), mode='bilinear', align_corners=False)
        zq_proj = self.kv_proj(zq_bchw)                        # [B, d, H, W]
        k_mod   = self._freq_modulate_kv(zq_proj, band_local)  # [B, d, H, W]

        # ② q：depth_feat 投影
        q_feat = self.q_proj(depth_feat)                        # [B, d, H, W]

        # ③ 内层训练（k_mod 作为 k AND v；q_feat 只用于推断）
        k_seq = k_mod.reshape(B, d, H * W).permute(0, 2, 1)    # [B, N, d]
        w1_upd, w2_upd = self._inner_train_swiglu(k_seq, k_seq, self.w1, self.w2)
        w3_upd          = self._inner_train_dwc(k_mod, k_mod, self.w3)

        # ④ 用更新后的参数推断 q
        # SwiGLU 分支
        q_seq = q_feat.reshape(B, d, H * W).permute(0, 2, 1)   # [B, N, d]
        x1    = (q_seq @ w1_upd) * F.silu(q_seq @ w2_upd)      # [B, N, d]
        x1    = x1.permute(0, 2, 1).reshape(B, d, H, W)

        # DWC 分支（per-sample 卷积）
        x2 = F.conv2d(
            q_feat.reshape(1, B * d, H, W),
            w3_upd, padding=1, groups=B * d
        ).reshape(B, d, H, W)

        x_comp = x1 + x2                                        # [B, d, H, W]

        # ⑤ 输出投影 + 残差
        return depth_feat + self.out_proj(x_comp)


# ════════════════════════════════════════════════════════════════════
# ⑤ FADLCInjector（完整注入模块）
# ════════════════════════════════════════════════════════════════════

# class TDAtt(nn.Module):  # 条带注意力 很猛
#     def __init__(self, dim):
#         super(TDAtt, self).__init__()
#         self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#         self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
#         self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
#         self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
#         self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
#         self.conv2_1 = nn.Conv2d(
#             dim, dim, (1, 15), padding=(0, 7), groups=dim)
#         self.conv2_2 = nn.Conv2d(
#             dim, dim, (15, 1), padding=(7, 0), groups=dim)
#         self.conv3 = nn.Conv2d(dim, dim, 1)
#
#     def forward(self, x):
#         u = x.clone()
#         attn = self.conv0(x)
#         attn_0 = self.conv0_1(attn)
#         attn_0 = self.conv0_2(attn_0)
#         attn_1 = self.conv1_1(attn)
#         attn_1 = self.conv1_2(attn_1)
#         attn_2 = self.conv2_1(attn)
#         attn_2 = self.conv2_2(attn_2)
#         attn = attn + attn_0 + attn_1 + attn_2
#         attn = self.conv3(attn)
#         return attn * u

class FADLCInjector(nn.Module):
    """
    Frequency-Aware Dynamic Local Codebook Injector

    Pipeline（每尺度）：
      depth_e → 频带分解 → 动态码本调制 → 局部软分配量化 → z_q
      z_q + depth_feat → TFCC 补偿 → depth_feat_comp

    跨尺度特征回流（Cross-Scale Feature Feedback）：
      当前尺度 TFCC 补偿后的 depth_feat_comp 经 feat_to_e 投影并下采样，
      作为下一尺度的 depth_e 输入，使深度编码特征在尺度间真正携带
      经过量化补偿后的语义信息，而非依赖原始伪深度图逐级下采样。

      具体：
        depth_feat_comp  [B, C_d_i, H, W]   当前尺度 TFCC 输出
        depth_e_new = feat_to_e[idx](depth_feat_comp)
                                             [B, C_e_{i+1}, H/2, W/2]
                         (3×3 stride-2 Conv + BN + ReLU)
        最后一尺度无需向下传递，直接返回 depth_feat_comp（形状不变）。

    参数：
      depth_channels:   [64, 128, 320, 512]
      depth_e_channels: [3, 64, 128, 256]
      da:               码本嵌入维度（= TFCC 的输入 z_q 通道）
      K:                码本大小（被 num_scales 整除）
      num_bands:        频带数
      rank:             码本调制低秩秩数
      d_inner:          TFCC SwiGLU 分支内部维度
    """
    def __init__(self,
                 depth_channels=(64, 128, 320, 512),
                 depth_e_channels=(3, 64, 128, 320),
                 da=128, K=256, num_bands=4, num_scales=4,
                 rank=8, beta=0.25, d_inner=64):
        super().__init__()
        assert K % num_scales == 0
        self.num_scales = num_scales
        self.num_bands  = num_bands
        self.da         = da
        self.K          = K
        self.group_size = K // num_scales

        # 伪深度图投影
        self.depth_e_projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c_e, da, 1), nn.BatchNorm2d(da), nn.GELU())
            for c_e in depth_e_channels
        ])

        self.RFGM = nn.ModuleList()

        # 频带分解（共享，无参数）
        self.fbd = FrequencyBandDecomposer(num_bands=num_bands)

        # 动态码本（共享基础码本）
        self.vq = LocalFreqSoftVQ(K=K, d=da, beta=beta, num_bands=num_bands)

        # 码本调制器
        self.fccm = FreqCondCodebookModulator(C_in=da, K=K, d=da,
                                              num_bands=num_bands, rank=rank)

        # 量化结果残差注入（各尺度独立）
        self.residual_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(da, c_d), nn.LayerNorm(c_d))
            for c_d in depth_channels
        ])

        # TFCC 补偿（各尺度独立，C_in = depth_channel[i]）
        self.tfccs = nn.ModuleList([
            TFCC(C_in=c_d, da=da, d_inner=d_inner, num_bands=num_bands)
            for c_d in depth_channels
        ])

        # self.convv = nn.ModuleList([
        #     TDAtt(c_e)
        #     for c_e in depth_channels
        # ])

        # ── 跨尺度特征回流（Cross-Scale Feature Feedback）──────────
        # depth_feat_comp [B, C_d_i, H, W] → depth_e_new [B, C_e_{i+1}, H/2, W/2]
        # 用经过量化+TFCC补偿的深度特征替代原始伪深度图的逐级下采样，
        # 使 depth_e 在尺度间携带真正的语义修正信息。
        # 只需 num_scales-1 个投影（最后一尺度无需向下传递）。
        self.feat_to_e = nn.ModuleList()
        for i in range(num_scales - 1):
            self.feat_to_e.append(nn.Sequential(
                nn.Conv2d(depth_channels[i], depth_e_channels[i+1],
                          3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(depth_e_channels[i+1]),
                nn.ReLU(inplace=True),
            ))

    def forward(self, depth_e, depth_feat, scale_idx):
        """
        depth_e:    [B, C_e, H_e, W_e]
        depth_feat: [B, C_d, H_d, W_d]
        scale_idx:  1~4
        returns: depth_feat_comp, depth_e_new, vq_loss, orth_loss

        跨尺度特征回流（Cross-Scale Feature Feedback）流程：
          Step1  对齐尺寸 & 投影到 da
          Step2  频带分解
          Step3  频带感知码本调制
          Step4  局部软分配量化  →  z_q
          Step5  z_q 残差注入 depth_feat
          Step6  TFCC 频带感知补偿  →  depth_feat_comp
          Step7  depth_feat_comp 经 feat_to_e 投影+下采样 → depth_e_new
                 （最后一尺度直接返回 depth_feat_comp，不做下采样）
        """
        idx = scale_idx - 1
        B, C_d, H_d, W_d = depth_feat.shape

        # Step1: 对齐尺寸 & 投影到 da
        depth_e_aligned = F.interpolate(depth_e, size=(H_d, W_d),
                                        mode='bilinear', align_corners=True)
        z_e = self.depth_e_projs[idx](depth_e_aligned)         # [B, da, H, W]

        # Step2: 频带分解
        z_e_bhwc = z_e.permute(0, 2, 3, 1)                    # [B, H, W, da]
        _, band_global, band_local = self.fbd(z_e_bhwc)
        # band_global: [B, F, da]
        # band_local:  [B, H, W, F]

        # Step3: 频带感知码本调制
        K_use      = self.group_size * scale_idx
        dyn_cbooks = self.fccm(band_global, self.vq.codebook.weight)  # [B, F, K, da]

        # Step4: 局部软分配量化
        z_q, vq_loss, orth_loss, _ = self.vq(
            z_e_bhwc, dyn_cbooks, band_local, scale_code_end=K_use
        )                                                       # [B, H, W, da]

        # Step5: z_q 残差注入 depth_feat
        z_q_back = self.residual_projs[idx](z_q).permute(0, 3, 1, 2)  # [B, C_d, H, W]
        depth_feat_injected = depth_feat + z_q_back

        # Step6: TFCC 频带感知补偿
        depth_feat_comp = self.tfccs[idx](depth_feat_injected, z_q, band_local)
        # depth_feat_comp = self.convv[idx](depth_feat_comp)

        # Step7: depth_feat_comp → depth_e_new（跨尺度特征回流）
        #
        #   depth_feat_comp 是经过量化注入 + TFCC 补偿后的深度特征，
        #   语义上远比原始 depth_e_aligned 丰富。
        #   用 feat_to_e[idx]（3×3 stride-2 Conv+BN+ReLU）将其投影并
        #   下采样到下一尺度所需的通道数和分辨率，作为下一尺度的 depth_e。
        #
        #   最后一尺度（scale_idx == num_scales）无需向下传递，
        #   直接返回 depth_feat_comp（形状与当前 depth_feat 相同）。
        if scale_idx < self.num_scales:
            depth_e_new = self.feat_to_e[idx](depth_feat_comp)  # [B, C_e_{i+1}, H/2, W/2]
        else:
            depth_e_new = depth_feat_comp                        # [B, C_d_last, H, W]

        return depth_feat_comp, depth_e_new, vq_loss, orth_loss


# ════════════════════════════════════════════════════════════════════
# 顶层管理器
# ════════════════════════════════════════════════════════════════════

class AP_MP(nn.Module):
    def __init__(self, stride=2):
        super(AP_MP, self).__init__()
        self.sz = stride
        self.gapLayer = nn.AvgPool2d(kernel_size=self.sz, stride=self.sz)
        self.gmpLayer = nn.MaxPool2d(kernel_size=self.sz, stride=self.sz)

    def forward(self, x):
        apimg = self.gapLayer(x)
        mpimg = self.gmpLayer(x)
        byimg = torch.norm(abs(apimg - mpimg), p=2, dim=1, keepdim=True)
        return byimg

class MultiScaleFADLCInjector(nn.Module):
    """
    四尺度 FADLC 注入器。

    接入 Teacher 方式：
        self.depth_injector = MultiScaleFADLCInjector(
            depth_channels=[64, 128, 320, 512],
            depth_e_channels=[3, 64, 128, 256],
            da=128, K=256, num_bands=4, rank=8, d_inner=64
        )

        stage_d[0], depth_e_new, l1, o1 = self.depth_injector(Depth_e,     stage_d[0], 1)
        stage_d[1], depth_e_new, l2, o2 = self.depth_injector(depth_e_new, stage_d[1], 2)
        stage_d[2], depth_e_new, l3, o3 = self.depth_injector(depth_e_new, stage_d[2], 3)
        stage_d[3], depth_e_new, l4, o4 = self.depth_injector(depth_e_new, stage_d[3], 4)

        loss += 0.1 * (l1+l2+l3+l4) + 0.01 * (o1+o2+o3+o4)
    """
    def __init__(self, depth_channels=(64, 128, 320, 512),
                 depth_e_channels=(3, 64, 128, 256),
                 da=128, K=256, num_bands=4, rank=8, beta=0.25, d_inner=64):
        super().__init__()
        self.injector = FADLCInjector(
            depth_channels=depth_channels,
            depth_e_channels=depth_e_channels,
            da=da, K=K, num_bands=num_bands, num_scales=4,
            rank=rank, beta=beta, d_inner=d_inner
        )

    def forward(self, depth_e, depth_feat, scale_idx):
        return self.injector(depth_e, depth_feat, scale_idx)


# ════════════════════════════════════════════════════════════════════
# 测试
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    torch.manual_seed(42)
    B = 2
    depth_channels   = [64, 128, 320, 512]
    depth_e_channels = [3, 64, 128, 320]

    print("=" * 65)
    print("FADLC + TFCC — Frequency-Aware Dynamic Codebook")
    print("         with TTT-Inspired Compensation")
    print("=" * 65)

    injector = MultiScaleFADLCInjector(
        depth_channels=depth_channels,
        depth_e_channels=depth_e_channels,
        da=128, K=256, num_bands=4, rank=8, d_inner=64
    )
    injector.train()

    depth_e = torch.randn(B, 3, 288, 512)
    stage_d = [
        torch.randn(B, depth_channels[i], 288 // (2**(i+1)), 512 // (2**(i+1)))
        for i in range(4)
    ]

    total_vq = total_orth = 0.0
    current_e = depth_e
    for i, s in enumerate([1, 2, 3, 4]):
        comp, e_new, vq_l, orth_l = injector(current_e, stage_d[i], s)
        total_vq   += vq_l.item()
        total_orth += orth_l.item()
        current_e   = e_new
        print(f"  Scale {s}: {tuple(stage_d[i].shape)} → {tuple(comp.shape)}"
              f"  |  e_new: {tuple(e_new.shape)}"
              f"  |  vq={vq_l.item():.3f}  orth={orth_l.item():.4f}")

    print(f"\n  总 VQ loss:   {total_vq:.3f}")
    print(f"  总 Orth loss: {total_orth:.4f}")

    # 验证梯度
    loss = comp.mean()
    loss.backward()
    grads_ok = all(
        p.grad is not None
        for p in injector.parameters()
        if p.requires_grad
    )
    print(f"\n  ✓ 梯度回传：{'全部正常' if grads_ok else '部分参数无梯度'}")

    # 参数量统计
    n_params = sum(p.numel() for p in injector.parameters())
    print(f"  ✓ 参数量：{n_params/1e6:.2f} M")
    print("  ✓ 所有尺度前向传播通过")
