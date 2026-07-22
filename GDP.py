"""
GBC-CPG Bottleneck
==================
输入:  f4  (B, C, H, W)   最深层特征
输出:
  f4_refined   (B, C, H, W)  增强后的 f4，送两路解码器
  depth_guide  (B, C, H, W)  深度分支注意力先验

流程：
  Stage-1 [GBC 替换 CPG seeds]:
    f4 → GBC → refined, att
    att 归属熵 → att_conf (B,1,H,W)  ∈[0,1]，高=清晰归属某缺陷原型

  Stage-2 [保留 CPG top-k 代表种子逻辑]:
    att_conf 作为 M^s（显著图）过滤 embedding
    → defect seeds → top-k 选代表
    → 代表种子做卷积权重 → 缺陷感知特征 F_def
    → 空间注意力增强 → defect prompt P_def

  输出构造:
    f4_refined  = f4 + proj_up(P_def)   残差增强
    depth_guide = conv(cat([P_def, 1-att_conf]))  不确定区域引导深度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ══════════════════════════════════════════════════════════════════
# GranularBall  (Stage-1，替换 CPG saliency seeds)
# ══════════════════════════════════════════════════════════════════
class GranularBall(nn.Module):
    def __init__(self, dim, num_balls=16, tau=1.0):
        super().__init__()
        self.K   = num_balls
        self.tau = tau
        # 可学习球中心（对应 CPG 的 Sseed）
        self.centers   = nn.Parameter(torch.randn(num_balls, dim) * 0.01)
        # per-ball 对角协方差（CPG 没有，GBC 改进点）
        self.log_sigma = nn.Parameter(torch.zeros(num_balls, dim))
        self.bn        = nn.BatchNorm2d(dim)
        self.refine    = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        x: (B, d, H, W)
        返回:
          refined   (B, d, H, W)  球重建 + 残差
          att       (B, HW, K)    软归属矩阵
          att_conf  (B, 1, H, W)  归属置信度 ∈[0,1]
        """
        B, d, H, W = x.shape
        z      = self.bn(x)
        z_flat = z.view(B, d, -1).permute(0, 2, 1)              # (B, HW, d)

        # 距离计算（sigma 缩放）
        dif    = z_flat.unsqueeze(2) - \
                 self.centers.unsqueeze(0).unsqueeze(0)           # (B, HW, K, d)
        sigma  = F.softplus(self.log_sigma).unsqueeze(0).unsqueeze(0)  # (1,1,K,d)
        dist2  = ((dif / (sigma + 1e-6)) ** 2).sum(-1)           # (B, HW, K)

        # 软归属（对应 CPG 的 Ssoft）
        att    = F.softmax(-dist2 / max(self.tau, 1e-6), dim=-1)  # (B, HW, K)

        # 重建特征（对应 CPG seeds 更新后重建 embedding）
        recon  = torch.matmul(att, self.centers)                  # (B, HW, d)
        recon  = recon.permute(0, 2, 1).view(B, d, H, W)
        refined = self.refine(recon + x)                          # 残差

        # 归属置信度（对应 M^s 的功能）
        eps      = 1e-8
        H_att    = -(att * torch.log(att + eps)).sum(-1)          # (B, HW)
        H_max    = torch.log(torch.tensor(
                       self.K, dtype=torch.float32, device=x.device))
        att_conf = (1.0 - H_att / (H_max + eps)).view(B, 1, H, W)  # (B,1,H,W)

        return refined, att, att_conf


# ══════════════════════════════════════════════════════════════════
# DefectSeedStage  (Stage-2，保留 CPG top-k 代表种子逻辑)
#
# 对应 CPG 公式(5)(6)(7)(8)：
#   score    = Co_seed ⊗ avg(Pem × Ms)^T
#   Co_rep   = gather(Co_seed, argtopk(score))
#   F_Co     = conv(L2(Pem), weight=v(Co_rep))
#   P_Co     = v(conv(F_Co × S_att))
#
# 适配单图道路缺陷（N=1，NL→HW，其余逻辑不变）：
#   att_conf 替代 Ms
#   refined  替代 Pem（GBC 输出的结构化特征）
# ══════════════════════════════════════════════════════════════════
class DefectSeedStage(nn.Module):
    def __init__(self, dim, top_k=16):
        super().__init__()
        self.top_k = top_k
        # 空间注意力（对应 CPG 的 S_att）
        self.spatial_att = nn.Sequential(
            nn.Conv2d(dim, 1, 1, bias=False),
            nn.Sigmoid(),
        )
        # consensus 特征后处理（对应 CPG 的 conv(F_Co × S_att)）
        self.post = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, refined, att_conf):
        """
        refined:  (B, d, H, W)  GBC 精炼特征，对应 CPG 的 Pem
        att_conf: (B, 1, H, W)  GBC 置信度图，对应 CPG 的 Ms

        Step1: att_conf 过滤 refined → defect seeds  (对应 Co_seed)
        Step2: top-k 选代表种子                       (对应 Co_rep)
        Step3: 代表种子做卷积权重 → F_def             (对应 F_Co)
        Step4: 空间注意力增强 → P_def                 (对应 P_Co)
        """
        B, d, H, W = refined.shape
        k = min(self.top_k, H * W)

        z_flat   = refined.view(B, d, -1).permute(0, 2, 1)      # (B, HW, d)
        c_flat   = att_conf.view(B, 1, -1).permute(0, 2, 1)     # (B, HW, 1)

        # Step1: 用 att_conf 加权得到 defect seeds (对应 Co_seed)
        defect_seeds = l2_norm(z_flat * c_flat, dim=-1)          # (B, HW, d)

        # Step2: top-k 选代表  (对应 CPG 公式5、6)
        # score = Co_seed ⊗ avg(Pem × Ms)^T
        avg_feat = (z_flat * c_flat).sum(1, keepdim=True) / \
                   (c_flat.sum(1, keepdim=True) + 1e-8)          # (B, 1, d)
        avg_feat = l2_norm(avg_feat, dim=-1)
        score    = torch.bmm(defect_seeds,
                             avg_feat.permute(0, 2, 1)).squeeze(-1)  # (B, HW)
        _, topk_idx = score.topk(k, dim=-1)                      # (B, k)
        rep_seeds   = torch.gather(
            defect_seeds,
            1,
            topk_idx.unsqueeze(-1).expand(-1, -1, d)
        )                                                         # (B, k, d)
        rep = l2_norm(rep_seeds.mean(dim=1), dim=-1)             # (B, d) 代表向量

        # Step3: rep 做卷积权重 → 缺陷感知特征  (对应 CPG 公式7)
        # F_Co = conv(L2(Pem), weight=v(Co_rep))
        attn  = torch.bmm(
            l2_norm(z_flat, dim=-1),
            rep.unsqueeze(-1)                                     # (B, d, 1)
        )                                                         # (B, HW, 1)
        attn  = F.softmax(attn, dim=1)
        F_def = (z_flat * attn).permute(0, 2, 1).view(B, d, H, W)  # (B,d,H,W)

        # Step4: 空间注意力增强  (对应 CPG 公式8)
        # P_Co = v(conv(F_Co × S_att))
        s_att = self.spatial_att(refined)                        # (B, 1, H, W)
        P_def = self.post(F_def * s_att)                         # (B, d, H, W)
        return P_def


# ══════════════════════════════════════════════════════════════════
# GBCCPGBottleneck  (完整模块，对外接口)
# ══════════════════════════════════════════════════════════════════
class GBCCPGBottleneck(nn.Module):
    """
    输入:  f4          (B, C, H, W)
    输出:
      f4_refined   (B, C, H, W)  = f4 + proj_up(P_def)
      depth_guide  (B, C, H, W)  = conv(cat([P_def_low, 1-att_conf]))

    内部降维 C → Cr = C//r，在低维空间做 GBC + 种子聚类，
    节省计算量并减少可调参数（参考 CPG 的 scale-r 线性层设计）。
    """
    def __init__(self, dim, num_balls=16, top_k=16, r=2):
        super().__init__()
        Cr = dim // r
        self.Cr = Cr

        # 降维（对应 CPG 的 tunable linear with scale r）
        self.proj_down = nn.Sequential(
            nn.Conv2d(dim, Cr, 1, bias=False),
            nn.BatchNorm2d(Cr),
        )
        # Stage-1: GBC
        self.gbc = GranularBall(Cr, num_balls=num_balls)
        # Stage-2: top-k 代表种子
        self.dss = DefectSeedStage(Cr, top_k=top_k)

        # 升维回 dim（对应 CPG 的 MLP_up）
        self.proj_up = nn.Sequential(
            nn.Conv2d(Cr, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        # depth_guide: P_def_low + RGB不确定图 → 深度引导信号
        self.depth_proj = nn.Sequential(
            nn.Conv2d(Cr + 1, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, f4):
        # 降维
        z = self.proj_down(f4)                        # (B, Cr, H, W)

        # Stage-1: GBC
        refined, att, att_conf = self.gbc(z)          # refined:(B,Cr,H,W)

        # Stage-2: top-k 代表种子
        P_def = self.dss(refined, att_conf)           # (B, Cr, H, W)

        # depth_guide（升维前构建，语义更纯净，参数更少）
        depth_guide = self.depth_proj(
            torch.cat([P_def, 1.0 - att_conf], dim=1)  # (B, Cr+1, H, W)
        )                                               # (B, C, H, W)

        # f4_refined: 升维后残差加到原始 f4
        f4_refined = f4 + self.proj_up(P_def)          # (B, C, H, W)

        return f4_refined, depth_guide


# ══════════════════════════════════════════════════════════════════
# 单元测试
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, H, W = 2, 256, 9, 16   # f4 尺寸 (288//32, 512//32)

    model = GBCCPGBottleneck(dim=C, num_balls=16, top_k=16, r=2)
    f4    = torch.randn(B, C, H, W)

    f4_refined, depth_guide = model(f4)

    print("=== GBCCPGBottleneck ===")
    print(f"输入  f4          : {f4.shape}")
    print(f"输出  f4_refined  : {f4_refined.shape}")
    print(f"输出  depth_guide : {depth_guide.shape}")

    # 残差验证：f4_refined 应该和 f4 形状完全一致
    assert f4_refined.shape == f4.shape
    assert depth_guide.shape == f4.shape

    # 参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"新增参数量: {params/1e6:.3f} M")

    # 梯度验证
    loss = f4_refined.mean() + depth_guide.mean()
    loss.backward()
    print("梯度回传: OK")
