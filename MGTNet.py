import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.Segformer.mix_transformer3 import mit_b2, mit_b0,mit_b1
from work_03.shiyan_model.Adapter import DynamicFilter
from work_03.shiyan_model.Fusion import Fusion
from work_03.shiyan_model.freq_dynamic_codebook3 import MultiScaleFADLCInjector
from work_03.shiyan_model.gbc_cpg_bottleneck import GBCCPGBottleneck


class EvidentialDualHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # 每路各出一组 Dirichlet 参数（每个像素、每个类的 evidence logit）
        # self.head_V = nn.Conv2d(in_ch, num_classes, 1)
        # self.head_D = nn.Conv2d(in_ch, num_classes, 1)

    # ── 内部：从 logit → Dirichlet 参数、opinion、期望概率 ──────
    @staticmethod
    def _to_opinion(logit, C):
        """
        logit: (B, C, H, W)  原始输出
        C: int  类别数
        返回:
          alpha  (B, C, H, W)  Dirichlet 参数 = e + 1
          E_p    (B, C, H, W)  期望概率
          u      (B, 1, H, W)  不确定度
          b      (B, C, H, W)  belief mass
        """
        e = F.softplus(logit)  # evidence ≥ 0
        alpha = e + 1.0  # α_c = e_c + 1
        S = alpha.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        E_p = alpha / S  # 期望概率
        u = C / S  # 不确定度（∈(0,1]）
        b = (alpha - 1.0) / S  # belief mass
        return alpha, E_p, u, b

    # ── 内部：Dempster-Shafer 二路聚合 ──────────────────────────
    @staticmethod
    def _dempster_shafer(b_V, u_V, b_D, u_D, C):
        """
        b_V, b_D: (B, C, H, W)  belief masses
        u_V, u_D: (B, 1, H, W)  uncertainty masses
        C: int
        返回:
          b_agg   (B, C, H, W)
          u_agg   (B, 1, H, W)
        """
        # 归一化因子 M = 1 - Σ_{i≠j} b_i^V * b_j^D
        # 等价于 M = Σ_c b_c^V*b_c^D + u^V + u^D - u^V*u^D
        M = (b_V * b_D).sum(dim=1, keepdim=True) + u_V + u_D - u_V * u_D
        M = M.clamp(min=1e-6)  # 避免除零

        b_agg = (b_V * b_D + b_V * u_D + b_D * u_V) / M  # (B, C, H, W)
        u_agg = u_V * u_D / M  # (B, 1, H, W)
        return b_agg, u_agg

    # ── 内部：Integrated Cross-Entropy loss（EviVLM 公式25）──────
    # @staticmethod
    # def _ice_loss(alpha, y_onehot):
    #     """
    #     alpha:    (B, C, H, W)
    #     y_onehot: (B, C, H, W)  one-hot 标签
    #     """
    #     S = alpha.sum(dim=1, keepdim=True)  # (B,1,H,W)
    #     # digamma 差值：ψ(S) - ψ(α_c)
    #     loss = y_onehot * (torch.digamma(S) - torch.digamma(alpha))
    #     return loss.sum(dim=1).mean()  # scalar

    # ────────────────────────────────────────────────────────────
    def forward(self, feat_V, feat_D):
        """
        feat_V, feat_D: (B, in_ch, H, W)
        y_onehot:       (B, C, H, W) float one-hot，训练时必须传入
        """
        C = self.num_classes

        # ── 每路 logit → opinion ──
        logit_V = feat_V  # (B, C, H, W)
        logit_D = feat_D

        alpha_V, prob_V, u_V, b_V = self._to_opinion(logit_V, C)
        alpha_D, prob_D, u_D, b_D = self._to_opinion(logit_D, C)

        # ── Dempster-Shafer 聚合 ──
        b_fused, u_agg = self._dempster_shafer(b_V, u_V, b_D, u_D, C)
        # 聚合后期望概率：由 b_fused + u_agg 重建 alpha_fused
        # alpha_fused_c = b_fused_c * (C / u_agg) + 1  （逆推）
        S_agg = C / u_agg.clamp(min=1e-6)
        alpha_fused = b_fused * S_agg + 1.0
        prob_fused = alpha_fused / alpha_fused.sum(dim=1, keepdim=True)

        # # ── 证据损失（仅训练时）──
        # loss_evi = torch.tensor(0.0, device=feat_V.device)
        # if y_onehot is not None:
        #     loss_V = self._ice_loss(alpha_V, y_onehot)
        #     loss_D = self._ice_loss(alpha_D, y_onehot)
        #     loss_fused = self._ice_loss(alpha_fused, y_onehot)
        #     loss_evi = loss_V + loss_D + loss_fused

        return alpha_V,alpha_D,alpha_fused

class OptimizedAdapter(nn.Module):
    def __init__(self, blk, dim=None, num_classes=3):
        super().__init__()
        self.block = blk
        # 这里 dim 应该是 block 的通道数/嵌入维：
        if dim is None:
            dim = blk.dwconv.out_channels
        hidden_dim = 128 # 或者 dim // 2, dim * 2 等你自己设定
        # 在这里接你的 LMSA_GMM
        self.mona = DynamicFilter(dim=dim, hidden_dim=128)

    def forward(self, x):
        x_permuted = x.permute(0, 2, 3, 1)
        x_enhanced = self.mona(x_permuted)
        x_enhanced = x_enhanced.permute(0, 3, 1, 2).contiguous()
        out = self.block(x_enhanced)
        return out

class BasicConv2d(nn.Module):    #很多模块的使用卷积层都是以其为基础，论文中的BConvN
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class FAM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, norm_layer=nn.BatchNorm2d):
        super(FAM, self).__init__()
        # self.conv_d1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        # self.conv_l = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channel_left*2, in_channel_left, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(in_channel_down)

    def forward(self, left, down):
        down_mask = down
        left_mask = left
        if down.size()[2:] != left_mask.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

class side_fusion(nn.Module):
    def __init__(self, in_channel, norm_layer =nn.BatchNorm2d):
        super(side_fusion,self).__init__()
        self.conv0 = nn.Conv2d(in_channel*2, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)
    def forward(self,sideout1,sideout2):
        out = torch.cat((sideout1,sideout2),dim=1)
        out = F.relu(self.bn0(self.conv0(out)),inplace=True)

        return out

class GlobalFusion_Single(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(GlobalFusion_Single, self).__init__()
        # self.conv1 = nn.Conv2d(in_channel, 256, 3, 1, 1)
        # self.conv2 = nn.Conv2d(256,       256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256 * 2,   256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, x_enc, x_side):
        x_enc  = x_enc
        x_side = x_side
        if x_side.size()[2:] != x_enc.size()[2:]:
            x_side = F.interpolate(x_side, x_enc.size()[2:], mode='bilinear')
        fused = x_enc * x_side
        out = torch.cat((x_enc, fused), dim=1)
        return F.relu(self.bn0(self.conv3(out)), inplace=True)

class Fusion_Single(nn.Module):
    def __init__(self, in_channel, norm_layer=nn.BatchNorm2d):
        super(Fusion_Single, self).__init__()
        self.conv0 = nn.Conv2d(in_channel * 2, 256, 3, 1, 1)
        # self.conv0 = nn.Conv2d(in_channel, 256, 3, 1, 1)
        self.bn0 = norm_layer(256)

    def forward(self, x, alpha, beta):
        # alpha: (B, C, 1, 1), beta: (B,1,1,1)
        out1 = alpha * x
        out2 = beta * x * x
        out  = torch.cat((out1, out2), dim=1)
        return F.relu(self.bn0(self.conv0(out)), inplace=True)

class Teacher(nn.Module):
    def __init__(self,
                 dinov3_weight_path='/media/user/shuju/LXNet/backbone/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth',
                 dinov3_local_path="/media/user/shuju/LXNet/dinov3-main"):
        super(Teacher, self).__init__()

        # RGB DINOV3
        self.dino1 = torch.hub.load(
            repo_or_dir=dinov3_local_path,
            model="dinov3_convnext_large",
            source="local",
            pretrained=False,
            trust_repo=True
        )
        if dinov3_weight_path:
            checkpoint = torch.load(dinov3_weight_path, map_location='cpu')
            self.dino1.load_state_dict(checkpoint, strict=True)
            print("✓ Local weights successfully loaded")

        for param in self.dino1.parameters():
            param.requires_grad = False

        adapted_stages = []
        for stage_idx, stage in enumerate(self.dino1.stages):
            adapted_blocks = []
            for block in stage:
                # 这里 dim 可以从 block 里拿，如果有 blk.dwconv.out_channels 就不用手写
                adapted_blocks.append(OptimizedAdapter(block, num_classes=3))
            adapted_stages.append(nn.Sequential(*adapted_blocks))
        self.dino1.stages = nn.ModuleList(adapted_stages)

        # Depth segformer_B1
        self.label_d = mit_b1()
        self.label_d.init_weights("/media/user/shuju/LXNet/backbone/Segformer/mit_b1.pth")
        channels = [64, 128, 320, 512]

        # Decoder
        self.fam54 = FAM(256, 256)
        self.fam43 = FAM(256, 256)
        self.fam32 = FAM(256, 256)
        self.gfusion3 = GlobalFusion_Single(256)
        self.gfusion2 = GlobalFusion_Single(256)
        self.gfusion1 = GlobalFusion_Single(256)
        self.fusion = Fusion_Single(256)
        self.sigmoid = nn.Sigmoid()
        self.linear_out = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            ##nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Linear(512, 256 + 1),
            nn.Sigmoid(),
        )

        self.fam54_d = FAM(256, 256)
        self.fam43_d = FAM(256, 256)
        self.fam32_d = FAM(256, 256)
        self.gfusion2_d = GlobalFusion_Single(256)
        self.gfusion1_d = GlobalFusion_Single(256)
        self.linear_outd = nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)

        self.Rreduce1 = nn.Conv2d(192, 256, 1)
        self.Rreduce2 = nn.Conv2d(384, 256, 1)
        self.Rreduce3 = nn.Conv2d(768, 256, 1)
        self.Rreduce4 = nn.Conv2d(1536, 256, 1)

        self.Dreduce1 = nn.Conv2d(channels[0], 256, 1)
        self.Dreduce2 = nn.Conv2d(channels[1], 256, 1)
        self.Dreduce3 = nn.Conv2d(channels[2], 256, 1)
        self.Dreduce4 = nn.Conv2d(channels[3], 256, 1)

        self.Fusion1 = Fusion(256, 72, 128)
        self.Fusion2 = Fusion(256, 36, 64)
        self.Fusion3 = Fusion(256, 18, 32)
        self.Fusion4 = Fusion(256, 9, 16)

        self.bottleneck = GBCCPGBottleneck(dim=256, num_balls=16, top_k=16, r=2)
        self.evi_head = EvidentialDualHead(num_classes=3)

        self.depth_injectors = MultiScaleFADLCInjector(
            depth_channels=channels,
            depth_e_channels=[3,64,128,320],
            da=128,
            K = 256,
            num_bands= 4,
            rank= 8,
            d_inner= 64
        )


    def encode_decode(self, RGB ,Depth,Depth_e):
        # RGB：[B,3,288,512]
        # Depth：[B,3,288,512]
        # Depth_e：[B,3,288,512]
        raw_size = RGB.size()[2:]
        bz = RGB.shape[0]
        stage_r = []
        stage_d = []
        enhanced_stage_d = []

        x = self.dino1._get_intermediate_layers(RGB, n=[0, 1, 2, 3])
        stage_r.append(self.Rreduce1(x[0][1]))
        stage_r.append(self.Rreduce2(x[1][1]))
        stage_r.append(self.Rreduce3(x[2][1]))
        stage_r.append(self.Rreduce4(x[3][1]))

        stage_d.append(self.label_d.layer1(Depth))
        stage_d[0], depth_e_new, vq_1, orth_1 = self.depth_injectors(Depth_e, stage_d[0], scale_idx=1)
        enhanced_stage_d.append(self.Dreduce1(stage_d[0]))

        stage_d.append(self.label_d.layer2(stage_d[0]))
        stage_d[1], depth_e_new, vq_2, orth_2 = self.depth_injectors(depth_e_new, stage_d[1], scale_idx=2)
        enhanced_stage_d.append(self.Dreduce2(stage_d[1]))

        stage_d.append(self.label_d.layer3(stage_d[1]))
        stage_d[2], depth_e_new, vq_3, orth_3 = self.depth_injectors(depth_e_new, stage_d[2], scale_idx=3)
        enhanced_stage_d.append(self.Dreduce3(stage_d[2]))

        stage_d.append(self.label_d.layer4(stage_d[2]))
        stage_d[3], depth_e_new, vq_4, orth_4 = self.depth_injectors(depth_e_new, stage_d[3], scale_idx=4)
        enhanced_stage_d.append(self.Dreduce4(stage_d[3]))

        # stage_d = self.label_d.forward_features(Depth)
        # stage_d[0] = self.Dreduce1(stage_d[0])
        # stage_d[1] = self.Dreduce2(stage_d[1])
        # stage_d[2] = self.Dreduce3(stage_d[2])
        # stage_d[3] = self.Dreduce4(stage_d[3])

        f1 = self.Fusion1(stage_r[0],enhanced_stage_d[0])
        f2 = self.Fusion2(stage_r[1],enhanced_stage_d[1])
        f3 = self.Fusion3(stage_r[2],enhanced_stage_d[2])
        f4 = self.Fusion4(stage_r[3],enhanced_stage_d[3])

        f4_refined, depth_guide = self.bottleneck(f4)

        # Decoder
        gap = self.gap1(f4_refined).view(bz, -1)
        feat = self.fc(gap)  # 现在只用 rgb
        gate = feat[:, -1].view(bz, 1, 1, 1)
        alpha = feat[:, :256].view(bz, 256, 1, 1)
        out4 = self.fam54(f3, f4_refined)  # H/16
        de3 = self.gfusion1(f2, out4)  # H/8
        out3 = self.fam43(de3, out4)  # H/8
        de2 = self.gfusion2(f1, out3)  # H/4
        out2 = self.fam32(de2, out3)  # H/4
        out = self.fusion(out2, alpha, gate)

        out4_d = self.fam54_d(enhanced_stage_d[2], enhanced_stage_d[3]*depth_guide)  # H/16
        de3 = self.gfusion1_d(enhanced_stage_d[1], out4_d)  # H/8
        out3_d = self.fam43_d(de3, out4_d)  # H/8
        de2 = self.gfusion2_d(enhanced_stage_d[0], out3_d)  # H/4
        out_d = self.fam32_d(de2, out3_d)  # H/4

        out1 = F.interpolate(self.linear_out(out4), size=raw_size, mode='bilinear')
        out2 = F.interpolate(self.linear_out(out3), size=raw_size, mode='bilinear')
        out = F.interpolate(self.linear_out(out), size=raw_size, mode='bilinear')
        out_d = F.interpolate(self.linear_outd(out_d), size=raw_size, mode='bilinear')

        alpha_V,alpha_D,alpha_fused = self.evi_head(out,out_d)

        return out, out1, out2, out_d,alpha_V,alpha_D,alpha_fused

    def forward(self, input,Depth_e):

        rgb = input[:, :3]
        modal_x = input[:, 3:]
        modal_x = torch.cat((modal_x, modal_x, modal_x), dim=1)

        out, out1, out2, out_d,alpha_V,alpha_D,alpha_fused = self.encode_decode(rgb,modal_x,Depth_e)

        return out, out1, out2, out_d,alpha_V,alpha_D,alpha_fused

    # def forward(self, input):
    #
    #     rgb = input[:, :3]
    #     modal_x = input[:, 3:4]
    #     modal_x = torch.cat((modal_x, modal_x, modal_x), dim=1)
    #     Depth_e = input[:, 4:7]
    #     out, out1, out2, out_d,alpha_V,alpha_D,alpha_fused = self.encode_decode(rgb,modal_x,Depth_e)
    #
    #     return out, out1, out2, out_d,alpha_V,alpha_D,alpha_fused

if __name__ == "__main__":
    model = Teacher().eval()
    with torch.no_grad():
        a = torch.rand(2, 3, 288, 512)
        b = torch.randn(2, 1, 288, 512)
        c = torch.randn(2, 3, 288, 512)
        images = torch.cat([a, b], dim=1)
        # images = torch.cat([images, c], dim=1)
        # out, out1, out2, loss = model(images)

        out, out1, out2, out_d,alpha_V,alpha_D,alpha_fused = model(images,c)
        print(alpha_fused.shape)

    # from util.util import compute_speed
    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #     net = Teacher().eval()
    #     flops, params = get_model_complexity_info(net, (7, 288, 512), as_strings=True, print_per_layer_stat=False)
    #     print('Flops: ' + flops)
    #     print('Params: ' + params)
    # #
    # compute_speed(net, input_size=(1, 7, 288, 512), iteration=500)