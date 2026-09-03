import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ================== MBRConv 可重参数化卷积 ==================
class MBRConv5(nn.Module):
    """5x5 多分支可重参数化卷积"""
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv5, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 5, 1, 2)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv2 = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv2_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 10, out_channels, 1)

    def forward(self, inp):
        x1 = self.conv(inp)
        x2 = self.conv1(inp)
        x3 = self.conv2(inp)
        x4 = self.conv_crossh(inp)
        x5 = self.conv_crossv(inp)
        x = torch.cat([x1, x2, x3, x4, x5,
                       self.conv_bn(x1), self.conv1_bn(x2), self.conv2_bn(x3),
                       self.conv_crossh_bn(x4), self.conv_crossv_bn(x5)], 1)
        out = self.conv_out(x)
        return out

    def slim(self):
        """推理时融合为单个5x5卷积"""
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias
        conv1_weight = F.pad(self.conv1.weight, (2, 2, 2, 2))
        conv1_bias = self.conv1.bias
        conv2_weight = F.pad(self.conv2.weight, (1, 1, 1, 1))
        conv2_bias = self.conv2.bias
        conv_crossv_weight = F.pad(self.conv_crossv.weight, (1, 1, 2, 2))
        conv_crossv_bias = self.conv_crossv.bias
        conv_crossh_weight = F.pad(self.conv_crossh.weight, (2, 2, 1, 1))
        conv_crossh_bias = self.conv_crossh.bias

        # BN融合
        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = -bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        bn = self.conv1_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = -bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv1_bn_weight = F.pad(self.conv1.weight, (2, 2, 2, 2))
        conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_bias = self.conv1.bias * k + b
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        bn = self.conv2_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = -bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv2_bn_weight = F.pad(self.conv2.weight, (1, 1, 1, 1))
        conv2_bn_weight = conv2_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_weight = conv2_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv2_bn_bias = self.conv2.bias * k + b
        conv2_bn_bias = conv2_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = -bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossv_bn_weight = F.pad(self.conv_crossv.weight, (1, 1, 2, 2))
        conv_crossv_bn_weight = conv_crossv_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_bias = self.conv_crossv.bias * k + b
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossh_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = -bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_crossh_bn_weight = F.pad(self.conv_crossh.weight, (2, 2, 1, 1))
        conv_crossh_bn_weight = conv_crossh_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_bias = self.conv_crossh.bias * k + b
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        weight = torch.cat([conv_weight, conv1_weight, conv2_weight,
                           conv_crossh_weight, conv_crossv_weight,
                           conv_bn_weight, conv1_bn_weight, conv2_bn_weight,
                           conv_crossh_bn_weight, conv_crossv_bn_weight], 0)
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])
        bias_ = torch.cat([conv_bias, conv1_bias, conv2_bias,
                          conv_crossh_bias, conv_crossv_bias,
                          conv_bn_bias, conv1_bn_bias, conv2_bn_bias,
                          conv_crossh_bn_bias, conv_crossv_bn_bias], 0)
        bias = torch.matmul(weight_compress, bias_)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias

class MBRConv3(nn.Module):
    """3x3 多分支可重参数化卷积"""
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 3, 1, 1)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv1 = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv1_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossh = nn.Conv2d(in_channels, out_channels * rep_scale, (3, 1), 1, (1, 0))
        self.conv_crossh_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_crossv = nn.Conv2d(in_channels, out_channels * rep_scale, (1, 3), 1, (0, 1))
        self.conv_crossv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 8, out_channels, 1)

    def forward(self, inp):
        x0 = self.conv(inp)
        x1 = self.conv1(inp)
        x2 = self.conv_crossh(inp)
        x3 = self.conv_crossv(inp)
        x = torch.cat([x0, x1, x2, x3,
                       self.conv_bn(x0), self.conv1_bn(x1),
                       self.conv_crossh_bn(x2), self.conv_crossv_bn(x3)], 1)
        out = self.conv_out(x)
        return out

    def slim(self):
        """推理时融合为单个3x3卷积"""
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias
        conv1_weight = F.pad(self.conv1.weight, (1, 1, 1, 1))
        conv1_bias = self.conv1.bias
        conv_crossh_weight = F.pad(self.conv_crossh.weight, (1, 1, 0, 0))
        conv_crossh_bias = self.conv_crossh.bias
        conv_crossv_weight = F.pad(self.conv_crossv.weight, (0, 0, 1, 1))
        conv_crossv_bias = self.conv_crossv.bias

        # BN融合
        bn = self.conv_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + (-bn.running_mean * k)
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        bn = self.conv1_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv1_bn_weight = F.pad(self.conv1.weight, (1, 1, 1, 1))
        conv1_bn_weight = conv1_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_weight = conv1_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv1_bn_bias = self.conv1.bias * k + (-bn.running_mean * k)
        conv1_bn_bias = conv1_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossh_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossh_bn_weight = F.pad(self.conv_crossh.weight, (1, 1, 0, 0))
        conv_crossh_bn_weight = conv_crossh_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_weight = conv_crossh_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossh_bn_bias = self.conv_crossh.bias * k + (-bn.running_mean * k)
        conv_crossh_bn_bias = conv_crossh_bn_bias * bn.weight + bn.bias

        bn = self.conv_crossv_bn[0]
        k = 1 / torch.sqrt(bn.running_var + bn.eps)
        conv_crossv_bn_weight = F.pad(self.conv_crossv.weight, (0, 0, 1, 1))
        conv_crossv_bn_weight = conv_crossv_bn_weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_weight = conv_crossv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_crossv_bn_bias = self.conv_crossv.bias * k + (-bn.running_mean * k)
        conv_crossv_bn_bias = conv_crossv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([conv_weight, conv1_weight, conv_crossh_weight, conv_crossv_weight,
                           conv_bn_weight, conv1_bn_weight, conv_crossh_bn_weight, conv_crossv_bn_weight], dim=0)
        bias = torch.cat([conv_bias, conv1_bias, conv_crossh_bias, conv_crossv_bias,
                         conv_bn_bias, conv1_bn_bias, conv_crossh_bn_bias, conv_crossv_bn_bias], dim=0)

        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.view(weight.size(0), -1))
        weight = weight.view(self.out_channels, self.in_channels, 3, 3)
        bias = torch.matmul(weight_compress, bias.unsqueeze(-1)).squeeze(-1)
        if self.conv_out.bias is not None:
            bias += self.conv_out.bias
        return weight, bias

class MBRConv1(nn.Module):
    """1x1 多分支可重参数化卷积"""
    def __init__(self, in_channels, out_channels, rep_scale=4):
        super(MBRConv1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rep_scale = rep_scale

        self.conv = nn.Conv2d(in_channels, out_channels * rep_scale, 1)
        self.conv_bn = nn.Sequential(nn.BatchNorm2d(out_channels * rep_scale))
        self.conv_out = nn.Conv2d(out_channels * rep_scale * 2, out_channels, 1)

    def forward(self, inp):
        x0 = self.conv(inp)
        x = torch.cat([x0, self.conv_bn(x0)], 1)
        out = self.conv_out(x)
        return out

    def slim(self):
        """推理时融合为单个1x1卷积"""
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        bn = self.conv_bn[0]
        k = 1 / (bn.running_var + bn.eps) ** .5
        b = -bn.running_mean / (bn.running_var + bn.eps) ** .5
        conv_bn_weight = self.conv.weight * k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_weight = conv_bn_weight * bn.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        conv_bn_bias = self.conv.bias * k + b
        conv_bn_bias = conv_bn_bias * bn.weight + bn.bias

        weight = torch.cat([conv_weight, conv_bn_weight], 0)
        weight_compress = self.conv_out.weight.squeeze()
        weight = torch.matmul(weight_compress, weight.permute([2, 3, 0, 1])).permute([2, 3, 0, 1])
        bias = torch.cat([conv_bias, conv_bn_bias], 0)
        bias = torch.matmul(weight_compress, bias)
        if isinstance(self.conv_out.bias, torch.Tensor):
            bias = bias + self.conv_out.bias
        return weight, bias


# ================== FST 特征自变换模块 ==================
class FST(nn.Module):
    """Feature Self-Transformation (训练版)"""
    def __init__(self, block, channels):
        super(FST, self).__init__()
        self.block = block
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = self.block(x)
        return self.weight1 * x1 * self.weight2 + self.bias

class FSTS(nn.Module):
    """Feature Self-Transformation (推理版，融合后)"""
    def __init__(self, block, channels):
        super(FSTS, self).__init__()
        self.block = block
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = self.block(x)
        return self.weight1 * x1 * self.weight2 + self.bias

class FSTR(nn.Module):
    """Feature Self-Transformation with Residual (训练版，带残差连接)"""
    def __init__(self, block, channels):
        super(FSTR, self).__init__()
        self.block = block
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = self.block(x)
        return x + self.weight1 * x1 * self.weight2 + self.bias  # 残差连接

class FSTRS(nn.Module):
    """Feature Self-Transformation with Residual (推理版，带残差连接)"""
    def __init__(self, block, channels):
        super(FSTRS, self).__init__()
        self.block = block
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = self.block(x)
        return x + self.weight1 * x1 * self.weight2 + self.bias  # 残差连接


# ================== 下采样/上采样模块 ==================
class Downsample(nn.Module):
    """下采样模块 - 步长2的卷积 + 平均池化"""
    def __init__(self, in_ch, out_ch, rep_scale=4):
        super().__init__()
        self.conv = MBRConv3(in_ch, out_ch, rep_scale=rep_scale)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(x))

    def slim(self):
        return self.conv.slim()

class Upsample(nn.Module):
    """上采样模块 - 双线性插值 + 卷积"""
    def __init__(self, in_ch, out_ch, rep_scale=4):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = MBRConv3(in_ch, out_ch, rep_scale=rep_scale)

    def forward(self, x):
        return self.conv(self.up(x))

    def slim(self):
        return self.conv.slim()


# ================== S0 分支 (多级U-Net) ==================
class S0BranchMultiScale(nn.Module):
    """
    S0分支 - 多级U-Net结构，skip connection带通道注意力
    分辨率: H×W → H/2×W/2 → H/4×W/4
    通道: base_ch → 2× → 4× → 2× → base_ch
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=8, rep_scale=4):
        super().__init__()
        self.base_ch = base_ch

        # Encoder
        self.enc0 = FST(
            nn.Sequential(
                MBRConv5(in_ch, base_ch, rep_scale=rep_scale),
                nn.PReLU(base_ch)
            ),
            base_ch
        )
        self.enc1 = Downsample(base_ch, base_ch * 2, rep_scale)
        self.enc2 = Downsample(base_ch * 2, base_ch * 4, rep_scale)

        # Bottleneck
        self.bottleneck = FST(
            nn.Sequential(
                MBRConv3(base_ch * 4, base_ch * 4, rep_scale=rep_scale),
                nn.PReLU(base_ch * 4),
                MBRConv3(base_ch * 4, base_ch * 4, rep_scale=rep_scale)
            ),
            base_ch * 4
        )

        # Skip connection 通道注意力
        self.skip_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(base_ch * 2, base_ch * 2, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.skip_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(base_ch, base_ch, rep_scale=rep_scale),
            nn.Sigmoid()
        )

        # Decoder
        self.dec2 = Upsample(base_ch * 4, base_ch * 2, rep_scale)
        self.dec1 = Upsample(base_ch * 2, base_ch, rep_scale)

        # Skip connections
        self.skip2 = MBRConv3(base_ch * 4, base_ch * 2, rep_scale)
        self.skip1 = MBRConv3(base_ch * 2, base_ch, rep_scale)

        # Output tail
        self.tail = MBRConv3(base_ch, out_ch, rep_scale)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)          # [B, base_ch, H, W]
        e1 = self.enc1(e0)         # [B, 2×ch, H/2, W/2]
        e2 = self.enc2(e1)         # [B, 4×ch, H/4, W/4]

        # Bottleneck
        b = self.bottleneck(e2)    # [B, 4×ch, H/4, W/4]

        # Decoder with skip connections (带注意力)
        d2 = self.dec2(b)          # [B, 2×ch, H/2, W/2]
        e1_att = e1 * self.skip_att2(e1)  # encoder特征加权
        d2 = self.skip2(torch.cat([d2, e1_att], dim=1))

        d1 = self.dec1(d2)         # [B, base_ch, H, W]
        e0_att = e0 * self.skip_att1(e0)  # encoder特征加权
        d1 = self.skip1(torch.cat([d1, e0_att], dim=1))

        # Output
        out = self.tail(d1)
        return out


# ================== 偏振分支 (多级U-Net) ==================
class PolarBranchMultiScale(nn.Module):
    """
    偏振分支 - 多级U-Net结构，支持每个阶段堆叠多个FST块
    分辨率: H×W → H/2×W/2 → H/4×W/4
    通道: base_ch → 2× → 4× → 2× → base_ch
    激活函数: SiLU (Swish) - 平滑非线性，梯度更稳定
    残差连接: 除enc0首个块外，所有FST块都带残差连接
    """
    def __init__(self, in_ch=4, out_ch=4, base_ch=8, rep_scale=4,
                 num_blocks=None):  # [enc0, enc1, enc2, bottleneck, dec2, dec1]
        super().__init__()
        self.base_ch = base_ch
        if num_blocks is None:
            num_blocks = [4, 4, 4, 4, 4, 4]
        self.num_blocks = num_blocks

        # ============ Encoder ============
        # enc0: 输入 in_ch -> base_ch，堆叠 num_blocks[0] 个块
        # 第一个块通道变换(in_ch->base_ch)，不能用残差
        # 后续块通道不变，使用FSTR带残差
        self.enc0 = nn.ModuleList()
        for i in range(num_blocks[0]):
            if i == 0:
                # 第一个块：通道变换，无残差
                self.enc0.append(FST(
                    nn.Sequential(
                        MBRConv5(in_ch, base_ch, rep_scale=rep_scale),
                        nn.SiLU()
                    ),
                    base_ch
                ))
            else:
                # 后续块：通道不变，带残差
                self.enc0.append(FSTR(
                    nn.Sequential(
                        MBRConv5(base_ch, base_ch, rep_scale=rep_scale),
                        nn.SiLU()
                    ),
                    base_ch
                ))

        self.enc1 = Downsample(base_ch, base_ch * 2, rep_scale)
        # enc1 后续堆叠 num_blocks[1] 个 FSTR (通道 base_ch*2，带残差)
        self.enc1_blocks = nn.ModuleList([
            FSTR(
                nn.Sequential(
                    MBRConv5(base_ch*2, base_ch*2, rep_scale=rep_scale),
                    nn.SiLU()
                ),
                base_ch*2
            ) for _ in range(num_blocks[1])
        ])

        self.enc2 = Downsample(base_ch * 2, base_ch * 4, rep_scale)
        self.enc2_blocks = nn.ModuleList([
            FSTR(
                nn.Sequential(
                    MBRConv5(base_ch*4, base_ch*4, rep_scale=rep_scale),
                    nn.SiLU()
                ),
                base_ch*4
            ) for _ in range(num_blocks[2])
        ])

        # ============ Bottleneck ============
        self.bottleneck_blocks = nn.ModuleList()
        for i in range(num_blocks[3]):
            self.bottleneck_blocks.append(FSTR(
                nn.Sequential(
                    MBRConv3(base_ch*4, base_ch*4, rep_scale=rep_scale),
                    nn.SiLU(),
                    MBRConv3(base_ch*4, base_ch*4, rep_scale=rep_scale)
                ),
                base_ch*4
            ))

        # ============ Skip connection 通道注意力 ============
        self.skip_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(base_ch * 2, base_ch * 2, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.skip_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            MBRConv1(base_ch, base_ch, rep_scale=rep_scale),
            nn.Sigmoid()
        )

        # ============ Decoder ============
        self.dec2 = Upsample(base_ch * 4, base_ch * 2, rep_scale)
        self.dec2_blocks = nn.ModuleList([
            FSTR(
                nn.Sequential(
                    MBRConv5(base_ch*2, base_ch*2, rep_scale=rep_scale),
                    nn.SiLU()
                ),
                base_ch*2
            ) for _ in range(num_blocks[4])
        ])

        self.dec1 = Upsample(base_ch * 2, base_ch, rep_scale)
        self.dec1_blocks = nn.ModuleList([
            FSTR(
                nn.Sequential(
                    MBRConv5(base_ch, base_ch, rep_scale=rep_scale),
                    nn.SiLU()
                ),
                base_ch
            ) for _ in range(num_blocks[5])
        ])

        # Skip connections (保持原样)
        self.skip2 = MBRConv3(base_ch * 4, base_ch * 2, rep_scale)
        self.skip1 = MBRConv3(base_ch * 2, base_ch, rep_scale)

        # Output tail
        self.tail = MBRConv3(base_ch, out_ch, rep_scale)

    def forward(self, x):
        # ============ Encoder ============
        # enc0 堆叠
        e0 = x
        for block in self.enc0:
            e0 = block(e0)
        # e0 shape: [B, base_ch, H, W]

        e1 = self.enc1(e0)   # [B, 2*base_ch, H/2, W/2]
        for block in self.enc1_blocks:
            e1 = block(e1)

        e2 = self.enc2(e1)   # [B, 4*base_ch, H/4, W/4]
        for block in self.enc2_blocks:
            e2 = block(e2)

        # ============ Bottleneck ============
        b = e2
        for block in self.bottleneck_blocks:
            b = block(b)

        # ============ Decoder ============
        d2 = self.dec2(b)    # [B, 2*base_ch, H/2, W/2]
        e1_att = e1 * self.skip_att2(e1)
        d2 = self.skip2(torch.cat([d2, e1_att], dim=1))
        for block in self.dec2_blocks:
            d2 = block(d2)

        d1 = self.dec1(d2)   # [B, base_ch, H, W]
        e0_att = e0 * self.skip_att1(e0)
        d1 = self.skip1(torch.cat([d1, e0_att], dim=1))
        for block in self.dec1_blocks:
            d1 = block(d1)

        out = self.tail(d1)
        return out


# ================== 主模型  ==================
class LPDFlash(nn.Module):
    def __init__(self, in_chans_s0=3, in_chans_polar=4,
                 base_ch=8, rep_scale=8,
                 use_12ch_input=True,
                 iap_enabled=False,
                 iap_C=0.95,
                 iap_safety_margin=0.8,
                 iap_eps=1e-8,
                 iap_k=2):
        super().__init__()
        self.eps = 1e-8
        self.use_12ch_input = use_12ch_input

        # iap 参数
        self.iap_enabled = iap_enabled
        self.iap_C = iap_C
        self.iap_safety_margin = iap_safety_margin
        self.iap_eps = iap_eps
        self.iap_k = iap_k

        # 分支冻结标志
        self.freeze_s0 = False
        self.freeze_polar = False

        # S0 增强分支
        self.s0_branch = S0BranchMultiScale(
            in_ch=in_chans_s0,
            out_ch=in_chans_s0,
            base_ch=base_ch,
            rep_scale=rep_scale
        )

        # 偏振去噪分支
        self.polar_branch = PolarBranchMultiScale(
            in_ch=in_chans_polar,
            out_ch=in_chans_polar,
            base_ch=base_ch,
            rep_scale=rep_scale
        )

    # ================================================================
    # 新版的稳定 iap（替换原有的 apply_iap）
    # ================================================================
    def apply_iap_stable(self, s0_rgb, C, safety_margin):
        """
        IAP 预处理：使用 均值 + k*标准差 代替分位数（抗噪）
        """
        luminance = s0_rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # 计算稳定阈值 v
        mean = luminance.mean(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
        std = luminance.std(dim=(2, 3), keepdim=True)    # [B, 1, 1, 1]

        v = mean + self.iap_k * std
        v = torch.clamp(v, min=self.iap_eps)

        # 计算 gamma
        target_max = C * safety_margin
        gamma = target_max / v

        # 应用缩放
        s0_norm = torch.clamp(s0_rgb * gamma, 0.0, 1.0)

        return s0_norm, gamma

    # ================================================================
    # 原有函数（set_freeze, forward, get_final_output, slim, count_parameters）
    # 注意：forward 内部调用了新的 apply_iap_stable
    # ================================================================

    def set_freeze(self, branch=None):
        """
        设置分支冻结模式
        Args:
            branch: 's0' - 冻结S0分支
                   'polar' - 冻结偏振分支
                   None - 解冻所有
        """
        self.freeze_s0 = (branch == 's0')
        self.freeze_polar = (branch == 'polar')

        if self.freeze_s0:
            for param in self.s0_branch.parameters():
                param.requires_grad = False
            print("[MODEL] S0分支已冻结（跳过前向计算）")

        if self.freeze_polar:
            for param in self.polar_branch.parameters():
                param.requires_grad = False
            print("[MODEL] 偏振分支已冻结（跳过前向计算）")

        if branch is None:
            for param in self.parameters():
                param.requires_grad = True
            print("[MODEL] 所有分支已解冻")

    def forward(self, x):
        # 分离输入
        if isinstance(x, tuple):
            s0, polar = x
        else:
            if self.use_12ch_input:
                B, C, H, W = x.shape
                data = x.view(B, 4, 3, H, W)

                # ===== IAP 预处理 =====
                if self.iap_enabled:
                    s0 = data.mean(dim=1)
                    s0, gamma = self.apply_iap_stable(s0, self.iap_C,
                                                      self.iap_safety_margin)
                    # 直接在 data 上缩放，避免重复 view 操作
                    data = data * gamma.view(B, 1, 1, 1, 1)
                    data = torch.clamp(data, 0.0, 1.0)

                # ===== 从 data 分 s0 和 polar（只做一次） =====
                s0 = data.mean(dim=1)
                rgb_weights = torch.tensor([0.299, 0.587, 0.114],
                                           device=x.device).view(1, 1, 3, 1, 1)
                polar = (data * rgb_weights).sum(dim=2)
            else:
                s0 = x[:, 0:3, :, :]
                polar = x[:, 3:7, :, :]

                if self.iap_enabled:
                    s0, gamma = self.apply_iap_stable(s0, self.iap_C,
                                                      self.iap_safety_margin)

        # S0 分支 (冻结时跳过计算)
        if self.freeze_s0:
            s0_out = s0
        else:
            s0_res = self.s0_branch(s0)
            s0_out = torch.clamp(s0 + s0_res, 0.0, 1.0)

        # 偏振分支 (冻结时跳过计算)
        if self.freeze_polar:
            polar_out = polar
        else:
            polar_res = self.polar_branch(polar)
            polar_out = polar + polar_res

        return s0_out, polar_out

    def get_final_output(self, branch1_out, branch2_out):
        """从两个分支的输出计算 S0, DoLP, AoP"""
        s0 = branch1_out
        i0 = branch2_out[:, 0:1, :, :]
        i45 = branch2_out[:, 1:2, :, :]
        i90 = branch2_out[:, 2:3, :, :]
        i135 = branch2_out[:, 3:4, :, :]
        S0_gray = (i0 + i45 + i90 + i135) * 0.25
        S1 = (i0 - i90) * 0.5
        S2 = (i45 - i135) * 0.5
        dolp = torch.sqrt(S1**2 + S2**2 + self.eps) / (S0_gray + self.eps)
        dolp = torch.clamp(dolp, 0.0, 1.0)
        aop = 0.5 * torch.atan2(S2, S1 + self.eps)
        aop = aop % math.pi
        return {
            's0': s0,
            'dolp': dolp,
            'aop': aop,
            'combined': torch.cat([s0, dolp, aop], dim=1)
        }

    def slim(self):
        """转换为推理时的轻量版本 (所有MBRConv融合为普通卷积)"""
        net_slim = LPDFlashSlim(
            in_chans_s0=self.s0_branch.enc0.block[0].conv.in_channels,
            in_chans_polar=self.polar_branch.enc0[0].block[0].conv.in_channels,
            base_ch=self.s0_branch.base_ch,
            num_blocks=self.polar_branch.num_blocks,
            use_12ch_input=self.use_12ch_input,
            iap_enabled=self.iap_enabled,
            iap_C=self.iap_C,
            iap_safety_margin=self.iap_safety_margin,
            iap_eps=self.iap_eps,
            iap_k=self.iap_k
        )
        weight_slim = net_slim.state_dict()

        # ============ S0分支权重映射 (单FST块结构) ============
        weight_slim['s0_enc0.bias'] = self.s0_branch.enc0.bias.data
        weight_slim['s0_enc0.weight1'] = self.s0_branch.enc0.weight1.data
        weight_slim['s0_enc0.weight2'] = self.s0_branch.enc0.weight2.data
        w, b = self.s0_branch.enc0.block[0].slim()
        weight_slim['s0_enc0.block.0.weight'] = w
        weight_slim['s0_enc0.block.0.bias'] = b
        weight_slim['s0_enc0.block.1.weight'] = self.s0_branch.enc0.block[1].weight.data

        w, b = self.s0_branch.enc1.slim()
        weight_slim['s0_enc1.0.weight'] = w
        weight_slim['s0_enc1.0.bias'] = b
        w, b = self.s0_branch.enc2.slim()
        weight_slim['s0_enc2.0.weight'] = w
        weight_slim['s0_enc2.0.bias'] = b

        weight_slim['s0_bottleneck.bias'] = self.s0_branch.bottleneck.bias.data
        weight_slim['s0_bottleneck.weight1'] = self.s0_branch.bottleneck.weight1.data
        weight_slim['s0_bottleneck.weight2'] = self.s0_branch.bottleneck.weight2.data
        for i in range(3):
            if isinstance(self.s0_branch.bottleneck.block[i], MBRConv3):
                w, b = self.s0_branch.bottleneck.block[i].slim()
                weight_slim[f's0_bottleneck.block.{i}.weight'] = w
                weight_slim[f's0_bottleneck.block.{i}.bias'] = b
            elif isinstance(self.s0_branch.bottleneck.block[i], nn.PReLU):
                weight_slim[f's0_bottleneck.block.{i}.weight'] = self.s0_branch.bottleneck.block[i].weight.data

        w, b = self.s0_branch.skip_att2[1].slim()
        weight_slim['s0_skip_att2.1.weight'] = w
        weight_slim['s0_skip_att2.1.bias'] = b
        w, b = self.s0_branch.skip_att1[1].slim()
        weight_slim['s0_skip_att1.1.weight'] = w
        weight_slim['s0_skip_att1.1.bias'] = b

        w, b = self.s0_branch.dec2.slim()
        weight_slim['s0_dec2.1.weight'] = w
        weight_slim['s0_dec2.1.bias'] = b
        w, b = self.s0_branch.skip2.slim()
        weight_slim['s0_skip2.weight'] = w
        weight_slim['s0_skip2.bias'] = b
        w, b = self.s0_branch.dec1.slim()
        weight_slim['s0_dec1.1.weight'] = w
        weight_slim['s0_dec1.1.bias'] = b
        w, b = self.s0_branch.skip1.slim()
        weight_slim['s0_skip1.weight'] = w
        weight_slim['s0_skip1.bias'] = b
        w, b = self.s0_branch.tail.slim()
        weight_slim['s0_tail.weight'] = w
        weight_slim['s0_tail.bias'] = b

        # ============ 偏振分支权重映射 ============
        for i, fst_block in enumerate(self.polar_branch.enc0):
            weight_slim[f'polar_enc0.{i}.bias'] = fst_block.bias.data
            weight_slim[f'polar_enc0.{i}.weight1'] = fst_block.weight1.data
            weight_slim[f'polar_enc0.{i}.weight2'] = fst_block.weight2.data
            w, b = fst_block.block[0].slim()
            weight_slim[f'polar_enc0.{i}.block.0.weight'] = w
            weight_slim[f'polar_enc0.{i}.block.0.bias'] = b

        w, b = self.polar_branch.enc1.slim()
        weight_slim['polar_enc1.0.weight'] = w
        weight_slim['polar_enc1.0.bias'] = b
        for i, fst_block in enumerate(self.polar_branch.enc1_blocks):
            weight_slim[f'polar_enc1_blocks.{i}.bias'] = fst_block.bias.data
            weight_slim[f'polar_enc1_blocks.{i}.weight1'] = fst_block.weight1.data
            weight_slim[f'polar_enc1_blocks.{i}.weight2'] = fst_block.weight2.data
            w, b = fst_block.block[0].slim()
            weight_slim[f'polar_enc1_blocks.{i}.block.0.weight'] = w
            weight_slim[f'polar_enc1_blocks.{i}.block.0.bias'] = b

        w, b = self.polar_branch.enc2.slim()
        weight_slim['polar_enc2.0.weight'] = w
        weight_slim['polar_enc2.0.bias'] = b
        for i, fst_block in enumerate(self.polar_branch.enc2_blocks):
            weight_slim[f'polar_enc2_blocks.{i}.bias'] = fst_block.bias.data
            weight_slim[f'polar_enc2_blocks.{i}.weight1'] = fst_block.weight1.data
            weight_slim[f'polar_enc2_blocks.{i}.weight2'] = fst_block.weight2.data
            w, b = fst_block.block[0].slim()
            weight_slim[f'polar_enc2_blocks.{i}.block.0.weight'] = w
            weight_slim[f'polar_enc2_blocks.{i}.block.0.bias'] = b

        for i, fst_block in enumerate(self.polar_branch.bottleneck_blocks):
            weight_slim[f'polar_bottleneck_blocks.{i}.bias'] = fst_block.bias.data
            weight_slim[f'polar_bottleneck_blocks.{i}.weight1'] = fst_block.weight1.data
            weight_slim[f'polar_bottleneck_blocks.{i}.weight2'] = fst_block.weight2.data
            for j in range(3):
                if isinstance(fst_block.block[j], MBRConv3):
                    w, b = fst_block.block[j].slim()
                    weight_slim[f'polar_bottleneck_blocks.{i}.block.{j}.weight'] = w
                    weight_slim[f'polar_bottleneck_blocks.{i}.block.{j}.bias'] = b

        w, b = self.polar_branch.skip_att2[1].slim()
        weight_slim['polar_skip_att2.1.weight'] = w
        weight_slim['polar_skip_att2.1.bias'] = b
        w, b = self.polar_branch.skip_att1[1].slim()
        weight_slim['polar_skip_att1.1.weight'] = w
        weight_slim['polar_skip_att1.1.bias'] = b

        w, b = self.polar_branch.dec2.slim()
        weight_slim['polar_dec2.1.weight'] = w
        weight_slim['polar_dec2.1.bias'] = b
        w, b = self.polar_branch.skip2.slim()
        weight_slim['polar_skip2.weight'] = w
        weight_slim['polar_skip2.bias'] = b
        for i, fst_block in enumerate(self.polar_branch.dec2_blocks):
            weight_slim[f'polar_dec2_blocks.{i}.bias'] = fst_block.bias.data
            weight_slim[f'polar_dec2_blocks.{i}.weight1'] = fst_block.weight1.data
            weight_slim[f'polar_dec2_blocks.{i}.weight2'] = fst_block.weight2.data
            w, b = fst_block.block[0].slim()
            weight_slim[f'polar_dec2_blocks.{i}.block.0.weight'] = w
            weight_slim[f'polar_dec2_blocks.{i}.block.0.bias'] = b

        w, b = self.polar_branch.dec1.slim()
        weight_slim['polar_dec1.1.weight'] = w
        weight_slim['polar_dec1.1.bias'] = b
        w, b = self.polar_branch.skip1.slim()
        weight_slim['polar_skip1.weight'] = w
        weight_slim['polar_skip1.bias'] = b
        for i, fst_block in enumerate(self.polar_branch.dec1_blocks):
            weight_slim[f'polar_dec1_blocks.{i}.bias'] = fst_block.bias.data
            weight_slim[f'polar_dec1_blocks.{i}.weight1'] = fst_block.weight1.data
            weight_slim[f'polar_dec1_blocks.{i}.weight2'] = fst_block.weight2.data
            w, b = fst_block.block[0].slim()
            weight_slim[f'polar_dec1_blocks.{i}.block.0.weight'] = w
            weight_slim[f'polar_dec1_blocks.{i}.block.0.bias'] = b

        w, b = self.polar_branch.tail.slim()
        weight_slim['polar_tail.weight'] = w
        weight_slim['polar_tail.bias'] = b

        net_slim.load_state_dict(weight_slim)
        return net_slim

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'total_M': total / 1e6,
            'trainable_M': trainable / 1e6
        }


# ================== 推理用轻量版本  ==================
class LPDFlashSlim(nn.Module):
    def __init__(self, in_chans_s0=3, in_chans_polar=4, base_ch=8,
                 num_blocks=None, use_12ch_input=False,
                 iap_enabled=True, iap_C=0.95, iap_safety_margin=0.8, iap_eps=1e-8,
                 iap_k=2):
        super().__init__()
        self.eps = 1e-8
        self.base_ch = base_ch
        self.use_12ch_input = use_12ch_input
        # iap 参数
        self.iap_enabled = iap_enabled
        self.iap_C = iap_C
        self.iap_safety_margin = iap_safety_margin
        self.iap_eps = iap_eps
        self.iap_k = iap_k

        # ============ S0 分支（与原代码相同） ============
        self.s0_enc0 = FSTS(
            nn.Sequential(
                nn.Conv2d(in_chans_s0, base_ch, 5, 1, 2),
                nn.PReLU(base_ch)
            ),
            base_ch
        )
        self.s0_enc1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, 1, 1),
            nn.AvgPool2d(2)
        )
        self.s0_enc2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 1, 1),
            nn.AvgPool2d(2)
        )

        self.s0_bottleneck = FSTS(
            nn.Sequential(
                nn.Conv2d(base_ch * 4, base_ch * 4, 3, 1, 1),
                nn.PReLU(base_ch * 4),
                nn.Conv2d(base_ch * 4, base_ch * 4, 3, 1, 1)
            ),
            base_ch * 4
        )

        self.s0_skip_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch * 2, base_ch * 2, 1),
            nn.Sigmoid()
        )
        self.s0_skip_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch, base_ch, 1),
            nn.Sigmoid()
        )

        self.s0_dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, 1, 1)
        )
        self.s0_skip2 = nn.Conv2d(base_ch * 4, base_ch * 2, 3, 1, 1)

        self.s0_dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1)
        )
        self.s0_skip1 = nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1)

        self.s0_tail = nn.Conv2d(base_ch, in_chans_s0, 3, 1, 1)

        # ============ 偏振分支（与原代码相同） ============
        self.polar_enc0 = nn.ModuleList()
        for i in range(num_blocks[0]):
            if i == 0:
                self.polar_enc0.append(FSTS(
                    nn.Sequential(
                        nn.Conv2d(in_chans_polar, base_ch, 5, 1, 2),
                        nn.SiLU()
                    ),
                    base_ch
                ))
            else:
                self.polar_enc0.append(FSTRS(
                    nn.Sequential(
                        nn.Conv2d(base_ch, base_ch, 5, 1, 2),
                        nn.SiLU()
                    ),
                    base_ch
                ))

        self.polar_enc1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, 1, 1),
            nn.AvgPool2d(2)
        )
        self.polar_enc1_blocks = nn.ModuleList([
            FSTRS(
                nn.Sequential(
                    nn.Conv2d(base_ch * 2, base_ch * 2, 5, 1, 2),
                    nn.SiLU()
                ),
                base_ch * 2
            ) for _ in range(num_blocks[1])
        ])

        self.polar_enc2 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, 1, 1),
            nn.AvgPool2d(2)
        )
        self.polar_enc2_blocks = nn.ModuleList([
            FSTRS(
                nn.Sequential(
                    nn.Conv2d(base_ch * 4, base_ch * 4, 5, 1, 2),
                    nn.SiLU()
                ),
                base_ch * 4
            ) for _ in range(num_blocks[2])
        ])

        self.polar_bottleneck_blocks = nn.ModuleList([
            FSTRS(
                nn.Sequential(
                    nn.Conv2d(base_ch * 4, base_ch * 4, 3, 1, 1),
                    nn.SiLU(),
                    nn.Conv2d(base_ch * 4, base_ch * 4, 3, 1, 1)
                ),
                base_ch * 4
            ) for _ in range(num_blocks[3])
        ])

        self.polar_skip_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch * 2, base_ch * 2, 1),
            nn.Sigmoid()
        )
        self.polar_skip_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch, base_ch, 1),
            nn.Sigmoid()
        )

        self.polar_dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 4, base_ch * 2, 3, 1, 1)
        )
        self.polar_dec2_blocks = nn.ModuleList([
            FSTRS(
                nn.Sequential(
                    nn.Conv2d(base_ch * 2, base_ch * 2, 5, 1, 2),
                    nn.SiLU()
                ),
                base_ch * 2
            ) for _ in range(num_blocks[4])
        ])
        self.polar_skip2 = nn.Conv2d(base_ch * 4, base_ch * 2, 3, 1, 1)

        self.polar_dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1)
        )
        self.polar_dec1_blocks = nn.ModuleList([
            FSTRS(
                nn.Sequential(
                    nn.Conv2d(base_ch, base_ch, 5, 1, 2),
                    nn.SiLU()
                ),
                base_ch
            ) for _ in range(num_blocks[5])
        ])
        self.polar_skip1 = nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1)

        self.polar_tail = nn.Conv2d(base_ch, in_chans_polar, 3, 1, 1)

    # ===== IAP 预处理（与主类完全一致） =====
    def apply_iap_stable(self, s0_rgb, C, safety_margin):
        """
        与主类 LPDFlash.apply_iap_stable 完全相同
        """
        luminance = s0_rgb.mean(dim=1, keepdim=True)
        mean = luminance.mean(dim=(2, 3), keepdim=True)
        std = luminance.std(dim=(2, 3), keepdim=True)
        v = mean + self.iap_k * std
        v = torch.clamp(v, min=self.iap_eps)
        target_max = C * safety_margin
        gamma = target_max / v
        s0_norm = torch.clamp(s0_rgb * gamma, 0.0, 1.0)
        return s0_norm, gamma

    def forward(self, x):
        if isinstance(x, tuple):
            s0, polar = x
        else:
            if self.use_12ch_input:
                B, C, H, W = x.shape
                data = x.view(B, 4, 3, H, W)

                # IAP 预处理
                if self.iap_enabled:
                    s0 = data.mean(dim=1)
                    s0, gamma = self.apply_iap_stable(s0, self.iap_C,
                                                      self.iap_safety_margin)
                    # 直接在 data 上缩放，避免重复 view 操作
                    data = data * gamma.view(B, 1, 1, 1, 1)
                    data = torch.clamp(data, 0.0, 1.0)

                # 从 data 分 s0 和 polar（只做一次）
                s0 = data.mean(dim=1)
                rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 1, 3, 1, 1)
                polar = (data * rgb_weights).sum(dim=2)
            else:
                s0 = x[:, 0:3, :, :]
                polar = x[:, 3:7, :, :]
                if self.iap_enabled:
                    s0, gamma = self.apply_iap_stable(s0, self.iap_C,
                                                      self.iap_safety_margin)

        # ============ S0 分支 ============
        e0 = self.s0_enc0(s0)
        e1 = self.s0_enc1(e0)
        e2 = self.s0_enc2(e1)

        b = self.s0_bottleneck(e2)

        d2 = self.s0_dec2(b)
        e1_att = e1 * self.s0_skip_att2(e1)
        d2 = self.s0_skip2(torch.cat([d2, e1_att], dim=1))

        d1 = self.s0_dec1(d2)
        e0_att = e0 * self.s0_skip_att1(e0)
        d1 = self.s0_skip1(torch.cat([d1, e0_att], dim=1))

        s0_out = s0 + self.s0_tail(d1)
        s0_out = torch.clamp(s0_out, 0.0, 1.0)

        # ============ 偏振分支 ============
        e0 = polar
        for block in self.polar_enc0:
            e0 = block(e0)

        e1 = self.polar_enc1(e0)
        for block in self.polar_enc1_blocks:
            e1 = block(e1)

        e2 = self.polar_enc2(e1)
        for block in self.polar_enc2_blocks:
            e2 = block(e2)

        b = e2
        for block in self.polar_bottleneck_blocks:
            b = block(b)

        d2 = self.polar_dec2(b)
        e1_att = e1 * self.polar_skip_att2(e1)
        d2 = self.polar_skip2(torch.cat([d2, e1_att], dim=1))
        for block in self.polar_dec2_blocks:
            d2 = block(d2)

        d1 = self.polar_dec1(d2)
        e0_att = e0 * self.polar_skip_att1(e0)
        d1 = self.polar_skip1(torch.cat([d1, e0_att], dim=1))
        for block in self.polar_dec1_blocks:
            d1 = block(d1)

        polar_out = polar + self.polar_tail(d1)

        return s0_out, polar_out

    def get_final_output(self, branch1_out, branch2_out):
        s0 = branch1_out
        i0 = branch2_out[:, 0:1, :, :]
        i45 = branch2_out[:, 1:2, :, :]
        i90 = branch2_out[:, 2:3, :, :]
        i135 = branch2_out[:, 3:4, :, :]
        s0_gray = (i0 + i45 + i90 + i135) * 0.25
        s1 = (i0 - i90) * 0.5
        s2 = (i45 - i135) * 0.5
        dolp = torch.sqrt(s1**2 + s2**2 + self.eps) / (s0_gray + self.eps)
        aop = 0.5 * torch.atan2(s2, s1 + self.eps)
        aop = aop % math.pi
        return {
            's0': s0,
            'dolp': dolp,
            'aop': aop,
            'combined': torch.cat([s0, dolp, aop], dim=1)
        }
