import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from pvtv2 import pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4
from timm.models.layers import DropPath, trunc_normal_

device = torch.device("cuda:0")
criterion = nn.CosineSimilarity(dim=1).to(device)


# =========================
# Basic init utilities
# =========================
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Dropout) or isinstance(m, DropPath):
            m.p = 0.00
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(
            m,
            (
                nn.ReLU, nn.GELU, nn.LeakyReLU, nn.AdaptiveAvgPool2d,
                nn.ReLU6, nn.MaxPool2d, nn.Softmax, nn.Upsample, nn.AvgPool2d
            )
        ):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        else:
            if hasattr(m, "initialize"):
                m.initialize()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=padding, dilation=dilation, bias=bias
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride,
        padding=0, bias=bias
    )


# =========================
# DTAS
# =========================
class DTAS(nn.Module):
    def __init__(self, in_channel):
        super(DTAS, self).__init__()
        self.offset_conv = nn.Conv2d(in_channel + 1, 2 * 3 * 3, kernel_size=3, padding=1, bias=False)
        self.dcn = ops.DeformConv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1),
            nn.Sigmoid()
        )

        nn.init.constant_(self.offset_conv.weight, 0)

    def forward(self, x, coarse_mask):
        coarse_mask = F.interpolate(
            coarse_mask.detach(),
            size=x.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        offset = self.offset_conv(torch.cat([x, coarse_mask], dim=1))
        x_def = self.dcn(x, offset)
        return x_def * self.gate(x_def)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# =========================
# Mask-aware PFGA
# =========================
class MaskAwarePFGA(nn.Module):
    class Branch(nn.Module):
        def __init__(self, dim: int, K: int, center_suppress: bool = True):
            super().__init__()
            self.center_suppress = center_suppress

            self.dw_h = nn.Conv2d(
                dim, dim, kernel_size=(1, K),
                padding=(0, K // 2), groups=dim, bias=False
            )
            self.dw_v = nn.Conv2d(
                dim, dim, kernel_size=(K, 1),
                padding=(K // 2, 0), groups=dim, bias=False
            )

            if self.center_suppress:
                self.dw_c = nn.Conv2d(
                    dim, dim, kernel_size=3, padding=1,
                    groups=dim, bias=False
                )
                self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
            else:
                self.dw_c = None
                self.register_parameter("beta", None)

        def forward(self, x):
            y = self.dw_v(self.dw_h(x))
            if self.center_suppress:
                center = self.dw_c(x)
                y = y - torch.tanh(self.beta) * center
            return y

        def initialize(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def __init__(
        self,
        dim: int,
        K_list=(9, 15, 31),
        center_suppress: bool = True,
        gate_use_mask: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.K_list = K_list
        self.gate_use_mask = gate_use_mask

        self.branches = nn.ModuleList([
            MaskAwarePFGA.Branch(dim, K, center_suppress=center_suppress)
            for K in K_list
        ])

        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        laplace = torch.tensor(
            [[0,  1, 0],
             [1, -4, 1],
             [0,  1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)
        self.register_buffer("laplace", laplace, persistent=False)

        gate_in_ch = 4 if gate_use_mask else 3
        self.gate_head = nn.Conv2d(gate_in_ch, len(K_list), kernel_size=1, bias=True)

        self.res_scale = nn.Parameter(torch.tensor(0.0))

    def _depthwise_filter(self, x, k):
        B, C, H, W = x.shape
        w = k.repeat(C, 1, 1, 1)
        return F.conv2d(x, w, padding=1, groups=C)

    def _freq_maps(self, x):
        gx = self._depthwise_filter(x, self.sobel_x)
        gy = self._depthwise_filter(x, self.sobel_y)
        lap = self._depthwise_filter(x, self.laplace)

        grad_mag = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

        mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        mean2 = F.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1)
        var = torch.clamp(mean2 - mean * mean, min=0.0)

        f1 = grad_mag.mean(dim=1, keepdim=True)
        f2 = lap.abs().mean(dim=1, keepdim=True)
        f3 = var.mean(dim=1, keepdim=True)

        return torch.cat([f1, f2, f3], dim=1)

    def forward(self, x, coarse_mask=None, return_gate=False):
        peris = [b(x) for b in self.branches]
        freq = self._freq_maps(x)

        if self.gate_use_mask and coarse_mask is not None:
            cm = F.interpolate(
                coarse_mask.detach(),
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            cm = torch.sigmoid(cm)
            gate_in = torch.cat([freq, cm], dim=1)
        else:
            gate_in = freq

        logits = self.gate_head(gate_in)
        alpha = torch.softmax(logits, dim=1)

        y = 0.0
        for i, peri in enumerate(peris):
            y = y + peri * alpha[:, i:i + 1, :, :]

        out = x + self.res_scale.tanh() * y

        if return_gate:
            return out, alpha
        return out

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# =========================
# Boundary head (BCAP)
# =========================
class BoundaryHead(nn.Module):
    def __init__(self, in_ch=128, mid_ch=64):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, mid_ch),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            conv3x3(mid_ch, mid_ch),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            conv1x1(mid_ch, 1)
        )

    def forward(self, x):
        return self.block(x)

    def initialize(self):
        weight_init(self)


# =========================
# Confidence head (SPCE)
# =========================
class ConfidenceHead(nn.Module):
    """
    3-channel confidence field:
      channel 0 -> reliable foreground
      channel 1 -> reliable background
      channel 2 -> uncertain
    """
    def __init__(self, in_ch=69, mid_ch=64):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, mid_ch),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            conv3x3(mid_ch, mid_ch),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            conv1x1(mid_ch, 3)
        )

    def forward(self, x):
        return self.block(x)

    def initialize(self):
        weight_init(self)


# =========================
# Main Net
# =========================
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        # backbone
        self.bkbone = pvt_v2_b4()
        load_path = './pvt_v2_b4.pth'
        pretrained_dict = torch.load(load_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.bkbone.state_dict()}
        self.bkbone.load_state_dict(pretrained_dict)

        # unify channels
        self.extra = nn.ModuleList([
            conv3x3(64, 64),    # stage2 -> F1
            conv3x3(128, 64),   # stage3 -> F2
            conv3x3(320, 64),   # stage4 -> F3
            conv3x3(512, 64),   # stage5 -> F4
        ])

        # PFGA
        self.pfga_3 = MaskAwarePFGA(dim=64, K_list=(9, 15, 31), center_suppress=True, gate_use_mask=True)
        self.pfga_2 = MaskAwarePFGA(dim=64, K_list=(9, 15, 31), center_suppress=True, gate_use_mask=True)

        # DTAS
        self.dtas_3 = DTAS(64)
        self.dtas_2 = DTAS(64)
        self.dtas_1 = DTAS(64)

        # segmentation heads
        self.head = nn.ModuleList([
            conv1x1(64, 1),       # out4
            conv1x1(64, 1),       # out3
            conv1x1(64, 1),       # out2
            conv1x1(64, 1),       # out1
            conv3x3(64 * 4, 1)    # out0
        ])

        # new heads
        self.boundary_head = BoundaryHead(in_ch=128, mid_ch=64)
        self.confidence_head = ConfidenceHead(in_ch=69, mid_ch=64)

        self.initialize()

    def forward(self, x, shape=None, epoch=None, return_gate=False):
        shape = x.size()[2:] if shape is None else shape

        attn_map, bk_stage5, bk_stage4, bk_stage3, bk_stage2 = self.bkbone(x)

        F1 = self.extra[0](bk_stage2)
        F2 = self.extra[1](bk_stage3)
        F3 = self.extra[2](bk_stage4)
        F4 = self.extra[3](bk_stage5)

        # coarse prediction
        f_4 = F.interpolate(F4, size=F1.size()[2:], mode='bilinear', align_corners=True)
        out4 = self.head[0](f_4)

        # stage 3
        f_3 = F.interpolate(F3, size=F1.size()[2:], mode='bilinear', align_corners=True)
        if return_gate:
            f_3, alpha3 = self.pfga_3(f_3, out4, return_gate=True)
        else:
            f_3 = self.pfga_3(f_3, out4)
        f_3_def = self.dtas_3(f_3, out4)
        out3 = self.head[1](f_3_def)

        # stage 2
        f_2 = F.interpolate(F2, size=F1.size()[2:], mode='bilinear', align_corners=True)
        if return_gate:
            f_2, alpha2 = self.pfga_2(f_2, out3, return_gate=True)
        else:
            f_2 = self.pfga_2(f_2, out3)
        f_2_def = self.dtas_2(f_2, out3)
        out2 = self.head[2](f_2_def)

        # stage 1
        f_1 = self.dtas_1(F1, out2)
        out1 = self.head[3](f_1)

        # final fusion
        feature_map = torch.cat([f_1, f_2_def, f_3_def, f_4], dim=1)
        out0 = self.head[4](feature_map)

        # =========================
        # New innovation branches
        # =========================

        # deep semantic-structural feature for propagation
        deep_mix = F3 + F.interpolate(F4, size=F3.shape[2:], mode='bilinear', align_corners=True)

        # boundary head
        f2_small = F.interpolate(f_2_def, size=F3.shape[2:], mode='bilinear', align_corners=True)
        f3_small = F.interpolate(f_3_def, size=F3.shape[2:], mode='bilinear', align_corners=True)
        boundary_logit = self.boundary_head(torch.cat([f2_small, f3_small], dim=1))

        # confidence head
        o1_small = F.interpolate(out1, size=deep_mix.shape[2:], mode='bilinear', align_corners=True)
        o2_small = F.interpolate(out2, size=deep_mix.shape[2:], mode='bilinear', align_corners=True)
        o3_small = F.interpolate(out3, size=deep_mix.shape[2:], mode='bilinear', align_corners=True)
        o4_small = F.interpolate(out4, size=deep_mix.shape[2:], mode='bilinear', align_corners=True)

        conf_input = torch.cat([
            deep_mix,
            torch.sigmoid(boundary_logit),
            o1_small, o2_small, o3_small, o4_small
        ], dim=1)
        conf_logits = self.confidence_head(conf_input)

        # upsample segmentation outputs to input size
        out0 = F.interpolate(out0, size=shape, mode='bilinear', align_corners=False)
        out1 = F.interpolate(out1, size=shape, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=shape, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=shape, mode='bilinear', align_corners=False)
        out4 = F.interpolate(out4, size=shape, mode='bilinear', align_corners=False)

        if self.cfg.mode == 'train':
            aux_dict = {
                "boundary_logit": boundary_logit,   # low-resolution boundary
                "conf_logits": conf_logits,         # low-resolution 3-class confidence
            }
            if return_gate:
                aux_dict["alpha3"] = alpha3
                aux_dict["alpha2"] = alpha2

            return out0, out1, out2, out3, out4, deep_mix, aux_dict
        else:
            return out0, attn_map

    def initialize(self):
        print('initialize net')
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot), strict=False)
        else:
            weight_init(self)