import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # stage 3: remove PFGA and DTAS
        f_3 = F.interpolate(F3, size=F1.size()[2:], mode='bilinear', align_corners=True)
        out3 = self.head[1](f_3)

        # stage 2: remove PFGA and DTAS
        f_2 = F.interpolate(F2, size=F1.size()[2:], mode='bilinear', align_corners=True)
        out2 = self.head[2](f_2)

        # stage 1: remove DTAS
        f_1 = F1
        out1 = self.head[3](f_1)

        # final fusion
        feature_map = torch.cat([f_1, f_2, f_3, f_4], dim=1)
        out0 = self.head[4](feature_map)

        # =========================
        # New innovation branches
        # =========================

        # deep semantic-structural feature for propagation
        deep_mix = F3 + F.interpolate(F4, size=F3.shape[2:], mode='bilinear', align_corners=True)

        # boundary head
        f2_small = F.interpolate(f_2, size=F3.shape[2:], mode='bilinear', align_corners=True)
        f3_small = F.interpolate(f_3, size=F3.shape[2:], mode='bilinear', align_corners=True)
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
            return out0, out1, out2, out3, out4, deep_mix, aux_dict
        else:
            return out0, attn_map

    def initialize(self):
        print('initialize net')
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot), strict=False)
        else:
            weight_init(self)