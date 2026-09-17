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
            # custom modules
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

        # start from zero offsets
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
# Main Net (PFGA removed)
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

        # unify channels to 64
        self.extra = nn.ModuleList([
            conv3x3(64, 64),    # stage2 -> F1
            conv3x3(128, 64),   # stage3 -> F2
            conv3x3(320, 64),   # stage4 -> F3
            conv3x3(512, 64),   # stage5 -> F4
        ])

        # DTAS unchanged
        self.dtas_3 = DTAS(64)
        self.dtas_2 = DTAS(64)
        self.dtas_1 = DTAS(64)

        # heads unchanged
        self.head = nn.ModuleList([
            conv1x1(64, 1),       # out4
            conv1x1(64, 1),       # out3
            conv1x1(64, 1),       # out2
            conv1x1(64, 1),       # out1
            conv3x3(64 * 4, 1)    # out0
        ])

        self.initialize()

    def forward(self, x, shape=None, epoch=None, return_gate=False):
        shape = x.size()[2:] if shape is None else shape

        # backbone outputs
        attn_map, bk_stage5, bk_stage4, bk_stage3, bk_stage2 = self.bkbone(x)

        # channel projection
        F1 = self.extra[0](bk_stage2)  # low-level
        F2 = self.extra[1](bk_stage3)
        F3 = self.extra[2](bk_stage4)
        F4 = self.extra[3](bk_stage5)  # deep semantic

        # deep coarse prediction
        f_4 = F.interpolate(F4, size=F1.size()[2:], mode='bilinear', align_corners=True)
        out4 = self.head[0](f_4)

        # -------- stage 3 refinement --------
        f_3 = F.interpolate(F3, size=F1.size()[2:], mode='bilinear', align_corners=True)

        # PFGA removed: directly use DTAS
        f_3_def = self.dtas_3(f_3, out4)
        out3 = self.head[1](f_3_def)

        # -------- stage 2 refinement --------
        f_2 = F.interpolate(F2, size=F1.size()[2:], mode='bilinear', align_corners=True)

        # PFGA removed: directly use DTAS
        f_2_def = self.dtas_2(f_2, out3)
        out2 = self.head[2](f_2_def)

        # -------- stage 1 refinement --------
        f_1 = self.dtas_1(F1, out2)
        out1 = self.head[3](f_1)

        # final fusion
        feature_map = torch.cat([f_1, f_2_def, f_3_def, f_4], dim=1)
        out0 = self.head[4](feature_map)

        # upsample to input size
        out0 = F.interpolate(out0, size=shape, mode='bilinear', align_corners=False)
        out1 = F.interpolate(out1, size=shape, mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=shape, mode='bilinear', align_corners=False)
        out3 = F.interpolate(out3, size=shape, mode='bilinear', align_corners=False)
        out4 = F.interpolate(out4, size=shape, mode='bilinear', align_corners=False)

        if self.cfg.mode == 'train':
            if return_gate:
                # keep output format consistent with previous version
                gate_dict = {
                    "alpha3": None,
                    "alpha2": None
                }
                return out0, out1, out2, out3, out4, F4, gate_dict
            else:
                return out0, out1, out2, out3, out4, F4, None
        else:
            return out0, attn_map

    def initialize(self):
        print('initialize net')
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot), strict=False)
        else:
            weight_init(self)