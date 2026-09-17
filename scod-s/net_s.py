import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

# --------- keep your weight_init as-is ---------
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Dropout) or isinstance(m, DropPath):
            m.p = 0.00
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.LeakyReLU) or isinstance(
                m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.ReLU6) or isinstance(m, nn.MaxPool2d) or isinstance(
                m, nn.Softmax):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        elif isinstance(m, nn.Upsample):
            pass
        elif isinstance(m, nn.AvgPool2d):
            pass
        else:
            # IMPORTANT: your init logic expects custom modules have initialize()
            m.initialize()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)




class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg

        self.bkbone = pvt_v2_b4()
        load_path = './pvt_v2_b4.pth'
        pretrained_dict = torch.load(load_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.bkbone.state_dict()}
        self.bkbone.load_state_dict(pretrained_dict)

        # unify channels to 64
        self.extra = nn.ModuleList([
            conv3x3(64,  64),
            conv3x3(128, 64),
            conv3x3(320, 64),
            conv3x3(512, 64),
        ])



        # head unchanged (still concat 4 features)
        self.head = nn.ModuleList([
            conv3x3(64 * 4, 1),
        ])

        self.initialize()

    def forward(self, x, shape=None, epoch=None):
        shape = x.size()[2:] if shape is None else shape

        # NOTE: your backbone returns: attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2
        attn_map, bk_stage5, bk_stage4, bk_stage3, bk_stage2 = self.bkbone(x)

        F1 = self.extra[0](bk_stage2)  # (B,64,H1,W1)
        F2 = self.extra[1](bk_stage3)  # (B,64,H2,W2)
        F3 = self.extra[2](bk_stage4)  # (B,64,H3,W3)
        F4 = self.extra[3](bk_stage5)  # (B,64,H4,W4)

        f1 = F1
        f2 = F.interpolate(F2, size=f1.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(F3, size=f1.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(F4, size=f1.size()[2:], mode='bilinear', align_corners=True)

        # ========= FPN-like concat + predict =========
        feature_map = torch.cat([f1, f2_fuse, f3_fuse, f4], dim=1)
        out0 = F.interpolate(self.head[0](feature_map), size=shape, mode='bilinear', align_corners=False)

        if self.cfg.mode == 'train':
            # keep your original output structure to avoid breaking trainer
            return out0, 0, out0, out0, out0, out0, f1
        else:
            return out0, attn_map

    def initialize(self):
        print('initialize net')
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot), strict=False)
        else:
            weight_init(self)
