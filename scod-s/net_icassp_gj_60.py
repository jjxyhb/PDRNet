import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2,pvt_v2_b1,pvt_v2_b3,pvt_v2_b4
from timm.models.layers import DropPath, trunc_normal_
device = torch.device("cuda:0")
criterion=nn.CosineSimilarity(dim=1).to(device)
def weight_init(module):

    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Linear):#which can be replaced as kaiming_normal_()
            trunc_normal_(m.weight,std=.02)
            if isinstance(m,nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.Dropout) or isinstance(m,DropPath):
             m.p=0.00
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.GELU) or isinstance(m, nn.LeakyReLU) or isinstance(m,
                                                                                                           nn.AdaptiveAvgPool2d) or isinstance(
                m, nn.ReLU6) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.Softmax):
            pass
        elif isinstance(m, nn.ModuleList):
            weight_init(m)
        elif isinstance(m,nn.Upsample):
            pass
        elif isinstance(m,nn.AvgPool2d):
            pass
        # elif isinstance(m,nn.ConvTranspose2d):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        else:
            m.initialize()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)



BatchNorm2d=nn.BatchNorm2d
BN_MOMENTUM=0.1


class DynamicCoordinateAttention(nn.Module):
    """
    Dynamic Coordinate Attention (DCA)
    Innovation: Replaces static scaling factors with Content-Aware Dynamic Scaling.
    The network now looks at the global context to decide whether to prioritize 
    vertical or horizontal features for each image instance.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # -------------------------------------------------------------------------
        # Part 1: Coordinate Processing (Same as Original)
        # -------------------------------------------------------------------------
        mid_channels = max(8, in_channels // reduction)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        )

        # -------------------------------------------------------------------------
        # Part 2: Content-Aware Dynamic Scaling (The Innovation)
        # -------------------------------------------------------------------------
        # 这是一个迷你的 "Meta-Network"，用于生成动态权重
        # 输入: Global Context -> 输出: 2 * C 个权重 (C for H-direction, C for W-direction)
        
        scale_reduction = reduction # 保持与主路径一致的压缩比，节约计算量
        scale_mid_channels = max(8, in_channels // scale_reduction)
        
        self.context_generator = nn.Sequential(
            # 1. 压缩全局空间信息 (B, C, H, W) -> (B, C, 1, 1)
            nn.AdaptiveAvgPool2d(1),
            
            # 2. 降维
            nn.Conv2d(in_channels, scale_mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            
            # 3. 升维预测 2倍通道 (前C个给H方向，后C个给W方向)
            nn.Conv2d(scale_mid_channels, in_channels * 2, kernel_size=1, bias=False),
            
            # 4. Sigmoid 将权重归一化到 (0, 1) 区间，作为 Gating 机制
            nn.Sigmoid() 
        )

    def forward(self, x):
        """
        x: Tensor[B, C, H, W]
        """
        b, c, h, w = x.shape

        # --- Step 1: Coordinate Pooling (原逻辑) ---
        x_h = F.adaptive_avg_pool2d(x, (h, 1))  # (B, C, H, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, w))  # (B, C, 1, W)

        # --- Step 2: Shared MLP Transformation (原逻辑) ---
        mlp_h = self.mlp(x_h)  # (B, C, H, 1)
        mlp_w = self.mlp(x_w)  # (B, C, 1, W)

        # --- Step 3: Dynamic Scaling (创新逻辑) ---
        # 这一步替代了原本的: scaled_h = mlp_h * self.sfh
        
        # 生成动态权重 (B, 2C, 1, 1)
        scales = self.context_generator(x) 
        
        # 将权重拆分为 H 方向权重和 W 方向权重
        # scale_h: (B, C, 1, 1), scale_w: (B, C, 1, 1)
        scale_h, scale_w = torch.split(scales, self.in_channels, dim=1)

        # 应用动态权重 (Channel-wise multiplication)
        # 这里的广播机制：(B, C, H, 1) * (B, C, 1, 1) -> (B, C, H, 1)
        scaled_h = mlp_h * scale_h
        scaled_w = mlp_w * scale_w

        # --- Step 4: Fusion & Attention Generation (原逻辑) ---
        attn_h = scaled_h.expand(-1, -1, -1, w) # (B, C, H, W)
        attn_w = scaled_w.expand(-1, -1, h, -1) # (B, C, H, W)

        attn = torch.sigmoid(attn_h + attn_w)
        out = x * attn + x

        return out

    def initialize(self):
        # 初始化逻辑
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




from functools import partial
class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.cfg = cfg
        self.bkbone = pvt_v2_b4()
        #load_path='./pvt_v2_b2.pth'
        load_path='./pvt_v2_b4.pth'
        pretrained_dict=torch.load(load_path)
        pretrained_dict={k:v for k,v in pretrained_dict.items() if k in self.bkbone.state_dict()}
        self.bkbone.load_state_dict(pretrained_dict)

        self.extra = nn.ModuleList([

            conv3x3(64,64),
            conv3x3(128,64),
            conv3x3(320,64),
            conv3x3(512,64),
            # BasicConv2d(64*4,64,3,padding=1),
            # nn.ReLU(True),
        ])

        self.head = nn.ModuleList([
            conv3x3(64*4 , 1),
            # conv3x3(64, 1),
            # conv3x3(64, 1),
            # conv3x3(64, 1),
            # conv3x3(64, 1),
        ])
        """*****************"""
        self.relu = nn.ReLU(True)
        self.norm = nn.LayerNorm(512)#norm_layer(embed_dim)  # 这个得检查一遍前面有没有norm这个东西了，这个东西可以当后面的norm用

        self.cts_bn=nn.BatchNorm2d(1)

        # 修改后的代码
        self.aca_4 = DynamicCoordinateAttention(512)
        self.aca_3 = DynamicCoordinateAttention(320)
        self.aca_2 = DynamicCoordinateAttention(128)
        self.aca_1 = DynamicCoordinateAttention(64)

        self.initialize()

    def forward(self, x, shape=None, epoch=None):

        shape = x.size()[2:] if shape is None else shape
        attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2=self.bkbone(x)#bk_stage5=[8,512,6,6]
        #cls_token4,cls_token3,cls_token2,cls_token1,\
        #    attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2=self.bkbone(x)#bk_stage5=[8,512,6,6]

        F_4 = self.aca_4(bk_stage5)
        F_3 = self.aca_3(bk_stage4)
        F_2 = self.aca_2(bk_stage3)
        F_1 = self.aca_1(bk_stage2)




        F1 = self.extra[0](F_1)  # 3x3,c->64
        F2 = self.extra[1](F_2)
        F3 = self.extra[2](F_3)
        F4 = self.extra[3](F_4)

        # F4 = self.extra[0](bk_stage2)  # 1/4
        # F3 = self.extra[1](bk_stage3)  # 1/8
        # F2 = self.extra[2](bk_stage4)  # 1/16
        # F1 = self.extra[3](bk_stage5)  # 1/32

        f_1 = F1
        f_2 = F.interpolate(F2, size=f_1.size()[2:], mode='bilinear', align_corners=True)
        f_3 = F.interpolate(F3, size=f_1.size()[2:], mode='bilinear', align_corners=True)
        f_4 = F.interpolate(F4, size=f_1.size()[2:], mode='bilinear', align_corners=True)


        feature_map=torch.cat([f_1,f_2,f_3,f_4],dim=1)

        """"""
        out0 = F.interpolate(self.head[0](feature_map), size=shape, mode='bilinear', align_corners=False)
        #

        # [16,1,320,320]
        #out0=out0+(out1*out2+out3*out4)
        if self.cfg.mode == 'train':
            return out0, 0,out0,out0,out0,out0,0#,fg,bg
        else:
            return out0,attn_map

    def initialize(self):
        print('initialize net')
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot), strict=False)
        else:
            weight_init(self)