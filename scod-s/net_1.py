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
        else:
            m.initialize()

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

class BasicConv2d(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,dilation=1):
        super(BasicConv2d,self).__init__()
        self.conv=nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,
                            stride=stride,padding=padding,dilation=dilation,bias=False)
        self.bn=nn.BatchNorm2d(out_planes)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        return x
    def initialize(self):
        weight_init(self)
############################################# Pooling ##############################################
class basicConv(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(basicConv, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
            # conv.append(nn.LayerNorm(out_channel, eps=1e-6))
        if relu:
            conv.append(nn.GELU())
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

    def initialize(self):
        weight_init(self)



BatchNorm2d=nn.BatchNorm2d
BN_MOMENTUM=0.1







class PyramidPooling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(PyramidPooling, self).__init__()
        hidden_channel = int(in_channel / 4)
        self.conv1 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv2 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv3 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.conv4 = basicConv(in_channel, hidden_channel, k=1, s=1, p=0)
        self.out = basicConv(in_channel * 2, out_channel, k=1, s=1, p=0)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(F.adaptive_avg_pool2d(x, 1)), size)
        feat2 = F.interpolate(self.conv2(F.adaptive_avg_pool2d(x, 2)), size)
        feat3 = F.interpolate(self.conv3(F.adaptive_avg_pool2d(x, 3)), size)
        feat4 = F.interpolate(self.conv4(F.adaptive_avg_pool2d(x, 4)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)

        return x

    def initialize(self):
        weight_init(self)







class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class SFTA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SFTA, self).__init__()
        self.down_channel = nn.Sequential(
            conv3x3(in_channel, out_channel, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            conv3x3(out_channel * 2, out_channel, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.TA = MSA_head(dim=out_channel)
    def forward(self, x, y):
        _, _, H1, _ = x.size()
        _, _, H2, _ = y.size()
        if H1 != H2:
            y = self.up(y)
        x = self.down_channel(x)
        out = self.fuse(torch.cat((x, y), dim=1))
        out = self.TA(out)
        return out

    def initialize(self):
        weight_init(self)
class MSA_head(nn.Module):
    def __init__(self, mode='dilation',dim=128, num_heads=8, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)



class OutPut(nn.Module):
    def __init__(self, in_chs, scale=1):
        super(OutPut, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_chs, in_chs, 1, bias=False),
                                 nn.BatchNorm2d(in_chs),
                                 nn.ReLU(inplace=True),
                                 nn.UpsamplingBilinear2d(scale_factor=scale),
                                 nn.Conv2d(in_chs, 1, 1),
                                 nn.Sigmoid())

    def forward(self, feat):
        return self.out(feat)

    def initialize(self):
        weight_init(self)















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

        self.initialize()


    def forward(self, x, shape=None, epoch=None):

        shape = x.size()[2:] if shape is None else shape
        attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2=self.bkbone(x)#bk_stage5=[8,512,6,6]Fcuda
        #cls_token4,cls_token3,cls_token2,cls_token1,\
        #    attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2=self.bkbone(x)#bk_stage5=[8,512,6,6]

        F1 = self.extra[0](bk_stage2)  # 3x3,c->64
        F2 = self.extra[1](bk_stage3)
        F3 = self.extra[2](bk_stage4)
        F4 = self.extra[3](bk_stage5)

        # F4 = self.extra[0](bk_stage2)  # 1/4
        # F3 = self.extra[1](bk_stage3)  # 1/8
        # F2 = self.extra[2](bk_stage4)  # 1/16
        # F1 = self.extra[3](bk_stage5)  # 1/32

        f_1 = F1
        f_2 = F.interpolate(F2, size=f_1.size()[2:], mode='bilinear', align_corners=True)
        f_3 = F.interpolate(F3, size=f_1.size()[2:], mode='bilinear', align_corners=True)
        f_4 = F.interpolate(F4, size=f_1.size()[2:], mode='bilinear', align_corners=True)


        """FPN:"""
        #try to predict use each feature

        """----"""
        feature_map=torch.cat([f_1,f_2,f_3,f_4],dim=1)
        #feature_map=self.extra[4](feature_map)
        # feature_map=self.extra[5](self.extra[4](feature_map))
        out0= feature_map
        cts=f_4#or=4
        """*************"""


        """instead of the hook in feature_loss"""
        hook = out0
        w = self.head[0].weight
        c = w.shape[1]
        c1 = F.conv2d(hook, w.transpose(0, 1), padding=(1, 1), groups=c)
        """*******"""

        """Contrast losses"""
        # for contrastive:test stage5 and fuse-feature
        #x_c = torch.sigmoid(self.cts_bn(self.head[0](out0)))
        #loss=self.contrast(x_c,cts)
        """"""
        out0 = F.interpolate(self.head[0](out0), size=shape, mode='bilinear', align_corners=False)
        #

        # [16,1,320,320]
        #out0=out0+(out1*out2+out3*out4)
        if self.cfg.mode == 'train':
            return out0, 0,out0,out0,out0,out0,c1#,fg,bg
        else:
            return out0,attn_map

    def initialize(self):
        print('initialize net')
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot), strict=False)
        else:
            weight_init(self)