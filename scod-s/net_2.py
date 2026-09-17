import torch
import torch.nn as nn
import torch.nn.functional as F
from pvtv2 import pvt_v2_b2,pvt_v2_b1,pvt_v2_b3,pvt_v2_b4
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange
from math import sqrt
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
        elif isinstance(m,MSC_Final):

            pass
        # elif isinstance(m,nn.ConvTranspose2d):
        #     nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        else:
            m.initialize()

class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias
        

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)

def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


# class GroupDynamicScale(nn.Module):
#     def __init__(self, dim, group, expansion_ratio=1, reweight_expansion_ratio=.125,
#                  act1_layer=StarReLU, act2_layer=nn.Identity,
#                  bias=False, num_filters=4, size=14, weight_resize=True, init_scale=1e-5,
#                  **kwargs):
#         super().__init__()
        
#         self.size = size
#         self.filter_size = size // 2 + 1
#         self.num_filters = num_filters
#         self.dim = dim
#         self.weight_resize = weight_resize
#         self.reweight = Mlp(dim, reweight_expansion_ratio, group * num_filters, bias=False)
#         self.complex_weights = nn.Parameter(
#             torch.randn(num_filters, dim//group, self.size, self.filter_size,dtype=torch.float32) * init_scale)
#         trunc_normal_(self.complex_weights, std=init_scale)
        
#     def forward(self, x):
#         B, C, H, W, = x.shape
#         x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
#         B, C, RH, RW, = x_rfft.shape
#         x = x.permute(0, 2, 3, 1)

#         routeing = self.reweight(x.mean(dim=(1, 2))).view(B, -1, self.num_filters).tanh_() # b, num_filters, group
#         weight = self.complex_weights
#         if not weight.shape[2:4] == x_rfft.shape[2:4]:
#             weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)
#         weight = torch.einsum('bgf,fchw->bgchw', routeing, weight)
#         weight = weight.reshape(B, C, RH, RW)
#         x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
#         x = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')
#         return x

# class AttentionwithAttInv(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., lf_dy_weight=True, hf_dy_weight=True):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         ### AttInv
#         self.lf_dy_weight = lf_dy_weight
#         self.hf_dy_weight = hf_dy_weight

#         if self.lf_dy_weight:
#             self.dy_freq_2 = nn.Linear(dim, self.num_heads, bias=True)
#             # self.lf_gamma= nn.Parameter(1e-5 * torch.ones((dim, 1, 1)),requires_grad=True) # with decay if dim > 1
#             self.lf_gamma= nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True) # no decay

#         if self.hf_dy_weight:
#             self.dy_freq = nn.Linear(dim, self.num_heads, bias=True)
#             # self.dy_freq_gate = nn.Linear(embed_dim, self.num_heads, bias=False)
#             # constant_(self.dy_freq.bias, -5)
#             # self.hf_gamma= nn.Parameter(1e-5 * torch.ones((dim, 1, 1)),requires_grad=True) # with decay if dim > 1
#             self.hf_gamma= nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True) # no decay
#         self.dy_freq_starrelu = StarReLU()
#         self.ignore_cls_token = 0
        

#     def forward(self, x, H=None, W=None):
#         B, N, C = x.shape

#         # hw_cls, b, c = x.shape
#         dy_freq_feat = self.dy_freq_starrelu(x[:, self.ignore_cls_token:])

#         if hasattr(self, 'dy_freq_2'):
#             dy_freq_lf = self.dy_freq_2(dy_freq_feat).tanh_()
#             dy_freq_lf = dy_freq_lf.reshape(B, N - self.ignore_cls_token,  self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
#             dy_freq_lf = dy_freq_lf.reshape(B, N - self.ignore_cls_token, C) 

#         if hasattr(self, 'dy_freq'):
#             dy_freq = F.softplus(self.dy_freq(dy_freq_feat))
#             dy_freq2 = dy_freq ** 2
#             # dy_freq = dy_freq2 / (dy_freq2 + 1)
#             dy_freq = 2 * dy_freq2 / (dy_freq2 + 0.3678)
#             # dy_freq_clone = dy_freq.transpose(1, 0).clone()
#             dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token,  self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
#             dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token, C) 
#             # dy_freq_spatial = dy_freq_spatial * dy_freq_channel
#             if self.ignore_cls_token > 0:
#                 dy_freq = torch.cat([torch.zeros([B, self.ignore_cls_token, C], device=dy_freq.device), dy_freq], dim=1)


#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, head, N, C//head
#         q, k, v = qkv[0], qkv[1], qkv[2]
        
#         q = q * self.scale

#         attn = (q @ k.transpose(-2, -1))
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)

#         # B, head, N, C//head
#         v = v.permute(0, 2, 1, 3).reshape(B, N, C)
#         v_hf = v - x
#         # x = x + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)
#         # x = x + x * self.lf_gamma.view(1, 1, -1) + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)
#         if hasattr(self, 'dy_freq_2'):
#             x = x + x * dy_freq_lf * self.lf_gamma.view(1, 1, -1)
#         if hasattr(self, 'dy_freq'):
#             x = x + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# class FdamBlock(nn.Module):
#     def __init__(self, dim, num_heads, group,mlp_ratio=4., drop_path=0., norm_layer=nn.LayerNorm, init_values=1e-4):
#         super().__init__()
#         # --- Standard components ---
#         self.norm1 = norm_layer(dim)
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         self.mlp = Mlp(dim=dim, mlp_ratio=mlp_ratio)
#         self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
#         self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        
#         # --- FDAM-specific upgrades ---
#         # 1. Replace Attention with AttentionwithAttInv
#         self.attn = AttentionwithAttInv(dim, num_heads=num_heads)
        
#         # 2. Add GroupDynamicScale after Attention and MLP
#         self.freq_scale_1 = GroupDynamicScale(dim=dim,group=group)
#         self.freq_scale_2 = GroupDynamicScale(dim=dim,group=group)

#     def forward(self, x, y H, W):
#         # 输入为 NCHW，这里统一在 NLC 下进行 LN/Attention/MLP，再在频域模块前后做 NCHW <-> NLC 变换
#         x_nlc = nchw_to_nlc(x)  # (N, L, C)
 
#         # --- Attention block with FDAM ---
#         x_att = self.attn(self.norm1(x_nlc))  # (N, L, C)
 
#         # 频域缩放：NLC -> NCHW -> NLC
#         x_att_reshaped = nlc_to_nchw(x_att, (H, W))
#         x_att_scaled = self.freq_scale_1(x_att_reshaped) + x_att_reshaped
#         x_att = nchw_to_nlc(x_att_scaled)
 
#         x_nlc = x_nlc + self.drop_path(self.gamma_1 * x_att)
 
#         # --- MLP block with FDAM ---
#         x_mlp = self.mlp(self.norm2(x_nlc))  # (N, L, C)
 
#         # 频域缩放：NLC -> NCHW -> NLC
#         x_mlp_reshaped = nlc_to_nchw(x_mlp, (H, W))
#         x_mlp_scaled = self.freq_scale_2(x_mlp_reshaped) + x_mlp_reshaped
#         x_mlp = nchw_to_nlc(x_mlp_scaled)
 
#         x_nlc = x_nlc + self.drop_path(self.gamma_2 * x_mlp)
 
#         # 返回 NCHW 以对接后续卷积
#         x_out = nlc_to_nchw(x_nlc, (H, W))
#         return x_out


# class TokenLPHF(nn.Module):
#     """
#     从 NCHW 特征图上做一个简化的多头注意力，输出：
#       LP = A @ V  （低频 / 被注意力聚合的成分）
#       HF = (I - A) @ V = V - LP （高频 / 细节残差）
#     说明：这是“近似 AttInv”的实现，不引入你上面动态门，纯粹拿 LP/HF 张量做跨层融合。
#     """
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.norm = nn.LayerNorm(dim)
#         self.qkv = nn.Linear(dim, dim * 3, bias=True)

#     def forward(self, x_nchw):
#         B, C, H, W = x_nchw.shape
#         x = nchw_to_nlc(x_nchw)             # [B, HW, C]
#         x = self.norm(x)
#         qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
#         q, k, v = qkv[0], qkv[1], qkv[2]    # [B,h,HW,hd]
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # [B,h,HW,HW]
#         lp = (attn @ v).transpose(1,2).reshape(B, H*W, C) # [B,HW,C]
#         lp = nlc_to_nchw(lp, (H, W))                      # [B,C,H,W]

#         v_ = v.transpose(1,2).reshape(B, H*W, C)          # 还原到 [B,HW,C]
#         v_ = nlc_to_nchw(v_, (H, W))
#         hf = v_ - lp                                      # (I - A) v
#         return lp, hf

# class CrossAttInvFusion(nn.Module):
#     """
#     输入：浅层 S ∈ ℝ^{B×C×Hs×Ws}，深层 D ∈ ℝ^{B×C×Hd×Wd}
#     输出：S'（浅层经深层LF引导后的结果 FL），D'（深层经浅层HF反哺后的结果 FH）
#     """
#     def __init__(self, C, num_heads=8, alpha_lf=1e-3, alpha_hf=5e-4,warmup_ratio=0.1):
#         super().__init__()
#         self.alpha_lf = nn.Parameter(torch.tensor(alpha_lf))  # 可学习的小系数
#         self.alpha_hf = nn.Parameter(torch.tensor(alpha_hf))

#         # 用于提取 LP/HF
#         self.lphf_s = TokenLPHF(dim=C, num_heads=num_heads)
#         self.lphf_d = TokenLPHF(dim=C, num_heads=num_heads)
        
#         self.warmup_ratio = warmup_ratio
#         self.enable_bu = True  # 评估阶段默认开启；训练时由 set_progress() 决定

#         # Top-Down (深→浅) 低频引导门
#         self.proj_td = nn.Conv2d(C, C, 1)
#         self.gate_td = nn.Sequential(nn.Conv2d(C, C, 1), nn.Sigmoid())

#         # Bottom-Up (浅→深) 高频反哺：深度可分离下采样 + 通道对齐 + 门控
#         self.down_bu = nn.Sequential(
#             nn.Conv2d(C, C, 3, stride=1, padding=1, groups=C),
#             nn.Conv2d(C, C, 1)
#         )
#         self.gate_bu = nn.Sequential(nn.Conv2d(C, C, 1), nn.Sigmoid())


#     @torch.no_grad()
#     def set_progress(self, curr_epoch:int=None, max_epoch:int=None, is_training:bool=True):
#         """
#         训练时按 epoch 比例控制是否开启 Bottom-Up；评估时恒开启
#         """
#         if not is_training or curr_epoch is None or max_epoch is None or max_epoch <= 0:
#             self.enable_bu = True
#             return
#         ratio = float(curr_epoch) / float(max_epoch)
#         self.enable_bu = (ratio >= self.warmup_ratio)

#     def forward(self, S, D):
#         """
#         S: 浅层 (B,C,Hs,Ws)
#         D: 深层 (B,C,Hd,Wd)
#         """
#         # 1) 取出各自的 LP/HF
#         LP_s, HF_s = self.lphf_s(S)                   # [B,C,Hs,Ws]
#         LP_d, _    = self.lphf_d(D)                   # [B,C,Hd,Wd]  （深层HF不用于本轮）

#         # 2) Top-Down：用深层LP引导浅层LP
#         g_td = F.interpolate(self.proj_td(LP_d), size=S.shape[-2:], mode='bilinear', align_corners=False)  # 对齐到浅层尺寸
#         w_td = self.gate_td(g_td)                     # [B,C,Hs,Ws]，∈(0,1)
#         S_out = S + self.alpha_lf * ( w_td * LP_s )   # 让浅层LP在深层结构模板下被“扶正”

#         # Bottom-Up（按进度开关）
#         if self.enable_bu:
#             with torch.no_grad():
#                 HF_s_detach = HF_s.detach()
#             h_bu = self.down_bu(HF_s_detach)
#             w_bu = self.gate_bu(h_bu)
#             D_out = D + self.alpha_hf * (w_bu * h_bu)
#         else:
#             D_out = D  # 暖启动阶段不反哺

#         return S_out, D_out
# class AttentionwithAttInv(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., lf_dy_weight=True, hf_dy_weight=True):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         ### AttInv
#         self.lf_dy_weight = lf_dy_weight
#         self.hf_dy_weight = hf_dy_weight

#         if self.lf_dy_weight:
#             self.dy_freq_2 = nn.Linear(dim, self.num_heads, bias=True)
#             self.lf_gamma = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)

#         if self.hf_dy_weight:
#             self.dy_freq = nn.Linear(dim, self.num_heads, bias=True)
#             self.hf_gamma = nn.Parameter(1e-5 * torch.ones((dim)), requires_grad=True)
#         self.dy_freq_starrelu = StarReLU()
#         self.ignore_cls_token = 0

#     def forward(self, x, H=None, W=None, return_lphf=False):
#         #import pdb; pdb.set_trace()
#         x_nchw = x  # 保存原始NCHW格式
#         x = nchw_to_nlc(x)
#         B, N, C = x.shape
#         dy_freq_feat = self.dy_freq_starrelu(x[:, self.ignore_cls_token:])
        
#         # Low-frequency components (lf)
#         if hasattr(self, 'dy_freq_2'):
#             dy_freq_lf = self.dy_freq_2(dy_freq_feat).tanh_()
#             dy_freq_lf = dy_freq_lf.reshape(B, N - self.ignore_cls_token, self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
#             dy_freq_lf = dy_freq_lf.reshape(B, N - self.ignore_cls_token, C)

#         # High-frequency components (hf)
#         if hasattr(self, 'dy_freq'):
#             dy_freq = F.softplus(self.dy_freq(dy_freq_feat))
#             dy_freq2 = dy_freq ** 2
#             dy_freq = 2 * dy_freq2 / (dy_freq2 + 0.3678)
#             dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token, self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads)
#             dy_freq = dy_freq.reshape(B, N - self.ignore_cls_token, C)
#             if self.ignore_cls_token > 0:
#                 dy_freq = torch.cat([torch.zeros([B, self.ignore_cls_token, C], device=dy_freq.device), dy_freq], dim=1)

#         # Attention calculations (qkv)
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         q = q * self.scale

#         # Attention matrix
#         attn = (q @ k.transpose(-2, -1))
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x_lp = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 低频分量LP

#         # Compute high-frequency residual
#         v = v.permute(0, 2, 1, 3).reshape(B, N, C)
#         v_hf = v - x_lp  # 高频分量HF
        
#         # 如果只需要返回LP和HF，直接返回原始的LP和HF（未增强）
#         if return_lphf:
#             # 将LP和HF转换回NCHW格式
#             _, H, W = x_nchw.shape[0], x_nchw.shape[2], x_nchw.shape[3]
#             LP = nlc_to_nchw(x_lp, (H, W))
#             HF = nlc_to_nchw(v_hf, (H, W))
#             return LP, HF
        
#         # 否则增强低频和高频分量，然后返回增强后的特征
#         if hasattr(self, 'dy_freq_2'):
#             x_lp = x_lp + x_lp * dy_freq_lf * self.lf_gamma.view(1, 1, -1)
#         if hasattr(self, 'dy_freq'):
#             v_hf = v_hf + dy_freq * v_hf * self.hf_gamma.view(1, 1, -1)
        
#         x = x_lp + v_hf
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

# class GroupDynamicScale(nn.Module):
#     def __init__(self, dim, group, expansion_ratio=1, reweight_expansion_ratio=.125, act1_layer=StarReLU, act2_layer=nn.Identity, bias=False, num_filters=4, size=14, weight_resize=True, init_scale=1e-5, **kwargs):
#         super().__init__()
        
#         self.size = size
#         self.filter_size = size // 2 + 1
#         self.num_filters = num_filters
#         self.dim = dim
#         self.weight_resize = weight_resize
#         self.reweight = Mlp(dim, reweight_expansion_ratio, group * num_filters, bias=False)
#         self.complex_weights = nn.Parameter(
#             torch.randn(num_filters, dim//group, self.size, self.filter_size, dtype=torch.float32) * init_scale)
#         trunc_normal_(self.complex_weights, std=init_scale)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
#         B, C, RH, RW = x_rfft.shape
#         x = x.permute(0, 2, 3, 1)

#         routeing = self.reweight(x.mean(dim=(1, 2))).view(B, -1, self.num_filters).tanh_() # b, num_filters, group
#         weight = self.complex_weights
#         if not weight.shape[2:4] == x_rfft.shape[2:4]:
#             weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)
#         weight = torch.einsum('bgf,fchw->bgchw', routeing, weight)
#         weight = weight.reshape(B, C, RH, RW)
#         x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
#         x = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')
#         return x

# class CrossAttInvFusion(nn.Module):
#     def __init__(self, C, num_heads=8, alpha_lf=1e-3, alpha_hf=5e-4, warmup_ratio=0.1):
#         super().__init__()
#         self.alpha_lf = nn.Parameter(torch.tensor(alpha_lf))
#         self.alpha_hf = nn.Parameter(torch.tensor(alpha_hf))

#         # 用于提取LP和HF的Attention模块
#         self.lphf_s = AttentionwithAttInv(dim=C, num_heads=num_heads)
#         self.lphf_d = AttentionwithAttInv(dim=C, num_heads=num_heads)
        
#         # 用于浅层和深层特征的GroupDynamicScale
#         self.group_scale_s = GroupDynamicScale(dim=C, group=8, expansion_ratio=1, reweight_expansion_ratio=0.125)
#         self.group_scale_d = GroupDynamicScale(dim=C, group=8, expansion_ratio=1, reweight_expansion_ratio=0.125)

#         self.warmup_ratio = warmup_ratio
#         self.enable_bu = True

#         # 深层HF增强浅层HF的模块
#         self.proj_hf_d2s = nn.Conv2d(C, C, 1)  # 深层HF投影到浅层
#         self.gate_hf_d2s = nn.Sequential(nn.Conv2d(C, C, 1), nn.Sigmoid())  # 门控机制
        
#         # 浅层LP增强深层LP的模块
#         self.proj_lf_s2d = nn.Conv2d(C, C, 1)  # 浅层LP投影到深层
#         self.gate_lf_s2d = nn.Sequential(nn.Conv2d(C, C, 1), nn.Sigmoid())  # 门控机制
        
#         # 最终融合模块
#         self.fusion_conv = nn.Conv2d(C * 2, C, 1)  # 融合两个增强后的特征

#     @torch.no_grad()
#     def set_progress(self, curr_epoch:int=None, max_epoch:int=None, is_training:bool=True):
#         if not is_training or curr_epoch is None or max_epoch is None or max_epoch <= 0:
#             self.enable_bu = True
#             return
#         ratio = float(curr_epoch) / float(max_epoch)
#         self.enable_bu = (ratio >= self.warmup_ratio)

#     def forward(self, S, D):
#         """
#         S: 浅层特征 (B, C, Hs, Ws)
#         D: 深层特征 (B, C, Hd, Wd)
#         """
#         # 1. 提取浅层和深层的LP和HF分量
#         LP_s, HF_s = self.lphf_s(S, return_lphf=True)
#         LP_d, HF_d = self.lphf_d(D, return_lphf=True)
        
#         # 2. 深层HF增强浅层HF，然后增强浅层特征
#         # 将深层HF对齐到浅层尺寸
#         HF_d_aligned = F.interpolate(HF_d, size=S.shape[-2:], mode='bilinear', align_corners=False)
#         # 投影和门控
#         g_hf_d2s = self.proj_hf_d2s(HF_d_aligned)
#         w_hf_d2s = self.gate_hf_d2s(g_hf_d2s)
#         # 用深层HF增强浅层HF
#         HF_s_enhanced = HF_s + self.alpha_hf * (w_hf_d2s * HF_s)
#         # 增强浅层特征
#         S_enhanced = S + HF_s_enhanced
#         # 通过GroupDynamicScale得到浅层增强特征
#         S_out = self.group_scale_s(S_enhanced)
        
#         # 3. 浅层LP增强深层LP，然后增强深层特征
#         # 将浅层LP对齐到深层尺寸
#         LP_s_aligned = F.interpolate(LP_s, size=D.shape[-2:], mode='bilinear', align_corners=False)
#         # 投影和门控
#         g_lf_s2d = self.proj_lf_s2d(LP_s_aligned)
#         w_lf_s2d = self.gate_lf_s2d(g_lf_s2d)
#         # 用浅层LP增强深层LP
#         LP_d_enhanced = LP_d + self.alpha_lf * (w_lf_s2d * LP_d)
#         # 增强深层特征
#         D_enhanced = D + LP_d_enhanced
#         # 通过GroupDynamicScale得到深层增强特征
#         D_out = self.group_scale_d(D_enhanced)
        
#         # 4. 最终融合两个增强特征
#         # 将两个特征对齐到相同尺寸（以浅层尺寸为准）
#         D_out_aligned = F.interpolate(D_out, size=S_out.shape[-2:], mode='bilinear', align_corners=False)
#         # 拼接并融合
#         fused = torch.cat([S_out, D_out_aligned], dim=1)
#         fused_out = self.fusion_conv(fused)
        
#         return fused_out


class MSC_Final(nn.Module):
    """
    最终版多尺度注意力融合模块（省显存 + 动态门控双分支）

    特点：
    1）多尺度通道拼接：y 的 3 个尺度在通道维拼接成 [B, H*W, 3C]，通过线性层压回 dim，
       注意力仍在 N = H*W 上算，不会把 token 数翻 3 倍，显存友好；
    2）双 Top-k 分支：coarse（大 k，偏全局）和 fine（小 k，偏局部高置信）；
    3）内容自适应门控：用一个小 MLP 根据每个 query token 的特征，动态融合 coarse / fine 两个分支，
       而不是简单两个全局标量权重。
    """

    def __init__(self, dim, num_heads=8, topk=True,
                 kernel=(3, 5, 7), s=(1, 1, 1), pad=(1, 2, 3),
                 qkv_bias=False, qk_scale=None,
                 attn_drop_ratio=0., proj_drop_ratio=0.,
                 k1=2, k2=3):
        """
        Args:
            dim: 通道数 C，需能被 num_heads 整除
            num_heads: 多头数
            topk: 是否启用 Top-k 稀疏注意力
            kernel/s/pad: 多尺度池化参数（3 个尺度）
            qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio: 同原版
            k1, k2: Top-k 分支的比例分母，N/k1 > N/k2（coarse > fine）
        """
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        # Q 从 x 生成
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # K/V 从多尺度融合后的 y 生成
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.k1 = k1
        self.k2 = k2
        self.topk = topk

        # -------- 多尺度通道拼接融合：3C -> C --------
        self.num_scales = len(kernel)
        assert self.num_scales == 3, "当前实现默认 3 个尺度，可按需改成 ModuleList 版本"
        self.msf = nn.Linear(dim * self.num_scales, dim)  # Multi-Scale Fusion

        # 多尺度池化（保持空间尺寸不变）
        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)

        # -------- 动态门控 MLP：从 query 特征自适应决定 coarse/fine 权重 --------
        hidden_dim = max(dim // 4, 16)
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),   # 输出 coarse / fine 两个 gate
            nn.Sigmoid()               # 映射到 (0,1)，后续再归一化
        )

    def forward(self, x, y):
        """
        Args:
            x: 高层特征 [B, C, H, W]
            y: 低层或多层融合特征 [B, C, H, W]
        Returns:
            out: 融合后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W

        # 1. 多尺度池化（保持 H, W 不变）
        y1 = self.avgpool1(y)   # [B, C, H, W]
        y2 = self.avgpool2(y)   # [B, C, H, W]
        y3 = self.avgpool3(y)   # [B, C, H, W]

        # 2. 展平为序列，并在“通道维”拼接：3C -> C
        #    [B, C, H, W] -> [B, H*W, C]
        y1_seq = rearrange(y1, 'b c h w -> b (h w) c')
        y2_seq = rearrange(y2, 'b c h w -> b (h w) c')
        y3_seq = rearrange(y3, 'b c h w -> b (h w) c')

        # [B, H*W, 3C]
        y_cat = torch.cat([y1_seq, y2_seq, y3_seq], dim=-1)
        # 线性融合多尺度信息：3C -> C，再做 LayerNorm
        y_feat = self.msf(y_cat)        # [B, H*W, C]
        y_feat = self.layer_norm(y_feat)

        # 3. x 展平为序列，用于生成 Q： [B, C, H, W] -> [B, H*W, C]
        x_seq = rearrange(x, 'b c h w -> b (h w) c')

        # 4. 生成 K / V（来自多尺度融合后的 y_feat）
        B, N1, C_ = y_feat.shape     # N1 = H*W
        assert N1 == N, "x 和 y 的空间尺寸必须一致"

        kv = self.kv(y_feat).reshape(B, N1, 2, self.num_heads, C_ // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)  # [2, B, num_heads, N1, head_dim]
        k, v = kv[0], kv[1]             # [B, num_heads, N1, head_dim]

        # 5. 生成 Q（来自 x_seq）
        q = self.q(x_seq).reshape(B, N, self.num_heads, C_ // self.num_heads)
        q = q.permute(0, 2, 1, 3)       # [B, num_heads, N, head_dim]

        # 6. 原始注意力得分： [B, num_heads, N, N1]，这里 N1 = N
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 7. 双 Top-k 分支：coarse（大 k）和 fine（小 k）
        if self.topk:
            # coarse 分支：保留更多 key（更偏全局）
            topk1 = max(1, N1 // self.k1)
            mask1 = torch.zeros_like(attn, dtype=torch.bool)
            idx1 = torch.topk(attn, k=topk1, dim=-1, largest=True)[1]  # [B, heads, N, topk1]
            mask1.scatter_(-1, idx1, True)

            attn1 = torch.where(mask1, attn, torch.full_like(attn, float('-inf')))
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            out_coarse = attn1 @ v  # [B, num_heads, N, head_dim]

            # fine 分支：保留更少 key（更偏局部高置信）
            topk2 = max(1, N1 // self.k2)
            mask2 = torch.zeros_like(attn, dtype=torch.bool)
            idx2 = torch.topk(attn, k=topk2, dim=-1, largest=True)[1]
            mask2.scatter_(-1, idx2, True)

            attn2 = torch.where(mask2, attn, torch.full_like(attn, float('-inf')))
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            out_fine = attn2 @ v  # [B, num_heads, N, head_dim]
        else:
            # 不稀疏时，退化成普通多头注意力，两分支相同
            attn_soft = attn.softmax(dim=-1)
            attn_soft = self.attn_drop(attn_soft)
            out_coarse = attn_soft @ v
            out_fine = out_coarse

        # 8. 动态门控：根据每个 query token 的特征自适应融合两个分支
        # gate: [B, N, 2] -> 对 coarse / fine 两个分支的权重
        gate = self.gate_mlp(x_seq)  # Sigmoid 输出在 (0,1)
        # 归一化：让 g_coarse + g_fine ≈ 1
        gate = gate / (gate.sum(dim=-1, keepdim=True) + 1e-6)

        g_coarse = gate[..., 0].unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        g_fine   = gate[..., 1].unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]

        # 按位置、按 head 动态融合
        out = out_coarse * g_coarse + out_fine * g_fine     # [B, num_heads, N, head_dim]

        # 9. 还原为 [B, C, H, W]
        out = out.transpose(1, 2).reshape(B, N, C_)  # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)

        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)

        return out



BatchNorm2d=nn.BatchNorm2d
BN_MOMENTUM=0.1
def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)

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
            conv3x3(64 , 1)
        ])


        self.fusion = nn.ModuleList([
            conv3x3(64*2 , 64),
            conv3x3(64*2 , 64)
        ])


        self.MSCv2= MSC_Final(dim=64, num_heads=8)


        self.initialize()


    def forward(self, x, epoch=None, shape=None):

        shape = x.size()[2:] if shape is None else shape

        #     # 告诉融合模块当前训练进度（训练时按比例开关，评估时恒开）
        # self.crossatteionInv.set_progress(
        #     curr_epoch=epoch,
        #     max_epoch=60,
        #     is_training=self.training
        # )
        attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2=self.bkbone(x)#bk_stage5=[8,512,6,6]
        #cls_token4,cls_token3,cls_token2,cls_token1,\
        #    attn_map,bk_stage5,bk_stage4,bk_stage3,bk_stage2=self.bkbone(x)#bk_stage5=[8,512,6,6]



        F1 = self.extra[0](bk_stage2)  # 64
        F2 = self.extra[1](bk_stage3)
        F3 = self.extra[2](bk_stage4)
        F4 = self.extra[3](bk_stage5)  #512



        f_2 = F2
        f_1 = F.interpolate(F1, size=f_2.size()[2:], mode='bilinear', align_corners=True)

        f_21 = self.fusion[0](torch.cat([f_1,f_2],dim=1)) #4x64x64x64

        f_3 = F3 #= F.interpolate(F3, size=f_1.size()[2:], mode='bilinear', align_corners=True)
        f_4 = F.interpolate(F4, size=f_3.size()[2:], mode='bilinear', align_corners=True) 

        f_34 = self.fusion[1](torch.cat([f_3,f_4],dim=1)) #4x64x32x32
        f_34s = F.interpolate(f_34, size=f_2.size()[2:], mode='bilinear', align_corners=True) 

        # CrossAttInvFusion返回融合后的特征和深层增强特征
        fused_out  = self.MSCv2(f_34s,f_21) # fused_out: 4x64x64x64, D_out: 4x64xHxW
        
        # 将深层特征对齐到融合特征尺寸并拼接
        #F1s = F.interpolate(F1, size=fused_out.shape[-2:], mode='bilinear', align_corners=False)
        #F4s = F.interpolate(F4, size=fused_out.shape[-2:], mode='bilinear', align_corners=False)
        #feature_map = torch.cat([fused_out, F1s,F4s], dim=1)  # 4x128x64x64
        out0 = F.interpolate(self.head[0](fused_out), size=shape, mode='bilinear', align_corners=False)

        # """FPN:"""
        # #try to predict use each feature

        # """----"""
        # feature_map=torch.cat([f_1,f_2,f_3,f_4],dim=1)
        # #feature_map=self.extra[4](feature_map)
        # # feature_map=self.extra[5](self.extra[4](feature_map))
        # out0= feature_map
        # cts=f_4#or=4
        # """*************"""


        # """instead of the hook in feature_loss"""
        # hook = out0
        # w = self.head[0].weight
        # c = w.shape[1]
        # c1 = F.conv2d(hook, w.transpose(0, 1), padding=(1, 1), groups=c)
        # """*******"""

        # """Contrast losses"""
        # # for contrastive:test stage5 and fuse-feature
        # #x_c = torch.sigmoid(self.cts_bn(self.head[0](out0)))
        # #loss=self.contrast(x_c,cts)
        # """"""
        # out0 = F.interpolate(self.head[0](out0), size=shape, mode='bilinear', align_corners=False)
        # #

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