import torch.nn.functional as F
import torch
from feature_loss import *
from tools import *
from utils import ramps
import numpy as np

device = torch.device("cuda:0")
criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').to(device)
loss_lsc = FeatureLoss().to(device)

# [修改重点 1] 将 'rgb' 替换为 'deep_f'，我们将利用抗伪装的深层特征代替原图颜色计算亲和力
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "deep_f": 0.5}]
loss_lsc_radius = 5
l = 0.3




def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        pp = Translate(0.15)
    elif op==2:
        pp = Crop(0.7, 0.7)
    return pp

def get_color_tranform(ops=[0,1,2,3,4,5]):
    op=np.random.choice(ops)
    if op==3:
        pp=GaussianBlur(5)
        return pp
    if op==4:
        pp=mask()
        return pp
    if op==5:
        pp=Color_jitter()
        return pp
    return None

def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0,1), padding=(1,1), groups=c)
    return c1, b

def unsymmetric_grad(x, y, calc, w1, w2):
    return calc(x, y.detach())*w1 + calc(x.detach(), y)*w2

# def get_diffusion_schedule(epoch, total_epoch):
#     """
#     Three-stage schedule:
#       stage A: conservative propagation
#       stage B: expansion
#       stage C: contraction / boundary stabilization
#     """
#     r = float(epoch - 1) / float(max(total_epoch - 1, 1))

#     if r < 0.25:
#         # early stage: learn scribble seeds first
#         l_lsc = 0.08
#         tau_fg = 0.80
#         tau_bg = 0.20
#     elif r < 0.70:
#         # middle stage: gradually expand
#         t = (r - 0.25) / (0.70 - 0.25)
#         l_lsc = 0.08 + t * (0.35 - 0.08)   # 0.08 -> 0.35
#         tau_fg = 0.80 - t * 0.10           # 0.80 -> 0.70
#         tau_bg = 0.20 + t * 0.10           # 0.20 -> 0.30
#     else:
#         # late stage: stabilize boundary, reduce aggressive propagation
#         t = (r - 0.70) / (1.00 - 0.70)
#         l_lsc = 0.35 - t * 0.10            # 0.35 -> 0.25
#         tau_fg = 0.70 + t * 0.05           # 0.70 -> 0.75
#         tau_bg = 0.30 - t * 0.05           # 0.30 -> 0.25

#     return l_lsc, tau_fg, tau_bg
def get_diffusion_schedule(epoch, total_epoch):
    """
    Three-stage schedule:
      stage A: conservative propagation
      stage B: expansion
      stage C: contraction / boundary stabilization
    """
    r = float(epoch - 1) / float(max(total_epoch - 1, 1))

    if r < 0.25:
        l_lsc = 0.08
        l_r2u = 0.05 
        #l_r2u = 0
        tau_fg = 0.80
        tau_bg = 0.20
    elif r < 0.70:
        t = (r - 0.25) / (0.70 - 0.25)
        l_lsc = 0.08 + t * (0.35 - 0.08)
        l_r2u = 0.05 + t * (0.15 - 0.05) 
        #l_r2u = 0
        tau_fg = 0.80 - t * 0.10
        tau_bg = 0.20 + t * 0.10
    else:
        t = (r - 0.70) / (1.00 - 0.70)
        l_lsc = 0.35 - t * 0.10
        l_r2u = 0.15 - t * 0.05  
        tau_fg = 0.70 + t * 0.05
        tau_bg = 0.30 - t * 0.05

    return l_lsc, l_r2u, tau_fg, tau_bg

# def build_confidence_interval_consensus(prob_list, tau_fg=0.7, tau_bg=0.3):
#     """
#     prob_list: list of [B,1,H,W], e.g. [sigmoid(out1), ..., sigmoid(out4)]

#     Returns:
#       M_cgamf: fused soft consensus in [0,1]
#       mask_consensus: tri-state hard consensus in {0,1,255}
#     """
#     weights = []
#     for p in prob_list:
#         entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
#         weights.append(torch.exp(-entropy))

#     sum_w = sum(weights) + 1e-8
#     weights_norm = [w / sum_w for w in weights]
#     M_cgamf = sum(w * p for w, p in zip(weights_norm, prob_list))

#     consensus_fg = (M_cgamf > tau_fg)
#     consensus_bg = (M_cgamf < tau_bg)

#     # tri-state: 255 means uncertain
#     mask_consensus = torch.full_like(M_cgamf, 255.0)
#     mask_consensus[consensus_bg] = 0.0
#     mask_consensus[consensus_fg] = 1.0

#     return M_cgamf, mask_consensus
 
def build_confidence_interval_consensus(prob_list, scribble_mask=None, tau_fg=0.8, tau_bg=0.4):
    """
    Seed-Preserving Consensus Diffusion

    prob_list: list of [B,1,H,W]
    scribble_mask: [B,1,H,W], 取值 {0,1,255}
        1   -> foreground scribble
        0   -> background scribble
        255 -> unlabeled

    Returns:
      M_cgamf: fused soft consensus in [0,1]
      mask_consensus: tri-state hard consensus in {0,1,255}
    """
    weights = []
    for p in prob_list:
        entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        weights.append(torch.exp(-entropy))

    sum_w = sum(weights) + 1e-8
    weights_norm = [w / sum_w for w in weights]
    M_cgamf = sum(w * p for w, p in zip(weights_norm, prob_list))

    consensus_fg = (M_cgamf > tau_fg)
    consensus_bg = (M_cgamf < tau_bg)

    mask_consensus = torch.full_like(M_cgamf, 255.0)
    mask_consensus[consensus_bg] = 0.0
    mask_consensus[consensus_fg] = 1.0

    # -------------------------------------------------
    # Seed-Preserving: scribble 作为硬锚点写入共识图
    # -------------------------------------------------
    if scribble_mask is not None:
        fg_seed = (scribble_mask == 1)
        bg_seed = (scribble_mask == 0)

        mask_consensus[fg_seed] = 1.0
        mask_consensus[bg_seed] = 0.0

        M_cgamf = M_cgamf.clone()
        M_cgamf[fg_seed] = 1.0
        M_cgamf[bg_seed] = 0.0

    return M_cgamf, mask_consensus

def reliable_to_uncertain_diffusion_loss(
    y_hat_softmax,
    kernels_desc,
    kernels_radius,
    sample,
    height_input,
    width_input,
    mask_consensus
):
    """
    Asymmetric Reliable-to-Uncertain Diffusion

    y_hat_softmax: [N,1,H,W]，前景概率
    mask_consensus: [N,1,H,W]，取值 {0,1,255}
        1   -> reliable foreground
        0   -> reliable background
        255 -> uncertain

    只让 reliable neighbors -> uncertain center
    不改动原始非-ALSC监督，只作为额外 diffusion 正则项。
    """
    assert y_hat_softmax.dim() == 4
    N, C, H, W = y_hat_softmax.shape
    dev = y_hat_softmax.device

    # 和 ALSC 共用同一套 affinity kernel
    kernels = FeatureLoss._create_kernels(
        kernels_desc,
        kernels_radius,
        sample,
        N,
        H,
        W,
        dev,
        custom_modality_downsamplers=None
    )

    consensus_unfolded = FeatureLoss._unfold(mask_consensus.float(), kernels_radius)   # [N,1,K,K,H,W]
    center_consensus = consensus_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(
        N, 1, 1, 1, H, W
    )

    # center 必须是不确定区域
    uncertain_center = (center_consensus == 255).float()

    # neighbor 必须是可靠区域
    reliable_neigh = ((consensus_unfolded == 0) | (consensus_unfolded == 1)).float()
    fg_neigh = (consensus_unfolded == 1).float()

    # 只允许 reliable neighbor -> uncertain center
    weights = kernels * uncertain_center * reliable_neigh

    denom = weights.sum(dim=(2, 3))                    # [N,1,H,W]
    pseudo_fg = (weights * fg_neigh).sum(dim=(2, 3)) / (denom + 1e-8)   # [N,1,H,W]

    valid = (denom > 1e-6).float()

    # 用 soft target 约束 uncertain center 的前景概率
    loss_map = torch.abs(y_hat_softmax - pseudo_fg)
    loss = (loss_map * valid).sum() / (valid.sum() + 1e-6)

    return loss



def train_loss(
    image,
    mask,
    net,
    ctx,
    ft_dct,
    w_ft=.1,
    ft_st=2,
    ft_fct=.5,
    ft_head=True,
    mtrsf_prob=1,
    ops=[0, 1, 2],
    w_l2g=0,
    l_me=0.1,
    me_st=50,
    me_all=False,
    multi_sc=0,
    l=0.3,
    sl=1):
    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']

    # =========================================================
    # Stage-wise diffusion schedule
    # =========================================================
    #l_lsc, tau_fg, tau_bg = get_diffusion_schedule(epoch, t_epo)
    l_lsc, l_r2u, tau_fg, tau_bg = get_diffusion_schedule(epoch, t_epo)

    ###### saliency structure consistency loss ######
    do_moretrsf = np.random.uniform() < mtrsf_prob

    if do_moretrsf:
        pre_color_transform = get_color_tranform()
        if pre_color_transform is None:
            pre_transform = get_transform(ops)
            image_tr = pre_transform(image)
        else:
            image_tr = pre_color_transform(image, mask)
        large_scale = True
    else:
        large_scale = np.random.uniform() < multi_sc
        image_tr = image
        pre_color_transform = None

    sc_fct = 0.5 if large_scale else 0.3
    image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)

    # network forward
    out0, out1, out2, out3, out4, deep_feat, hook0 = net(image)
    out0_s, out1_s, out2_s, out3_s, out4_s, deep_feat_s, _ = net(image_scale)

    ### Calc intra consistency loss (entropy)
    loss_intra = []
    if epoch >= me_st:
        def entrp(t):
            etp = -(F.softmax(t, dim=1) * F.log_softmax(t, dim=1)).sum(dim=1)
            msk = (etp < 0.5)
            return (etp * msk).sum() / (msk.sum() or 1)

        me = lambda x: entrp(torch.cat((x * 0, x), 1))

        if not me_all:
            e = me(out0)
            loss_intra.append(
                e * get_current_consistency_weight(
                    epoch - me_st,
                    consistency=l_me,
                    consistency_rampup=t_epo - me_st
                )
            )
            loss_intra = loss_intra + [0, 0, 0, 0]
            sw.add_scalar('intra entropy', e.item(), global_step)
        else:
            ga = get_current_consistency_weight(
                epoch - me_st,
                consistency=l_me,
                consistency_rampup=t_epo - me_st
            )
            for i in [out0, out1, out2, out3, out4]:
                loss_intra.append(me(i) * ga)
            sw.add_scalar('intra entropy', loss_intra[0].item(), global_step)
    else:
        loss_intra.extend([0 for _ in range(5)])

    def out_proc(*args):
        a = list(args)
        a = [i.sigmoid() for i in a]
        a = [torch.cat((1 - i, i), 1) for i in a]
        return a

    out0_p, out1_p, out2_p, out3_p, out4_p = out_proc(out0, out1, out2, out3, out4)
    out0_sp, out1_sp, out2_sp, out3_sp, out4_sp = out_proc(out0_s, out1_s, out2_s, out3_s, out4_s)

    # structure consistency
    if not do_moretrsf:
        out0_scale = F.interpolate(out0_p[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=True)
        out0_s_ = out0_sp[:, 1:2]
    else:
        if pre_color_transform is None:
            out0_ss = pre_transform(out0_p)
        else:
            out0_ss = out0_p
        out0_scale = F.interpolate(out0_ss[:, 1:2], scale_factor=1.0, mode='bilinear', align_corners=True)
        out0_s_ = F.interpolate(out0_sp[:, 1:2], scale_factor=1.0 / sc_fct, mode='bilinear', align_corners=True)

    try:
        loss_ssc = Contrastive_loss(out0_s_, out0_scale.detach())
    except NameError:
        loss_ssc = 0.0

    gt = mask.squeeze(1).long()

    bg_label = gt.clone()
    fg_label = gt.clone()
    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255

    # =========================================================
    # Confidence-interval consensus (replace hard 0.5 threshold)
    # =========================================================
    # probs = [torch.sigmoid(o) for o in [out1, out2, out3, out4]]
    # M_cgamf, mask_consensus = build_confidence_interval_consensus(
    #     probs,
    #     tau_fg=tau_fg,
    #     tau_bg=tau_bg
    # )
    probs = [torch.sigmoid(o).detach() for o in [out1, out2, out3, out4]]

    # mask 当前取值: 0(background scribble), 1(foreground scribble), 255(unlabeled)
    M_cgamf, mask_consensus = build_confidence_interval_consensus(
        probs,
        scribble_mask=mask,
        tau_fg=tau_fg,
        tau_bg=tau_bg
    )

    # =========================================================
    # ALSC with tri-state confidence truncation
    # =========================================================
    # out0_down = F.interpolate(out0_p[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    # target_size = out0_down.shape[2:]

    # deep_feat_ = F.interpolate(deep_feat, size=target_size, mode='bilinear', align_corners=True)
    # mask_consensus_ = F.interpolate(mask_consensus, size=target_size, mode='nearest')

    # sample = {'deep_f': deep_feat_}

    # out0_down = F.interpolate(out0_p[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)

    # loss2_lsc = loss_lsc(
    #     out0_down,
    #     loss_lsc_kernels_desc_defaults,
    #     loss_lsc_radius,
    #     sample,
    #     deep_feat_.shape[2],
    #     deep_feat_.shape[3],
    #     mask_consensus=mask_consensus_
    # )['loss']
    target_size = deep_feat.shape[2:]

    deep_feat_ = deep_feat
    sample = {'deep_f': deep_feat_}

    out0_down = F.interpolate(
        out0_p[:, 1:2],
        size=target_size,
        mode='bilinear',
        align_corners=True
    )

    mask_consensus_ = F.interpolate(
        mask_consensus.float(),
        size=target_size,
        mode='nearest'
    )

    # 原 ALSC：只在 reliable-reliable same-region 内做局部一致性
    loss2_lsc = loss_lsc(
        out0_down,
        loss_lsc_kernels_desc_defaults,
        loss_lsc_radius,
        sample,
        deep_feat_.shape[2],
        deep_feat_.shape[3],
        mask_consensus=mask_consensus_
    )['loss']

    # 新增 R2U：只让 reliable neighbors -> uncertain center
    loss2_r2u = reliable_to_uncertain_diffusion_loss(
        out0_down,
        loss_lsc_kernels_desc_defaults,
        loss_lsc_radius,
        sample,
        deep_feat_.shape[2],
        deep_feat_.shape[3],
        mask_consensus_
    )


    # =========================================================
    # Total loss
    # =========================================================
    loss_main = (
        loss_ssc
        + (criterion(out0_p, fg_label) + criterion(out0_p, bg_label))
        + l_lsc * loss2_lsc
        + l_r2u * loss2_r2u
        + loss_intra[0]
    )

    loss_aux1 = criterion(out1_p, fg_label) + criterion(out1_p, bg_label)
    loss_aux2 = criterion(out2_p, fg_label) + criterion(out2_p, bg_label)
    loss_aux3 = criterion(out3_p, fg_label) + criterion(out3_p, bg_label)
    loss_aux4 = criterion(out4_p, fg_label) + criterion(out4_p, bg_label)

    # =========================================================
    # Logging
    # =========================================================
    if ctx:
        sw.add_scalar('sched/l_lsc', l_lsc, global_step)
        sw.add_scalar('sched/l_r2u', l_r2u, global_step)
        sw.add_scalar('sched/tau_fg', tau_fg, global_step)
        sw.add_scalar('sched/tau_bg', tau_bg, global_step)

        with torch.no_grad():
            fg_ratio = (mask_consensus == 1).float().mean()
            bg_ratio = (mask_consensus == 0).float().mean()
            uncertain_ratio = (mask_consensus == 255).float().mean()

        sw.add_scalar('consensus/fg_ratio', fg_ratio.item(), global_step)
        sw.add_scalar('consensus/bg_ratio', bg_ratio.item(), global_step)
        sw.add_scalar('consensus/uncertain_ratio', uncertain_ratio.item(), global_step)
        sw.add_scalar('loss/lsc', loss2_lsc.item(), global_step)
        sw.add_scalar('loss/r2u', loss2_r2u.item(), global_step)

    return loss_main, loss_aux1, loss_aux2, loss_aux3, loss_aux4






















# def train_loss(image, mask, net, ctx, ft_dct, w_ft=.1, ft_st = 2, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=50, me_all=False, multi_sc=0, l=0.3, sl=1):

#     if ctx:
#         epoch = ctx['epoch']
#         global_step = ctx['global_step']
#         sw = ctx['sw']
#         t_epo = ctx['t_epo']

#     ######  saliency structure consistency loss  ######
#     do_moretrsf = np.random.uniform() < mtrsf_prob
#     if do_moretrsf:
#         pre_color_transform = get_color_tranform()
#         if pre_color_transform == None:
#             pre_transform = get_transform(ops)
#             image_tr = pre_transform(image)
#         else:
#             image_tr = pre_color_transform(image, mask)
#         large_scale = True
#     else:
#         large_scale = np.random.uniform() < multi_sc
#         image_tr = image
        
#     sc_fct = 0.5 if large_scale else 0.3
#     image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=True)

#     # [修改重点 2] 接收包含融合结果、多尺度预测和深层特征的全新网络输出
#     # 按照上一步 net.py 的改造，网络应返回：主输出, 尺度1-4输出, 深层特征, 附加信息
#     out0, out1, out2, out3, out4, deep_feat, hook0 = net(image)
#     out0_s, out1_s, out2_s, out3_s, out4_s, deep_feat_s, _ = net(image_scale)

#     ### Calc intra_consisten loss (entropy)
#     loss_intra = []
#     if epoch >= me_st:
#         def entrp(t):
#             etp = -(F.softmax(t, dim=1) * F.log_softmax(t, dim=1)).sum(dim=1)
#             msk = (etp < 0.5)
#             return (etp * msk).sum() / (msk.sum() or 1)
#         me = lambda x: entrp(torch.cat((x * 0, x), 1))
        
#         if not me_all:
#             e = me(out0)
#             loss_intra.append(e * get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st))
#             loss_intra = loss_intra + [0,0,0,0]
#             sw.add_scalar('intra entropy', e.item(), global_step)
#         elif me_all:
#             ga = get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st)
#             for i in [out0, out1, out2, out3, out4]:
#                 loss_intra.append(me(i) * ga)
#             sw.add_scalar('intra entropy', loss_intra[0].item(), global_step)
#     else:
#         loss_intra.extend([0 for _ in range(5)])

#     def out_proc(*args):
#         a = list(args)
#         a = [i.sigmoid() for i in a]
#         a = [torch.cat((1 - i, i), 1) for i in a]
#         return a

#     # 双通道化处理 (背景通道与前景通道)
#     out0_p, out1_p, out2_p, out3_p, out4_p = out_proc(out0, out1, out2, out3, out4)
#     out0_sp, out1_sp, out2_sp, out3_sp, out4_sp = out_proc(out0_s, out1_s, out2_s, out3_s, out4_s)

#     # 结构一致性特征提取
#     if not do_moretrsf:
#         out0_scale = F.interpolate(out0_p[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=True)
#         out0_s_ = out0_sp[:, 1:2]
#     else:
#         if pre_color_transform == None:
#             out0_ss = pre_transform(out0_p)
#         else:
#             out0_ss = out0_p
#         out0_scale = F.interpolate(out0_ss[:, 1:2], scale_factor=1.0, mode='bilinear', align_corners=True)
#         out0_s_ = F.interpolate(out0_sp[:, 1:2], scale_factor=1.0/sc_fct, mode='bilinear', align_corners=True)
        
#     try:
#         loss_ssc = Contrastive_loss(out0_s_, out0_scale.detach())
#     except NameError:
#         loss_ssc = 0.0 # 适配若原代码外部未导入 Contrastive_loss 的情况

#     gt = mask.squeeze(1).long()  
#     bg_label = gt.clone()
#     fg_label = gt.clone()
#     bg_label[gt != 0] = 255
#     fg_label[gt == 0] = 255

#     # =========================================================================
#     # # [修改重点 3] 纯端到端内部 CGAMF 共识融合机制
#     # # 提取内部 4 个不同感受野/尺度的前景预测概率
#     # probs = [torch.sigmoid(o) for o in [out1, out2, out3, out4]]
#     # weights = []
    
#     # for p in probs:
#     #     # 像素级信息熵评估：熵越小，说明网络对该像素归属越确信
#     #     entropy = - (p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
#     #     weights.append(torch.exp(-entropy))
        
#     # # 像素级权重归一化
#     # sum_w = sum(weights) + 1e-8
#     # weights_norm = [w / sum_w for w in weights]
    
#     # # 构建高共识融合伪掩码
#     # M_cgamf = sum(w * p for w, p in zip(weights_norm, probs))
#     # # 截断生成硬伪标签共识图
#     # mask_consensus = (M_cgamf > 0.5).float()
#     probs = [torch.sigmoid(o) for o in [out1, out2, out3, out4]]

#     weights = []
#     for p in probs:
#         entropy = - (p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
#         weights.append(torch.exp(-entropy))

#     sum_w = sum(weights) + 1e-8
#     weights_norm = [w / sum_w for w in weights]

#     M_cgamf = sum(w * p for w, p in zip(weights_norm, probs))

#     tau_fg = 0.7
#     tau_bg = 0.3

#     consensus_fg = (M_cgamf > tau_fg).float()
#     consensus_bg = (M_cgamf < tau_bg).float()
#     consensus_uncertain = 1.0 - consensus_fg - consensus_bg
#     # =========================================================================

#     # =========================================================================
#     # [修改重点 4] 配合 CGAMF 和深层特征的高级 ALSC 损失
#     # 将所需特征下采样到特征亲和力计算尺度 (0.25)
#     # 首先确定预测掩码下采样后的目标尺寸 (通常是 128x128)
#     out0_down = F.interpolate(out0_p[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
#     target_size = out0_down.shape[2:] # 提取 (H_down, W_down)

#     # 强制将深层语义特征和共识掩码对齐到目标尺寸，避免维度不匹配
#     deep_feat_ = F.interpolate(deep_feat, size=target_size, mode='bilinear', align_corners=True)
#     mask_consensus_ = F.interpolate(mask_consensus, size=target_size, mode='nearest')
    
#     # 传入高阶语义特征 deep_f 以替换不稳定的 RGB 像素对比
#     sample = {'deep_f': deep_feat_}
#     out0_down = F.interpolate(out0_p[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
    
#     # 计算带有共识屏障 (mask_consensus) 的不对称局部结构一致性损失
#     loss2_lsc = loss_lsc(out0_down, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, 
#                          deep_feat_.shape[2], deep_feat_.shape[3], mask_consensus=mask_consensus_)['loss']
#     # =========================================================================

#     # 构建并组合整体训练损失
#     # 1. 主输出复合损失
#     loss_main = loss_ssc + (criterion(out0_p, fg_label) + criterion(out0_p, bg_label)) + l * loss2_lsc + loss_intra[0]
    
#     # 2. 多尺度输出辅助交叉熵损失 (增强深层梯度的传递并优化内部多掩码)
#     loss_aux1 = criterion(out1_p, fg_label) + criterion(out1_p, bg_label)
#     loss_aux2 = criterion(out2_p, fg_label) + criterion(out2_p, bg_label)
#     loss_aux3 = criterion(out3_p, fg_label) + criterion(out3_p, bg_label)
#     loss_aux4 = criterion(out4_p, fg_label) + criterion(out4_p, bg_label)

#     return loss_main, loss_aux1, loss_aux2, loss_aux3, loss_aux4