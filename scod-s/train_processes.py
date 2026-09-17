import torch.nn.functional as F
import torch
from feature_loss import *
from tools import *
from utils import ramps
import numpy as np

device = torch.device("cuda:0")
criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').to(device)
loss_lsc = FeatureLoss().to(device)

# 使用更稳的半径，避免显存爆炸
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "deep_f": 0.5}]
loss_lsc_radius = 7
l = 0.3


def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def get_transform(ops=[0, 1, 2]):
    op = np.random.choice(ops)
    if op == 0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op == 1:
        pp = Translate(0.15)
    elif op == 2:
        pp = Crop(0.7, 0.7)
    return pp


def get_color_tranform(ops=[0, 1, 2, 3, 4, 5]):
    op = np.random.choice(ops)
    if op == 3:
        pp = GaussianBlur(5)
        return pp
    if op == 4:
        pp = mask()
        return pp
    if op == 5:
        pp = Color_jitter()
        return pp
    return None


def get_featuremap(h, x):
    w = h.weight
    b = h.bias
    c = w.shape[1]
    c1 = F.conv2d(x, w.transpose(0, 1), padding=(1, 1), groups=c)
    return c1, b


def unsymmetric_grad(x, y, calc, w1, w2):
    return calc(x, y.detach()) * w1 + calc(x.detach(), y) * w2


# =========================================================
# Schedule for new innovations
# =========================================================
def get_diffusion_schedule(epoch, total_epoch):
    """
    Schedule for:
      l_lsc  : original local structure consistency
      l_bcap : boundary-constrained anisotropic propagation
      l_tprs : topology-preserving region stabilizer
      tau_conf: reliability threshold for learned confidence field
    """
    r = float(epoch - 1) / float(max(total_epoch - 1, 1))

    if r < 0.25:
        l_lsc = 0.08
        l_bcap = 0.05
        l_tprs = 0.02
        tau_conf = 0.60
    elif r < 0.70:
        t = (r - 0.25) / (0.70 - 0.25)
        l_lsc = 0.08 + t * (0.22 - 0.08)    # 0.08 -> 0.22
        l_bcap = 0.05 + t * (0.12 - 0.05)   # 0.05 -> 0.12
        l_tprs = 0.02 + t * (0.05 - 0.02)   # 0.02 -> 0.05
        tau_conf = 0.60 - t * 0.08          # 0.60 -> 0.52
    else:
        t = (r - 0.70) / (1.00 - 0.70)
        l_lsc = 0.22 - t * 0.06             # 0.22 -> 0.16
        l_bcap = 0.12 - t * 0.04            # 0.12 -> 0.08
        l_tprs = 0.05 - t * 0.01            # 0.05 -> 0.04
        tau_conf = 0.52 + t * 0.04          # 0.52 -> 0.56

    return l_lsc, l_bcap, l_tprs, tau_conf


# =========================================================
# Innovation 1:
# Seed-Preserved Confidence Estimator (SPCE)
# =========================================================
def build_seed_preserved_confidence_map(conf_logits, scribble_mask=None, tau_conf=0.55):
    """
    conf_logits: [B,3,H,W]
        channel 0 -> reliable foreground
        channel 1 -> reliable background
        channel 2 -> uncertain
    scribble_mask: [B,1,h,w] or [B,1,H,W], values {0,1,255}
    """
    conf_prob = torch.softmax(conf_logits, dim=1)
    p_fg = conf_prob[:, 0:1]
    p_bg = conf_prob[:, 1:2]
    p_u = conf_prob[:, 2:3]

    reliable_fg = (p_fg > tau_conf) & (p_fg > p_bg) & (p_fg > p_u)
    reliable_bg = (p_bg > tau_conf) & (p_bg > p_fg) & (p_bg > p_u)

    mask_conf = torch.full_like(p_fg, 255.0)
    mask_conf[reliable_bg] = 0.0
    mask_conf[reliable_fg] = 1.0

    if scribble_mask is not None:
        if scribble_mask.shape[2:] != mask_conf.shape[2:]:
            scribble_mask = F.interpolate(
                scribble_mask.float(),
                size=mask_conf.shape[2:],
                mode='nearest'
            )

        fg_seed = (scribble_mask == 1)
        bg_seed = (scribble_mask == 0)

        mask_conf[fg_seed] = 1.0
        mask_conf[bg_seed] = 0.0

        p_fg = p_fg.clone()
        p_bg = p_bg.clone()
        p_u = p_u.clone()

        p_fg[fg_seed] = 1.0
        p_bg[fg_seed] = 0.0
        p_u[fg_seed] = 0.0

        p_fg[bg_seed] = 0.0
        p_bg[bg_seed] = 1.0
        p_u[bg_seed] = 0.0

    return {
        "mask_conf": mask_conf,   # {0,1,255}
        "p_fg": p_fg,
        "p_bg": p_bg,
        "p_u": p_u
    }


# =========================================================
# Innovation 2:
# Boundary-Constrained Anisotropic Propagation (BCAP)
# =========================================================
def boundary_constrained_anisotropic_propagation_loss(
    y_hat_softmax,
    kernels_desc,
    kernels_radius,
    sample,
    height_input,
    width_input,
    mask_confidence,
    boundary_prob
):
    """
    y_hat_softmax: [N,1,H,W] foreground probability
    mask_confidence: [N,1,H,W] in {0,1,255}
    boundary_prob: [N,1,H,W] in [0,1]
    """
    assert y_hat_softmax.dim() == 4
    N, C, H, W = y_hat_softmax.shape
    dev = y_hat_softmax.device

    if boundary_prob.shape[2:] != (H, W):
        boundary_prob = F.interpolate(boundary_prob, size=(H, W), mode='bilinear', align_corners=True)

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

    conf_unfold = FeatureLoss._unfold(mask_confidence.float(), kernels_radius)  # [N,1,K,K,H,W]
    center_conf = conf_unfold[:, :, kernels_radius, kernels_radius, :, :].view(N, 1, 1, 1, H, W)

    boundary_unfold = FeatureLoss._unfold(boundary_prob.float(), kernels_radius)
    center_boundary = boundary_unfold[:, :, kernels_radius, kernels_radius, :, :].view(N, 1, 1, 1, H, W)

    uncertain_center = (center_conf == 255).float()
    reliable_neigh = ((conf_unfold == 0) | (conf_unfold == 1)).float()
    fg_neigh = (conf_unfold == 1).float()

    # high boundary should block propagation
    boundary_barrier = 1.0 - torch.max(center_boundary, boundary_unfold)
    boundary_barrier = torch.clamp(boundary_barrier, min=0.0, max=1.0)

    weights = kernels * uncertain_center * reliable_neigh * boundary_barrier

    denom = weights.sum(dim=(2, 3))  # [N,1,H,W]
    pseudo_fg = (weights * fg_neigh).sum(dim=(2, 3)) / (denom + 1e-8)

    valid = (denom > 1e-6).float()
    loss_map = torch.abs(y_hat_softmax - pseudo_fg)
    loss = (loss_map * valid).sum() / (valid.sum() + 1e-6)

    return loss


# =========================================================
# Innovation 3:
# Topology-Preserving Region Stabilizer (TPRS)
# =========================================================
def _soft_erode(x):
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_open(x):
    return _soft_dilate(_soft_erode(x))


def _soft_close(x):
    return _soft_erode(_soft_dilate(x))


def topology_preserving_region_stabilizer(
    y_hat_softmax,
    feat,
    boundary_prob=None,
    mask_confidence=None
):
    """
    Two-part topology stabilizer:
      1) morphology consistency (island/hole suppression)
      2) region prototype consistency
    """
    if feat.shape[2:] != y_hat_softmax.shape[2:]:
        feat = F.interpolate(feat, size=y_hat_softmax.shape[2:], mode='bilinear', align_corners=True)

    if boundary_prob is not None and boundary_prob.shape[2:] != y_hat_softmax.shape[2:]:
        boundary_prob = F.interpolate(boundary_prob, size=y_hat_softmax.shape[2:], mode='bilinear', align_corners=True)

    # ---------- (a) morphology-based topology regularization ----------
    opened = _soft_open(y_hat_softmax)
    closed = _soft_close(y_hat_softmax)

    topo_map = F.relu(y_hat_softmax - opened) + F.relu(closed - y_hat_softmax)

    if boundary_prob is not None:
        topo_weight = 1.0 - boundary_prob.detach()
        topo_map = topo_map * topo_weight

    loss_morph = topo_map.mean()

    # ---------- (b) region prototype consistency ----------
    feat_n = F.normalize(feat, p=2, dim=1)

    if mask_confidence is not None and mask_confidence.shape[2:] != feat_n.shape[2:]:
        mask_confidence = F.interpolate(mask_confidence.float(), size=feat_n.shape[2:], mode='nearest')

    if mask_confidence is not None:
        reliable_fg = (mask_confidence == 1).float()
        reliable_bg = (mask_confidence == 0).float()
    else:
        reliable_fg = (y_hat_softmax > 0.7).float()
        reliable_bg = (y_hat_softmax < 0.3).float()

    if reliable_fg.sum().item() > 1 and reliable_bg.sum().item() > 1:
        proto_fg = (feat_n * reliable_fg).sum(dim=(2, 3), keepdim=True) / (reliable_fg.sum(dim=(2, 3), keepdim=True) + 1e-6)
        proto_bg = (feat_n * reliable_bg).sum(dim=(2, 3), keepdim=True) / (reliable_bg.sum(dim=(2, 3), keepdim=True) + 1e-6)

        proto_fg = F.normalize(proto_fg, p=2, dim=1)
        proto_bg = F.normalize(proto_bg, p=2, dim=1)

        sim_fg = (feat_n * proto_fg).sum(dim=1, keepdim=True)
        sim_bg = (feat_n * proto_bg).sum(dim=1, keepdim=True)

        proto_loss_map = y_hat_softmax * (1.0 - sim_fg) + (1.0 - y_hat_softmax) * (1.0 - sim_bg)

        if boundary_prob is not None:
            proto_loss_map = proto_loss_map * (1.0 - boundary_prob.detach())

        loss_proto = proto_loss_map.mean()
    else:
        loss_proto = y_hat_softmax.new_tensor(0.0)

    return loss_morph + 0.5 * loss_proto


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
    # Schedule
    # =========================================================
    l_lsc, l_bcap, l_tprs, tau_conf = get_diffusion_schedule(epoch, t_epo)

    # =========================================================
    # Original saliency structure consistency loss
    # =========================================================
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
    out0, out1, out2, out3, out4, deep_feat, aux_dict = net(image)
    out0_s, out1_s, out2_s, out3_s, out4_s, deep_feat_s, aux_dict_s = net(image_scale)

    boundary_logit = aux_dict["boundary_logit"]
    conf_logits = aux_dict["conf_logits"]

    # =========================================================
    # Original intra consistency
    # =========================================================
    loss_intra = []
    if epoch >= me_st:
        def entrp(t):
            etp = -(F.softmax(t, dim=1) * F.log_softmax(t, dim=1)).sum(dim=1)
            msk = (etp < 0.5)
            return (etp * msk).sum() / (msk.sum() + 1e-6)

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

    # =========================================================
    # Original structure consistency
    # =========================================================
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
    # Innovation 1: learned seed-preserved confidence map
    # =========================================================
    conf_out = build_seed_preserved_confidence_map(
        conf_logits,
        scribble_mask=mask,
        tau_conf=tau_conf
    )
    mask_confidence = conf_out["mask_conf"]

    # =========================================================
    # Prepare low-resolution maps for propagation / stabilizer
    # =========================================================
    target_size = deep_feat.shape[2:]   # deep_mix resolution (normally 1/16)

    deep_feat_ = deep_feat
    sample = {'deep_f': deep_feat_}

    out0_down = F.interpolate(
        out0_p[:, 1:2],
        size=target_size,
        mode='bilinear',
        align_corners=True
    )

    boundary_prob = torch.sigmoid(boundary_logit)
    if boundary_prob.shape[2:] != target_size:
        boundary_prob = F.interpolate(boundary_prob, size=target_size, mode='bilinear', align_corners=True)

    if mask_confidence.shape[2:] != target_size:
        mask_confidence = F.interpolate(mask_confidence.float(), size=target_size, mode='nearest')

    # =========================================================
    # Original LSC, but now guided by learned confidence field
    # =========================================================
    loss2_lsc = loss_lsc(
        out0_down,
        loss_lsc_kernels_desc_defaults,
        loss_lsc_radius,
        sample,
        deep_feat_.shape[2],
        deep_feat_.shape[3],
        mask_consensus=mask_confidence
    )['loss']

    # =========================================================
    # Innovation 2: BCAP
    # =========================================================
    loss2_bcap = boundary_constrained_anisotropic_propagation_loss(
        out0_down,
        loss_lsc_kernels_desc_defaults,
        loss_lsc_radius,
        sample,
        deep_feat_.shape[2],
        deep_feat_.shape[3],
        mask_confidence,
        boundary_prob
    )

    # =========================================================
    # Innovation 3: TPRS
    # =========================================================
    loss2_tprs = topology_preserving_region_stabilizer(
        out0_down,
        deep_feat_,
        boundary_prob=boundary_prob,
        mask_confidence=mask_confidence
    )

    # =========================================================
    # Total loss
    # 非扩散原始监督不动，只在结构传播支路增加新损失
    # =========================================================
    loss_main = (
        loss_ssc
        + (criterion(out0_p, fg_label) + criterion(out0_p, bg_label))
        + l_lsc * loss2_lsc
        + l_bcap * loss2_bcap
        + l_tprs * loss2_tprs
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
        sw.add_scalar('sched/l_bcap', l_bcap, global_step)
        sw.add_scalar('sched/l_tprs', l_tprs, global_step)
        sw.add_scalar('sched/tau_conf', tau_conf, global_step)

        with torch.no_grad():
            fg_ratio = (mask_confidence == 1).float().mean()
            bg_ratio = (mask_confidence == 0).float().mean()
            uncertain_ratio = (mask_confidence == 255).float().mean()
            boundary_mean = boundary_prob.mean()

        sw.add_scalar('confidence/fg_ratio', fg_ratio.item(), global_step)
        sw.add_scalar('confidence/bg_ratio', bg_ratio.item(), global_step)
        sw.add_scalar('confidence/uncertain_ratio', uncertain_ratio.item(), global_step)
        sw.add_scalar('boundary/mean', boundary_mean.item(), global_step)

        sw.add_scalar('loss/lsc', loss2_lsc.item(), global_step)
        sw.add_scalar('loss/bcap', loss2_bcap.item(), global_step)
        sw.add_scalar('loss/tprs', loss2_tprs.item(), global_step)

    return loss_main, loss_aux1, loss_aux2, loss_aux3, loss_aux4