# Propagation functions retained from the research implementation.
# Hard confidence partitioning is intentional for source fidelity.
import torch
import torch.nn.functional as F
from .feature_loss import FeatureLoss

def get_diffusion_schedule(epoch, total_epoch):
    r = float(epoch - 1) / float(max(total_epoch - 1, 1))
    if r < 0.25:
        l_lsc = 0.08
        l_bcap = 0.05
        l_tprs = 0.02
        tau_conf = 0.6
    elif r < 0.7:
        t = (r - 0.25) / (0.7 - 0.25)
        l_lsc = 0.08 + t * (0.22 - 0.08)
        l_bcap = 0.05 + t * (0.12 - 0.05)
        l_tprs = 0.02 + t * (0.05 - 0.02)
        tau_conf = 0.6 - t * 0.08
    else:
        t = (r - 0.7) / (1.0 - 0.7)
        l_lsc = 0.22 - t * 0.06
        l_bcap = 0.12 - t * 0.04
        l_tprs = 0.05 - t * 0.01
        tau_conf = 0.52 + t * 0.04
    return (l_lsc, l_bcap, l_tprs, tau_conf)

def build_seed_preserved_confidence_map(conf_logits, scribble_mask=None, tau_conf=0.55):
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
            scribble_mask = F.interpolate(scribble_mask.float(), size=mask_conf.shape[2:], mode='nearest')
        fg_seed = scribble_mask == 1
        bg_seed = scribble_mask == 0
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
    return {'mask_conf': mask_conf, 'p_fg': p_fg, 'p_bg': p_bg, 'p_u': p_u}

def boundary_constrained_anisotropic_propagation_loss(y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input, mask_confidence, boundary_prob):
    assert y_hat_softmax.dim() == 4
    N, C, H, W = y_hat_softmax.shape
    dev = y_hat_softmax.device
    if boundary_prob.shape[2:] != (H, W):
        boundary_prob = F.interpolate(boundary_prob, size=(H, W), mode='bilinear', align_corners=True)
    kernels = FeatureLoss._create_kernels(kernels_desc, kernels_radius, sample, N, H, W, dev, custom_modality_downsamplers=None)
    conf_unfold = FeatureLoss._unfold(mask_confidence.float(), kernels_radius)
    center_conf = conf_unfold[:, :, kernels_radius, kernels_radius, :, :].view(N, 1, 1, 1, H, W)
    boundary_unfold = FeatureLoss._unfold(boundary_prob.float(), kernels_radius)
    center_boundary = boundary_unfold[:, :, kernels_radius, kernels_radius, :, :].view(N, 1, 1, 1, H, W)
    uncertain_center = (center_conf == 255).float()
    reliable_neigh = ((conf_unfold == 0) | (conf_unfold == 1)).float()
    fg_neigh = (conf_unfold == 1).float()
    boundary_barrier = 1.0 - torch.max(center_boundary, boundary_unfold)
    boundary_barrier = torch.clamp(boundary_barrier, min=0.0, max=1.0)
    weights = kernels * uncertain_center * reliable_neigh * boundary_barrier
    denom = weights.sum(dim=(2, 3))
    pseudo_fg = (weights * fg_neigh).sum(dim=(2, 3)) / (denom + 1e-08)
    valid = (denom > 1e-06).float()
    loss_map = torch.abs(y_hat_softmax - pseudo_fg)
    loss = (loss_map * valid).sum() / (valid.sum() + 1e-06)
    return loss

def _soft_erode(x):
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)

def _soft_dilate(x):
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

def _soft_open(x):
    return _soft_dilate(_soft_erode(x))

def _soft_close(x):
    return _soft_erode(_soft_dilate(x))

def topology_preserving_region_stabilizer(y_hat_softmax, feat, boundary_prob=None, mask_confidence=None):
    if feat.shape[2:] != y_hat_softmax.shape[2:]:
        feat = F.interpolate(feat, size=y_hat_softmax.shape[2:], mode='bilinear', align_corners=True)
    if boundary_prob is not None and boundary_prob.shape[2:] != y_hat_softmax.shape[2:]:
        boundary_prob = F.interpolate(boundary_prob, size=y_hat_softmax.shape[2:], mode='bilinear', align_corners=True)
    opened = _soft_open(y_hat_softmax)
    closed = _soft_close(y_hat_softmax)
    topo_map = F.relu(y_hat_softmax - opened) + F.relu(closed - y_hat_softmax)
    if boundary_prob is not None:
        topo_weight = 1.0 - boundary_prob.detach()
        topo_map = topo_map * topo_weight
    loss_morph = topo_map.mean()
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
        proto_fg = (feat_n * reliable_fg).sum(dim=(2, 3), keepdim=True) / (reliable_fg.sum(dim=(2, 3), keepdim=True) + 1e-06)
        proto_bg = (feat_n * reliable_bg).sum(dim=(2, 3), keepdim=True) / (reliable_bg.sum(dim=(2, 3), keepdim=True) + 1e-06)
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
