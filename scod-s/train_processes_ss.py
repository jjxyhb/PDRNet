import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from feature_loss import *
from tools import *
from utils import ramps

device = torch.device("cuda:0")

# ---------- Loss configs ----------
bce_logits = nn.BCEWithLogitsLoss(reduction='none').to(device)

loss_lsc = FeatureLoss().to(device)
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5


# ---------- helpers ----------
def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)


def Contrastive_loss(x, y):
    return torch.mean(torch.abs(x - y))


# -------------------------
# SSIM (pure torch, 1-channel)
# -------------------------
def _gaussian_window(window_size=11, sigma=1.5, device='cuda'):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = g[:, None] * g[None, :]
    w = w[None, None, :, :]  # [1,1,ws,ws]
    return w


def ssim_loss(x, y, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    """
    x, y: [B,1,H,W] in [0,1]
    return: mean( (1-ssim)/2 )
    """
    w = _gaussian_window(window_size, sigma, device=x.device)

    mu_x = F.conv2d(x, w, padding=window_size // 2)
    mu_y = F.conv2d(y, w, padding=window_size // 2)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, w, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y * y, w, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, w, padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8)
    loss = (1 - ssim_map) * 0.5
    return loss.mean()


# -------------------------
# Scribble-anchored crop (keeps output size)
# -------------------------
def scribble_anchored_crop(image, mask, crop_ratio=0.7, prefer_fg=True):
    """
    image: [B,3,H,W], mask: [B,1,H,W] with {0,1,255}
    crop around a random fg scribble pixel if exists (prefer_fg=True), else random.
    Return cropped+resized back to [H,W].
    """
    B, _, H, W = image.shape
    ch = max(1, int(H * crop_ratio))
    cw = max(1, int(W * crop_ratio))

    out_img = []
    out_msk = []

    for b in range(B):
        m = mask[b, 0]
        # candidate centers
        if prefer_fg:
            ys, xs = torch.where(m == 1)
        else:
            ys, xs = torch.where(m != 255)  # any scribble

        if ys.numel() > 0:
            idx = torch.randint(0, ys.numel(), (1,), device=image.device).item()
            cy = ys[idx].item()
            cx = xs[idx].item()
        else:
            cy = torch.randint(0, H, (1,), device=image.device).item()
            cx = torch.randint(0, W, (1,), device=image.device).item()

        y1 = int(np.clip(cy - ch // 2, 0, H - ch))
        x1 = int(np.clip(cx - cw // 2, 0, W - cw))
        y2 = y1 + ch
        x2 = x1 + cw

        img_c = image[b:b+1, :, y1:y2, x1:x2]
        msk_c = mask[b:b+1, :, y1:y2, x1:x2]

        img_c = F.interpolate(img_c, size=(H, W), mode='bilinear', align_corners=False)
        # mask use nearest to keep labels
        msk_c = F.interpolate(msk_c.float(), size=(H, W), mode='nearest').long()

        out_img.append(img_c)
        out_msk.append(msk_c)

    return torch.cat(out_img, dim=0), torch.cat(out_msk, dim=0)


def get_transform(ops=[0, 1, 2]):
    op = np.random.choice(ops)
    if op == 0:
        flip = np.random.randint(0, 2)
        return Flip(flip)
    elif op == 1:
        return Translate(0.15)
    elif op == 2:
        return Crop(0.7, 0.7)


def get_color_tranform(ops=[0, 1, 2, 3, 4, 5]):
    op = np.random.choice(ops)
    if op == 3:
        return GaussianBlur(5)
    if op == 4:
        return mask()
    if op == 5:
        return Color_jitter()
    return None


def masked_bce_logits(logit_1ch, gt_1ch):
    """
    logit_1ch: [B,1,H,W]
    gt_1ch: [B,H,W] with {0,1,255}
    returns: (loss_fg + loss_bg) where each is averaged over its valid pixels.
    """
    B, _, H, W = logit_1ch.shape
    gt = gt_1ch

    fg = (gt == 1)
    bg = (gt == 0)

    # fg loss
    if fg.any():
        lf = bce_logits(logit_1ch.squeeze(1)[fg], torch.ones_like(logit_1ch.squeeze(1)[fg]))
        lf = lf.mean()
    else:
        lf = logit_1ch.sum() * 0.0

    # bg loss
    if bg.any():
        lb = bce_logits(logit_1ch.squeeze(1)[bg], torch.zeros_like(logit_1ch.squeeze(1)[bg]))
        lb = lb.mean()
    else:
        lb = logit_1ch.sum() * 0.0

    return lf + lb


def confidence_weight(p_teacher, gamma=2.0):
    """
    p_teacher: [B,1,H,W] in [0,1]
    weight high when confident (near 0 or 1)
    """
    w = (2.0 * torch.abs(p_teacher - 0.5)).clamp(0, 1)  # [0,1]
    return w ** gamma


def psta_loss(student_p, teacher_p, w_conf=None, use_ssim=True, use_l1=True):
    """
    student_p, teacher_p: [B,1,H,W] in [0,1]
    w_conf: [B,1,H,W] or None
    """
    loss = 0.0
    if use_ssim:
        loss = loss + ssim_loss(student_p, teacher_p)
    if use_l1:
        l1 = torch.abs(student_p - teacher_p)
        if w_conf is not None:
            l1 = l1 * w_conf
            loss = loss + l1.sum() / (w_conf.sum() + 1e-6)
        else:
            loss = loss + l1.mean()
    return loss


def train_loss(image, mask, net, ctx, ft_dct,
               w_ft=.1, ft_st=2, ft_fct=.5, ft_head=True,
               mtrsf_prob=1, ops=[0, 1, 2],
               w_l2g=0, l_me=0.1, me_st=50, me_all=False,
               multi_sc=0, l=0.3, sl=1.0,
               psta_ramp=30,  # ramp-up epochs for PSTA weight
               gamma_conf=2.0):
    """
    sl: used as base weight of PSTA (kept signature compatible with your call)
    """

    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']
    else:
        epoch, global_step, t_epo = 1, 0, 60
        sw = None

    # -------------------------
    # 1) Build transformed image_tr (harder sample)
    # -------------------------
    do_moretrsf = np.random.uniform() < mtrsf_prob
    image_tr = image
    mask_tr = mask

    if do_moretrsf:
        pre_color_transform = get_color_tranform()
        if pre_color_transform is None:
            pre_transform = get_transform(ops)

            # if crop op is chosen, use scribble-anchored crop instead of random Crop
            if isinstance(pre_transform, Crop):
                image_tr, mask_tr = scribble_anchored_crop(image, mask, crop_ratio=0.7, prefer_fg=True)
            else:
                image_tr = pre_transform(image)
                mask_tr = mask  # keep original scribble positions for loss
        else:
            image_tr = pre_color_transform(image, mask)
            mask_tr = mask

    # -------------------------
    # 2) Multi-scale chain: full -> mid -> low
    # -------------------------
    # full size
    logit_full, _, _, _, _, _, _ = net(image_tr)

    # mid and low images
    image_mid = F.interpolate(image_tr, scale_factor=0.5, mode='bilinear', align_corners=False)
    image_low = F.interpolate(image_tr, scale_factor=0.25, mode='bilinear', align_corners=False)

    logit_mid, _, _, _, _, _, _ = net(image_mid)
    logit_low, _, _, _, _, _, _ = net(image_low)

    # probabilities
    p_full = torch.sigmoid(logit_full)
    p_mid = torch.sigmoid(logit_mid)
    p_low = torch.sigmoid(logit_low)

    # -------------------------
    # 3) Scribble supervision (correct & strong): BCEWithLogits on labeled pixels
    # -------------------------
    gt = mask_tr.squeeze(1).long()  # [B,H,W] {0,1,255}
    loss_sup = masked_bce_logits(logit_full, gt)

    # -------------------------
    # 4) LSC (use 2-class prob at 0.25 scale)
    # -------------------------
    image_ = F.interpolate(image_tr, scale_factor=0.25, mode='bilinear', align_corners=False)
    sample = {'rgb': image_}

    p_full_025 = F.interpolate(p_full, scale_factor=0.25, mode='bilinear', align_corners=False)
    prob2 = torch.cat([1 - p_full_025, p_full_025], dim=1)  # [B,2,h,w]
    loss_lsc_val = loss_lsc(prob2, loss_lsc_kernels_desc_defaults, loss_lsc_radius,
                            sample, image_.shape[2], image_.shape[3])['loss']

    # -------------------------
    # 5) PSTA: SSIM + SmoothL1 with confidence weighting, chain full->mid->low
    # -------------------------
    with torch.no_grad():
        # teacher for mid: downsample full
        p_full_to_mid = F.interpolate(p_full, size=p_mid.shape[2:], mode='bilinear', align_corners=False)
        w_mid = confidence_weight(p_full_to_mid, gamma=gamma_conf)

        # teacher for low: use mid (detach) to mimic chain
        p_mid_det = p_mid.detach()
        p_mid_to_low = F.interpolate(p_mid_det, size=p_low.shape[2:], mode='bilinear', align_corners=False)
        w_low = confidence_weight(p_mid_to_low, gamma=gamma_conf)

    loss_psta_mid = psta_loss(p_mid, p_full_to_mid.detach(), w_conf=w_mid, use_ssim=True, use_l1=True)
    loss_psta_low = psta_loss(p_low, p_mid_to_low.detach(), w_conf=w_low, use_ssim=True, use_l1=True)
    loss_psta = loss_psta_mid + loss_psta_low

    # ramp-up PSTA weight
    w_psta = get_current_consistency_weight(epoch, consistency=sl, consistency_rampup=psta_ramp)

    # -------------------------
    # 6) Entropy regularization (late stage) on full prediction
    # -------------------------
    loss_intra = 0.0
    if epoch >= me_st:
        # build 2-class logits from 1ch logit: [-x, x]
        logits2 = torch.cat([-logit_full, logit_full], dim=1)  # [B,2,H,W]
        prob = F.softmax(logits2, dim=1)
        ent = -(prob * torch.log(prob + 1e-8)).sum(dim=1)  # [B,H,W]
        msk = (ent < 0.5)
        ent_val = (ent * msk).sum() / (msk.sum() + 1e-6)

        loss_intra = ent_val * get_current_consistency_weight(epoch - me_st, consistency=l_me, consistency_rampup=max(1, t_epo - me_st))
        if sw is not None:
            sw.add_scalar('intra_entropy', ent_val.item(), global_step)

    # -------------------------
    # 7) (Optional) keep your old SSC as a lightweight term (now in prob space)
    #     - compare p_mid upsampled vs p_full downsampled (already inside PSTA)
    #     - we can keep an extra simple L1 on full vs resized full for stability
    # -------------------------
    # Here: minimal, avoid double-counting too hard
    loss_ssc = torch.mean(torch.abs(p_full_to_mid.detach() - p_mid))

    # -------------------------
    # 8) Total
    # -------------------------
    loss2 = loss_sup + l * loss_lsc_val + w_psta * loss_psta + 0.2 * loss_ssc + loss_intra

    return loss2, loss2 * 0.0, loss2 * 0.0, loss2 * 0.0, loss2 * 0.0
