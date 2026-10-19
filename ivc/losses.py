"""The full research objective, with safe handling of empty scribble classes.

The probability-input cross entropy and hard confidence partition are retained
for fidelity. They are not the partial BCE / soft SBRP described in the PDF.
"""
import math
import torch
import torch.nn.functional as F
from .augment import make_perturbation
from .feature_loss import FeatureLoss
from .propagation import (
    get_diffusion_schedule, build_seed_preserved_confidence_map,
    boundary_constrained_anisotropic_propagation_loss,
    topology_preserving_region_stabilizer,
)


def scribble_loss(logit, mask):
    p = logit.sigmoid()
    probabilities = torch.cat([1 - p, p], dim=1)
    target = mask[:, 0].long()
    losses = []
    for label in (0, 1):
        selected = target == label
        if selected.any():
            targets = torch.where(selected, target, torch.full_like(target, 255))
            losses.append(F.cross_entropy(probabilities, targets, ignore_index=255))
    return sum(losses, logit.sum() * 0.)


def training_loss(model, image, mask, epoch, total_epochs, radius=7):
    out0, out1, out2, out3, out4, deep, auxiliary = model(image, return_aux=True)
    perturbed, align = make_perturbation(image, mask)
    small = F.interpolate(perturbed, scale_factor=.5, mode='bilinear', align_corners=True)
    transformed_logits = model(small)
    student = F.interpolate(transformed_logits.sigmoid(), size=out0.shape[-2:], mode='bilinear', align_corners=True)
    consistency = F.l1_loss(student, align(out0.sigmoid()).detach())

    l_lsc, l_bcap, l_tprs, threshold = get_diffusion_schedule(epoch, total_epochs)
    confidence = build_seed_preserved_confidence_map(auxiliary['conf_logits'], mask, threshold)['mask_conf']
    target_size = deep.shape[-2:]
    prediction = F.interpolate(out0.sigmoid(), size=target_size, mode='bilinear', align_corners=True)
    boundary = auxiliary['boundary_logit'].sigmoid()
    kernels = [{'weight': 1, 'xy': 6, 'deep_f': .5}]
    sample = {'deep_f': deep}
    lsc = FeatureLoss()(prediction, kernels, radius, sample, *target_size,
                        mask_consensus=confidence)['loss']
    bcap = boundary_constrained_anisotropic_propagation_loss(
        prediction, kernels, radius, sample, *target_size, confidence, boundary)
    tprs = topology_preserving_region_stabilizer(prediction, deep, boundary, confidence)

    entropy = out0.sum() * 0.
    if epoch >= 20:
        two_logits = torch.cat([out0 * 0, out0], dim=1)
        entropy_map = -(two_logits.softmax(1) * two_logits.log_softmax(1)).sum(1)
        selected = (entropy_map < .5).float()
        raw_entropy = (entropy_map * selected).sum() / (selected.sum() + 1e-6)
        progress = min(max((epoch - 20.) / max(total_epochs - 20, 1), 0.), 1.)
        entropy = .05 * math.exp(-5 * (1 - progress) ** 2) * raw_entropy

    classification = [scribble_loss(x, mask) for x in (out0, out1, out2, out3, out4)]
    main = classification[0] + consistency + l_lsc*lsc + l_bcap*bcap + l_tprs*tprs + entropy
    total = main + sum(weight*loss for weight, loss in zip((.8, .6, .4, .2), classification[1:]))
    metrics = {'total': total, 'classification': classification[0], 'consistency': consistency,
               'lsc': lsc, 'bcap': bcap, 'tprs': tprs, 'entropy': entropy}
    return total, {key: float(value.detach()) for key, value in metrics.items()}
