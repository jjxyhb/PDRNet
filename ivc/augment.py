"""Device-preserving perturbations used by the original consistency objective."""
import random
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode, ColorJitter
from .data import MEAN, STD


def make_perturbation(image, mask):
    """Return a perturbed view and a matching geometry transform for predictions."""
    op = random.randrange(6)
    identity = lambda x: x
    if op == 0:
        axis = random.choice([-1, -2])
        align = lambda x: x.flip(axis)
        return align(image), align
    if op == 1:
        dx = random.choice([-1, 1]) * round(image.shape[-1] * .15)
        dy = random.choice([-1, 1]) * round(image.shape[-2] * .15)
        align = lambda x: TF.affine(x, 0, [dx, dy], 1., [0., 0.], interpolation=InterpolationMode.BILINEAR)
        return align(image), align
    if op == 2:
        h, w = image.shape[-2:]
        sh, sw = int(h * .7), int(w * .7)
        top, left = random.randrange(h - sh + 1), random.randrange(w - sw + 1)
        align = lambda x: F.interpolate(x[..., top:top+sh, left:left+sw], size=(h, w), mode='bilinear', align_corners=False)
        return align(image), align
    if op == 3:
        return TF.gaussian_blur(image, [5, 5], [random.uniform(.1, 2.)] * 2), identity
    if op == 4:
        keep = (torch.rand_like(mask.float()) < .293) | (mask != 1)
        return image * keep, identity
    # ColorJitter expects RGB in [0,1], not normalized features.
    mean = image.new_tensor(MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(STD).view(1, 3, 1, 1)
    rgb = ((image * std + mean) / 255.).clamp(0, 1)
    jitter = ColorJitter(.8, .8, .8, .2)
    rgb = torch.stack([jitter(x) for x in rgb])
    return (rgb * 255. - mean) / std, identity
