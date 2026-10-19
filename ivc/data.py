"""Image/scribble data with explicit label encoding and nearest-neighbor masks."""
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MEAN = np.array([.485, .456, .406], dtype=np.float32) * 256
STD = np.array([.229, .224, .225], dtype=np.float32) * 256


def image_paths(directory, list_file=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError('Image directory not found: {}'.format(directory))
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS)
    by_stem = {}
    for path in files:
        if path.stem in by_stem:
            raise ValueError('Duplicate image stem: {}'.format(path.stem))
        by_stem[path.stem] = path
    if list_file:
        names = [Path(line.strip()).stem for line in Path(list_file).read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate names in {}'.format(list_file))
        missing = [name for name in names if name not in by_stem]
        if missing:
            raise FileNotFoundError('Missing images: {}'.format(missing[:8]))
        files = [by_stem[name] for name in names]
    if not files:
        raise ValueError('No images found in {}'.format(directory))
    return files


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Cannot decode image: {}'.format(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_tensor(image, size):
    image = (image.astype(np.float32) - MEAN) / STD
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()


def decode_scribble(mask, encoding):
    allowed = {0, 1, 2} if encoding == '012' else {0, 1, 255}
    values = set(np.unique(mask).tolist())
    if not values <= allowed:
        raise ValueError('Invalid scribble values {} for encoding {}'.format(sorted(values), encoding))
    if encoding == '012':
        decoded = np.full(mask.shape, 255, dtype=np.uint8)
        decoded[mask == 1] = 1
        decoded[mask == 2] = 0
        return decoded
    return mask


class ScribbleDataset(Dataset):
    def __init__(self, image_dir, label_dir, list_file=None, size=512, encoding='012'):
        self.images = image_paths(image_dir, list_file)
        self.label_dir = Path(label_dir)
        self.size, self.encoding = size, encoding
        missing = [p.stem for p in self.images if not (self.label_dir / (p.stem + '.png')).is_file()]
        if missing:
            raise FileNotFoundError('Missing scribble PNGs: {}'.format(missing[:8]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = read_rgb(path)
        label_path = self.label_dir / (path.stem + '.png')
        mask = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError('Cannot decode label: {}'.format(label_path))
        if mask.ndim == 3:
            if not np.all(mask == mask[:, :, :1]):
                raise ValueError('Use grayscale/index labels, not colored scribble previews: {}'.format(label_path))
            mask = mask[:, :, 0]
        if mask.shape != image.shape[:2]:
            raise ValueError('Image/label shape mismatch: {}'.format(path.name))
        mask = decode_scribble(mask, self.encoding)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if random.random() < .5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        image = image_tensor(image, self.size)
        return image, torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)
