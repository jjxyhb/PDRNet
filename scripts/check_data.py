"""Validate training file pairs, label encodings and annotation coverage."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from ivc.data import decode_scribble, image_paths, read_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--label-dir', default='Label')
    parser.add_argument('--label-encoding', choices=['012', '01ignore255'], default='012')
    parser.add_argument('--train-list')
    args = parser.parse_args()
    root = Path(args.data_root)
    list_file = args.train_list or (root / 'train.txt' if (root / 'train.txt').exists() else None)
    images = image_paths(root / 'train/Imgs', list_file)
    totals = {'foreground': 0, 'background': 0, 'unlabeled': 0}
    no_fg = no_bg = 0
    for path in images:
        image = read_rgb(path)
        label_path = root / 'train' / args.label_dir / (path.stem + '.png')
        label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
        if label is None:
            raise FileNotFoundError('Cannot read {}'.format(label_path))
        if label.ndim == 3:
            if not np.all(label == label[:, :, :1]):
                raise ValueError('Color preview cannot be used as class-index label: {}'.format(label_path))
            label = label[:, :, 0]
        if label.shape != image.shape[:2]:
            raise ValueError('Image and label sizes differ: {}'.format(path.name))
        label = decode_scribble(label, args.label_encoding)
        fg, bg, unlabeled = [int((label == value).sum()) for value in (1, 0, 255)]
        if fg + bg == 0:
            raise ValueError('No labeled pixels: {}'.format(label_path))
        no_fg += fg == 0
        no_bg += bg == 0
        for key, count in zip(totals, [fg, bg, unlabeled]):
            totals[key] += count
    total = sum(totals.values())
    print(json.dumps({'images': len(images), 'label_dir': args.label_dir,
                      'encoding': args.label_encoding, 'pixels': totals,
                      'fractions': {key: value/total for key, value in totals.items()},
                      'images_without_foreground': no_fg,
                      'images_without_background': no_bg}, indent=2))


if __name__ == '__main__':
    main()
