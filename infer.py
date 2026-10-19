"""Predict original-resolution PNG masks from a model checkpoint (no GT needed)."""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ivc.checkpoint import load_model
from ivc.data import EXTENSIONS, image_paths, image_tensor, read_rgb
from ivc.model import Net
from ivc.runtime import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='One image or a directory of images')
    parser.add_argument('--output', required=True)
    parser.add_argument('--list', help='Optional image IDs when --input is a directory')
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.image_size < 64 or args.image_size % 32:
        parser.error('--image-size must be a multiple of 32 and >=64')
    device = get_device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(4)
    cv2.setNumThreads(0)
    model = Net()
    load_model(model, args.checkpoint)
    model.to(device).eval()
    source, output = Path(args.input), Path(args.output)
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            parser.error('--input must be an image or image directory')
        paths = [source]
    else:
        paths = image_paths(source, args.list)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, path in enumerate(paths):
            image = read_rgb(path)
            x = image_tensor(image, args.image_size).unsqueeze(0).to(device)
            prediction = model(x, shape=image.shape[:2])[0, 0].sigmoid()
            prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
            array = (prediction.cpu().numpy() * 255).round().astype(np.uint8)
            target = output / (path.stem + '.png')
            if target.resolve() == path.resolve():
                raise ValueError('Output would overwrite input image; choose another --output')
            if not cv2.imwrite(str(target), array):
                raise IOError('Failed to save {}'.format(target))
            if (i+1) % 100 == 0 or i+1 == len(paths):
                print('Saved {}/{} masks to {}'.format(i+1, len(paths), output), flush=True)


if __name__ == '__main__':
    main()
