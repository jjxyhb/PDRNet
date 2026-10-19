"""Train the complete model with configurable paths and epoch-boundary resume."""
import argparse
import json
import math
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from ivc.checkpoint import load_model
from ivc.data import ScribbleDataset, image_paths, image_tensor, read_rgb
from ivc.losses import training_loss
from ivc.model import Net
from ivc.runtime import (append_json, get_device, restore_rng, rng_state,
                         save_checkpoint, seed_everything, seed_worker, triangle_lr)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).parent / 'config.json'))
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--train-list', help='Defaults to DATA_ROOT/train.txt; absent file means all training images')
    parser.add_argument('--label-dir', default='Label')
    parser.add_argument('--label-encoding', choices=['012', '01ignore255'], default='012')
    parser.add_argument('--backbone-weights', default='pretrained/pvt_v2_b4.pth')
    parser.add_argument('--from-scratch', action='store_true', help='Debug only; do not load ImageNet pretraining')
    parser.add_argument('--resume', help='Release last.pth (model + optimizer + epoch + RNG)')
    parser.add_argument('--output', default='runs/default')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--image-size', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--base-lr', type=float, default=1e-5)
    parser.add_argument('--peak-lr', type=float, default=1e-2)
    parser.add_argument('--backbone-lr-factor', type=float, default=.1)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--radius', type=int, default=7)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--val-images', help='Optional dedicated validation split (not the benchmark test set)')
    parser.add_argument('--val-masks')
    parser.add_argument('--val-list')
    parser.add_argument('--val-every', type=int, default=5)
    preliminary, _ = parser.parse_known_args()
    config = json.loads(Path(preliminary.config).read_text(encoding='utf-8'))
    known = {action.dest for action in parser._actions}
    unknown = set(config) - known
    if unknown:
        parser.error('Unknown config keys: {}'.format(sorted(unknown)))
    parser.set_defaults(**config)
    args = parser.parse_args()
    if args.image_size < 64 or args.image_size % 32:
        parser.error('--image-size must be a multiple of 32 and >=64')
    if min(args.epochs, args.batch_size, args.radius, args.log_every, args.val_every) <= 0 or args.workers < 0:
        parser.error('Epochs, batch size, radius and logging intervals must be positive; workers >=0')
    if bool(args.val_images) != bool(args.val_masks):
        parser.error('--val-images and --val-masks must be supplied together')
    return args


@torch.no_grad()
def validate(model, args, device):
    model.eval()
    maes = []
    for path in image_paths(args.val_images, args.val_list):
        image = read_rgb(path)
        mask_path = Path(args.val_masks) / (path.stem + '.png')
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError('Missing/invalid validation mask: {}'.format(mask_path))
        x = image_tensor(image, args.image_size).unsqueeze(0).to(device)
        prediction = model(x, shape=mask.shape)[0, 0].sigmoid()
        prediction = (prediction - prediction.min()) / (prediction.max() - prediction.min() + 1e-8)
        maes.append(float(np.abs(prediction.cpu().numpy() - mask/255.).mean()))
    model.train()
    return sum(maes) / len(maes)


def main():
    args = arguments()
    seed_everything(args.seed)
    cv2.setNumThreads(0)
    device = get_device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(4)
    root, output = Path(args.data_root), Path(args.output)
    if not args.resume and output.exists() and any(output.glob('*.pth')):
        raise FileExistsError('Output already contains checkpoints; use --resume or a new --output')
    output.mkdir(parents=True, exist_ok=True)
    train_list = args.train_list or (root / 'train.txt' if (root / 'train.txt').is_file() else None)
    dataset = ScribbleDataset(root / 'train/Imgs', root / 'train' / args.label_dir,
                              train_list, args.image_size, args.label_encoding)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == 'cuda',
                        worker_init_fn=seed_worker)
    backbone_path = None if args.resume or args.from_scratch else args.backbone_weights
    model = Net(backbone_weights=backbone_path).to(device)
    base, head = [], []
    for name, parameter in model.named_parameters():
        (base if name.startswith('bkbone.') else head).append(parameter)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}],
                                lr=args.base_lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    start_epoch, global_step, best_mae = 0, 0, math.inf
    if args.resume:
        checkpoint = load_model(model, args.resume)
        required = {'optimizer', 'epoch', 'global_step', 'rng', 'args'}
        if not required <= set(checkpoint):
            raise ValueError('Resume requires a release last.pth, not a historical model-only checkpoint')
        current = vars(args)
        for key in ('epochs', 'batch_size', 'image_size', 'base_lr', 'peak_lr',
                    'backbone_lr_factor', 'momentum', 'weight_decay', 'radius',
                    'label_encoding', 'label_dir', 'seed', 'workers', 'data_root', 'train_list'):
            if current[key] != checkpoint['args'][key]:
                raise ValueError('Resume configuration mismatch for {} (restore original setting)'.format(key))
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_mae = checkpoint.get('best_mae', math.inf)
        restore_rng(checkpoint['rng'])
    (output / 'config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    print('Training {} images, {} steps/epoch, {} device'.format(len(dataset), len(loader), device), flush=True)
    if start_epoch >= args.epochs:
        print('Checkpoint has already completed the configured epochs.', flush=True)
        return
    total_steps = args.epochs * len(loader)
    model.train()
    for epoch in range(start_epoch, args.epochs):
        started, sums = time.time(), {}
        for batch_index, (image, mask) in enumerate(loader):
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lr = triangle_lr(global_step, total_steps, args.base_lr, args.peak_lr)
            optimizer.param_groups[0]['lr'] = args.backbone_lr_factor * lr
            optimizer.param_groups[1]['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            loss, values = training_loss(model, image, mask, epoch+1, args.epochs, args.radius)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss: {}'.format(values))
            loss.backward()
            optimizer.step()
            global_step += 1
            for key, value in values.items():
                sums[key] = sums.get(key, 0.) + value
            if batch_index % args.log_every == 0:
                print('epoch {}/{} step {}/{} lr {:.6g} loss {:.6f}'.format(
                    epoch+1, args.epochs, batch_index+1, len(loader), lr, values['total']), flush=True)
        record = {'epoch': epoch+1, 'global_step': global_step, 'seconds': time.time()-started,
                  'loss': {key: value/len(loader) for key, value in sums.items()}}
        improved = False
        if args.val_images and ((epoch+1) % args.val_every == 0 or epoch+1 == args.epochs):
            record['val_mae'] = validate(model, args, device)
            improved = record['val_mae'] < best_mae
            best_mae = min(best_mae, record['val_mae'])
        checkpoint = {'format_version': 1, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                      'global_step': global_step, 'best_mae': best_mae,
                      'rng': rng_state(), 'args': vars(args)}
        save_checkpoint(output / 'last.pth', checkpoint)
        if improved:
            save_checkpoint(output / 'best.pth', checkpoint)
        append_json(output / 'train.jsonl', record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
