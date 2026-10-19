"""Reproducible setup, logging and learning-rate schedule."""
import json
import random
from pathlib import Path
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def get_device(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install matching torch/torchvision CUDA builds or use --device cpu.')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    return device


def triangle_lr(step, total_steps, base_lr, peak_lr):
    # Zero-based step: first step starts at base_lr (fixes the original -1 index).
    phase = step / max(total_steps, 1)
    return base_lr + (peak_lr - base_lr) * max(0., 1. - abs(2.*phase - 1.))


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(data, str(temporary))
    temporary.replace(path)


def append_json(path, record):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + '\n')
