"""CPU-safe loading of backbone, historical and release checkpoints."""
from pathlib import Path
import torch


def read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Checkpoint not found: {}'.format(path))
    # PyTorch 1.13 compatibility. Load only checkpoints from trusted sources.
    return torch.load(str(path), map_location='cpu')


def state_dict_from(checkpoint):
    state = checkpoint
    for name in ('model', 'state_dict', 'model_state_dict'):
        if isinstance(state, dict) and name in state:
            state = state[name]
            break
    if not isinstance(state, dict):
        raise ValueError('Expected a state_dict or a checkpoint containing model/state_dict')
    return {key[7:] if key.startswith('module.') else key: value
            for key, value in state.items()}


def load_model(model, path):
    checkpoint = read_checkpoint(path)
    model.load_state_dict(state_dict_from(checkpoint), strict=True)
    return checkpoint


def load_backbone(backbone, path):
    state = state_dict_from(read_checkpoint(path))
    if any(key.startswith('bkbone.') for key in state):
        state = {key[len('bkbone.'):]: value for key, value in state.items()
                 if key.startswith('bkbone.')}
    expected = backbone.state_dict()
    filtered = {key: value for key, value in state.items() if key in expected}
    missing = set(expected) - set(filtered)
    if missing:
        raise ValueError('Incomplete PVTv2-B4 checkpoint; missing: {}'.format(sorted(missing)[:8]))
    backbone.load_state_dict(filtered, strict=True)
    print('Loaded PVTv2-B4 backbone: {} ({} tensors)'.format(path, len(filtered)))
