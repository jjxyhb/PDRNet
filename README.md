**PD-RPNet: Progressive Decoding with Reliable Propagation for Scribble-Supervised Camouflaged Object Detection**.


## dataset

/path/to/CodDataset/
├── train.txt                
├── train/
│   ├── Imgs/
│   ├── Label/               
│   └── Label_scribble/      
└── test/
    ├── CAMO/              
    ├── COD10K/
    └── NC4K/        

## Contents

```text
IVC_ck/
├── train.py                   # Full training + epoch checkpoints + resume
├── infer.py                   # Single image / directory inference, no GT needed
├── config.json                # Existing server training defaults
├── requirements.txt
├── environment.yml
├── REPRODUCTION.md            # 中文完整复现步骤、预训练权重与超参数
├── VALIDATION.md              # Release verification and practical limits
├── THIRD_PARTY_NOTICES.md
├── third_party/               # Preserved upstream license notices
├── scripts/
│   ├── check_data.py
│   └── download_backbone.py
└── ivc/
    ├── model.py
    ├── backbone.py
    ├── checkpoint.py
    ├── data.py
    ├── augment.py
    ├── losses.py
    ├── propagation.py
    ├── feature_loss.py
    └── runtime.py
```

Datasets, pretrained/trained weights, saved predictions, training logs and ablation
variants are not distributed. External data and checkpoint paths are command-line
arguments. A CUDA-enabled torchvision build with `DeformConv2d` support is needed
for GPU training/inference.

## config.json
|---|---|
| backbone | ImageNet  PVTv2-B4 |
| epochs / batch | 120 / 8 |
| image size | 512 × 512 |
| optimizer | SGD + Nesterov，FP32 |
| momentum / weight decay | 0.9 / 5e-4 |
| triangular base / peak LR | 1e-5 / 1e-2 |
| spatial / feature bandwidth | 6 / 0.5 |
| auxiliary output weights | 0.8 / 0.6 / 0.4 / 0.2，main 1 |
| entropy start / max weight | epoch 20 / 0.05 |
| seed | 2026 |

## Quick start

```bash
conda env create -f environment.yml
conda activate ivc
python scripts/download_backbone.py

# Download and arrange the dataset yourself, then validate the class-index masks.
python scripts/check_data.py --data-root /path/to/CodDataset --label-dir Label
python train.py --data-root /path/to/CodDataset \
  --backbone-weights pretrained/pvt_v2_b4.pth --output runs/main

python infer.py --checkpoint runs/main/last.pth \
  --input /path/to/CAMO/Imgs --output predictions/CAMO
```

Only an ImageNet backbone is needed for training from scratch. Inference needs a
**complete segmentation checkpoint**; the ImageNet backbone alone cannot predict
camouflaged-object masks. Existing historical `module.*` state_dict checkpoints
can also be loaded directly for inference, without the backbone checkpoint.

For original S-COD scribbles use `--label-dir Label_scribble --label-encoding 012`.
The original server default `Label` may contain expanded labels and is not
equivalent to the raw scribbles; its provenance is required for comparable runs.

## Attribution

Backbone: [PVTv2](https://github.com/whai362/PVT).
Training foundations: [CRNet / S-COD](https://github.com/dddraxxx/Weakly-Supervised-Camouflaged-Object-Detection-with-Scribble-Annotations),
PCOD ("Just a Hint: Point-Supervised Camouflaged Object Detection", ECCV 2024), and
[SCWSSOD](https://github.com/siyueyu/SCWSSOD).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for provenance and upstream terms.
