# HPCMamba

HPCMamba: Hybrid Parallel CNN-Mamba Network for Driver State Detection

This repository contains the cleaned implementation used for HPCMamba experiments. The model definition is available at:

```text
ultralytics/cfg/models/HPCMamba/HPCMamba.yaml
```


## Main Files

```text
HPCMamba/
|-- hpcmamba_train.py
|-- selective_scan/
`-- ultralytics/
    |-- cfg/models/HPCMamba/HPCMamba.yaml
    `-- nn/modules/
        |-- hpcmamba.py
        |-- common_utils_hpcmamba.py
        `-- block.py
```


## Installation

The code was developed and tested in a CUDA-enabled Linux environment. We recommend using a fresh conda environment.

```bash
conda create -n hpcmamba python=3.11 -y
conda activate hpcmamba
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch==2.3.0 torchvision torchaudio
```

Install the remaining dependencies:

```bash
pip install seaborn thop timm einops
cd selective_scan
pip install .
cd ..
pip install -e .
```

If your CUDA, PyTorch, or compiler versions differ, rebuild `selective_scan` after activating the target environment.

## Dataset

HPCMamba follows the standard Ultralytics dataset YAML format.

Example COCO-style dataset YAML:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: class_0
  1: class_1
```

For COCO, the default config is:

```text
ultralytics/cfg/datasets/coco.yaml
```

## Training

Run training from the repository root:

```bash
python hpcmamba_train.py \
  --task train \
  --data ultralytics/cfg/datasets/coco.yaml \
  --config ultralytics/cfg/models/HPCMamba/HPCMamba.yaml \
  --imgsz 640 \
  --epochs 300 \
  --batch_size 32 \
  --device 0 \
  --workers 8 \
  --amp \
  --project output_dir/coco \
  --name hpcmamba
```


## Validation

```bash
python hpcmamba_train.py \
  --task val \
  --data ultralytics/cfg/datasets/coco.yaml \
  --config output_dir/coco/hpcmamba/weights/best.pt \
  --imgsz 640 \
  --device 0 \
  --project output_dir/coco \
  --name hpcmamba_val
```

To validate a trained checkpoint with the Ultralytics Python API:

```python
from ultralytics import YOLO

model = YOLO("output_dir/coco/hpcmamba/weights/best.pt")
metrics = model.val(data="ultralytics/cfg/datasets/coco.yaml", imgsz=640, device=0)
print(metrics)
```

## Export

Export to ONNX:

```python
from ultralytics import YOLO

model = YOLO("output_dir/coco/hpcmamba/weights/best.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

Export to TensorRT engine:

```python
from ultralytics import YOLO

model = YOLO("output_dir/coco/hpcmamba/weights/best.pt")
model.export(format="engine", imgsz=640, half=True, device=0)
```


## Acknowledgement

This project is built on top of:

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- [VMamba selective scan](https://github.com/MzeroMiko/VMamba)

We thank the authors of these open-source projects for their excellent work.

## License

This repository follows the license terms of the upstream Ultralytics YOLO codebase. Please check `LICENSE` for details.
