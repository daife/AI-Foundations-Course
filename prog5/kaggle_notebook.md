# Kaggle notebook for experiment 5

## 1. Clone repository

```python
from pathlib import Path
import subprocess

WORKDIR = Path("/kaggle/working")
PROJECT_DIR = WORKDIR / "AI-Foundations-Course"
REPO_URL = "https://github.com/daife/AI-Foundations-Course.git"

if PROJECT_DIR.exists():
    print(f"Project already exists: {PROJECT_DIR}")
else:
    subprocess.run(["git", "clone", REPO_URL, str(PROJECT_DIR)], check=True)
```

```python
%cd /kaggle/working/AI-Foundations-Course
!ls -a
```

## 2. Check GPU

```python
!nvidia-smi
```

```python
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN:", torch.backends.cudnn.version())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

## 3. Install dependencies

```python
!pip install -q -r requirements.txt
```

```python
from pathlib import Path
import subprocess

ULTRALYTICS_DIR = Path("ultralytics")
if ULTRALYTICS_DIR.exists():
    print(f"Ultralytics source already exists: {ULTRALYTICS_DIR.resolve()}")
else:
    subprocess.run(
        ["git", "clone", "https://github.com/ultralytics/ultralytics.git", str(ULTRALYTICS_DIR)],
        check=True,
    )

cfg_dataset_dir = ULTRALYTICS_DIR / "ultralytics" / "cfg" / "datasets"
cfg_dataset_dir.mkdir(parents=True, exist_ok=True)

dataset_yaml = f"""path: {Path("prog5/VOC2007").resolve()}
train: train/images
val: val/images

nc: 20
names:
  0: aeroplane
  1: bicycle
  2: bird
  3: boat
  4: bottle
  5: bus
  6: car
  7: cat
  8: chair
  9: cow
  10: diningtable
  11: dog
  12: horse
  13: motorbike
  14: person
  15: pottedplant
  16: sheep
  17: sofa
  18: train
  19: tvmonitor
"""
(cfg_dataset_dir / "yolo.yaml").write_text(dataset_yaml, encoding="utf-8")
```

```python
from ultralytics import YOLO
import ultralytics

ultralytics.checks()
```

## 4. Check dataset structure

```python
from pathlib import Path

DATA_ROOT = Path("prog5/VOC2007")
for split in ["train", "val"]:
    images = sorted((DATA_ROOT / split / "images").glob("*"))
    labels = sorted((DATA_ROOT / split / "labels").glob("*.txt"))
    print(split, "images:", len(images), "labels:", len(labels))
```

```python
!sed -n '1,80p' prog5/voc2007.yaml
```

```python
!sed -n '1,80p' ultralytics/ultralytics/cfg/datasets/yolo.yaml
```

```python
!grep -n "nc:" ultralytics/ultralytics/cfg/models/v5/yolov5.yaml | head
```

```python
from pathlib import Path

model_yaml = Path("ultralytics/ultralytics/cfg/models/v5/yolov5.yaml")
lines = model_yaml.read_text(encoding="utf-8").splitlines()
lines = [
    "nc: 20 # number of classes"
    if line.strip().startswith("nc:")
    else line
    for line in lines
]
model_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
```

```python
!grep -n "nc:" ultralytics/ultralytics/cfg/models/v5/yolov5.yaml | head
```

## 5. Train YOLOv5

Formal training for the report:

```python
!python prog5/train_yolov5_voc.py
```

The command above corresponds to:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/ultralytics/cfg/models/v5/yolov5.yaml")
model.train(
    data="ultralytics/ultralytics/cfg/datasets/yolo.yaml",
    epochs=200,
    batch=24,
    imgsz=640,
    workers=8,
    degrees=1,
    shear=0.5,
    perspective=0,
    flipud=0.5,
    mixup=0.5,
    pretrained=False,
    scale=0.5,
    multi_scale=True,
    save_period=1,
    device=0,
)
```

## 6. TensorBoard

```python
%load_ext tensorboard
%tensorboard --logdir runs/detect/train
```

## 7. View Training Figures

```python
from pathlib import Path

RUN_DIR = Path("runs/detect/train")
print("Run directory:", RUN_DIR.resolve())
for path in sorted(RUN_DIR.glob("*")):
    print(path)
```

```python
from IPython.display import Image, display

for name in [
    "results.png",
    "labels.jpg",
    "labels_correlogram.jpg",
    "confusion_matrix.png",
    "F1_curve.png",
    "P_curve.png",
    "R_curve.png",
    "PR_curve.png",
]:
    image_path = RUN_DIR / name
    if image_path.exists():
        print(name)
        display(Image(filename=str(image_path)))
```
