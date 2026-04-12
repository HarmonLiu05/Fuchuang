# AgeDB 人脸数据集适配实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将训练框架迁移到 AgeDB 跨年龄人脸数据集，验证时间 APN 在人脸场景的有效性

**Architecture:** 新增独立数据适配层 `data/prepare_agedb.py`，通过配置选择器接入现有训练流程，无需改动模型或损失函数

**Tech Stack:** Python 3.12, PyTorch, torchvision, yaml

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `data/prepare_agedb.py` | **新增** | AgeDB 数据加载器 + 数据集类 |
| `train_turtle.py` | **修改** | 加数据集选择器（约 10 行） |
| `configs/config_agedb_ce.yaml` | **新增** | 纯交叉熵 Baseline 配置 |
| `configs/config_agedb_apn.yaml` | **新增** | 时间 APN 配置 |
| `tests/test_agedb_loader.py` | **新增** | 测试数据加载 |

---

### Task 1: 创建 AgeDB 数据适配层

**Files:**
- Create: `data/prepare_agedb.py`
- Test: `tests/test_agedb_loader.py`

- [ ] **Step 1: 创建数据适配模块**

创建 `data/prepare_agedb.py`：

```python
"""
AgeDB 人脸数据集加载器
支持按年龄（时间）划分训练/测试集，返回时间信息用于 APN Triplet Loss

文件名格式: ID_Name_Age_gender.jpg
- identity = Name（人名）
- time_info = Age（年龄作为"时间"信息）
"""
import os
import re
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch


class AgeDBDataset(Dataset):
    """AgeDB 数据集加载器"""

    def __init__(self, config, split='train', transform=None, return_time=False, identity_map=None):
        self.config = config
        self.split = split
        self.transform = transform
        self.return_time = return_time

        root_dir = config['data']['root_dir']
        if not os.path.exists(root_dir):
            raise RuntimeError(f"AgeDB 目录不存在: {root_dir}")

        # 第一遍：扫描所有图片，按人名分组
        person_images = defaultdict(list)
        pattern = re.compile(r'^\d+_([^_]+)_(\d+)_([mf])\.jpg$')

        for filename in os.listdir(root_dir):
            if not filename.endswith('.jpg'):
                continue
            match = pattern.match(filename)
            if not match:
                continue

            name = match.group(1)
            try:
                age = int(match.group(2))
            except ValueError:
                continue

            filepath = os.path.join(root_dir, filename)
            person_images[name].append({
                'path': filepath,
                'age': age,
                'filename': filename
            })

        # 过滤：只保留图片数 >= min_samples 的个体
        min_samples = config['data'].get('min_samples_per_identity', 5)
        valid_people = {
            name: imgs for name, imgs in person_images.items()
            if len(imgs) >= min_samples
        }

        if not valid_people:
            raise RuntimeError(f"没有符合条件的个体（min_samples={min_samples}）")

        # 第二遍：按年龄排序，7:3 划分训练/测试
        if identity_map is not None:
            self.identity_map = identity_map
            self.label_counter = len(identity_map)
        else:
            self.identity_map = {}
            self.label_counter = 0

        self.image_list = []
        train_count = 0
        test_count = 0

        for name in sorted(valid_people.keys()):
            imgs = sorted(valid_people[name], key=lambda x: x['age'])
            total = len(imgs)
            split_idx = max(1, int(total * 0.7))

            if self.split == 'train':
                selected = imgs[:split_idx]
            else:
                selected = imgs[split_idx:]

            if name not in self.identity_map:
                self.identity_map[name] = self.label_counter
                self.label_counter += 1

            label = self.identity_map[name]
            for img in selected:
                self.image_list.append({
                    'path': img['path'],
                    'label': label,
                    'age': float(img['age']),
                    'identity_name': name
                })

            if self.split == 'train':
                train_count += len(selected)
            else:
                test_count += len(selected)

        self.num_identities = len(self.identity_map)
        print(f"加载 AgeDB {split} 数据集: {len(self.image_list)} 张图片, "
              f"{self.num_identities} 个个体")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        sample = self.image_list[idx]

        try:
            image = Image.open(sample['path']).convert('RGB')
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"无法加载图片 {sample['path']}: {e}")

        if self.transform:
            image = self.transform(image)

        if self.return_time:
            return image, sample['label'], sample['age']

        return image, sample['label']


def prepare_agedb_dataloaders(config, return_time=False):
    """
    准备 AgeDB 数据集的 DataLoader

    Args:
        config: 配置字典
        return_time: 是否返回时间信息（用于 APN Triplet Loss）

    Returns:
        train_loader, test_loader, num_identities, train_dataset
    """
    from data.dataset import get_train_transform, get_val_transform

    train_dataset = AgeDBDataset(
        config,
        split='train',
        transform=get_train_transform(config),
        return_time=return_time
    )

    test_dataset = AgeDBDataset(
        config,
        split='test',
        transform=get_val_transform(config),
        identity_map=train_dataset.identity_map,
        return_time=return_time
    )

    num_identities = train_dataset.num_identities

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader, num_identities, train_dataset
```

- [ ] **Step 2: 提交**

```bash
git add data/prepare_agedb.py
git commit -m "feat: 新增 AgeDB 人脸数据集适配层

- 解析文件名提取人名/年龄/性别
- 按年龄排序，7:3 划分训练/测试集
- 支持返回时间信息用于 APN Triplet Loss"
```

---

### Task 2: 在训练脚本中添加数据集选择器

**Files:**
- Modify: `train_turtle.py` (约第 310 行，数据加载处)

- [ ] **Step 1: 修改训练脚本**

找到 `prepare_turtle_dataloaders` 调用处（约在 `main()` 函数中），替换为：

```python
# 原代码:
# train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(config)

# 改为:
dataset_type = config['data'].get('dataset_type', 'turtle')
if dataset_type == 'agedb':
    from data.prepare_agedb import prepare_agedb_dataloaders
    train_loader, test_loader, num_identities, train_dataset = prepare_agedb_dataloaders(
        config, return_time=True
    )
    print(">>> 使用 AgeDB 人脸数据集")
else:
    train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(
        config, return_time=True
    )
    print(">>> 使用龟类数据集")
```

- [ ] **Step 2: 提交**

```bash
git add train_turtle.py
git commit -m "feat: 添加数据集选择器，支持 AgeDB 人脸数据集

- 通过 config['data']['dataset_type'] 选择数据集
- 默认 'turtle'，设为 'agedb' 时切换"
```

---

### Task 3: 创建 AgeDB 配置文件

**Files:**
- Create: `configs/config_agedb_ce.yaml`
- Create: `configs/config_agedb_apn.yaml`

- [ ] **Step 1: 创建纯交叉熵配置**

创建 `configs/config_agedb_ce.yaml`：

```yaml
# AgeDB 人脸数据集 - 纯交叉熵 Baseline
# 567 人，16,488 张图片，跨年龄（0-96 岁）

data:
  root_dir: "/workspace/AgeDB"
  dataset_type: "agedb"
  min_samples_per_identity: 5
  image_size: 224
  num_workers: 8

model:
  backbone: resnet101
  pretrained: true
  bottleneck_dim: 512
  use_mlp_bottleneck: true
  dropout: 0.4
  arcface_s: 30.0
  arcface_m: 0.35
  use_se_block: false
  label_smoothing: 0.1

training:
  batch_size: 64
  accumulation_steps: 1
  epochs: 80
  freeze_until_epoch: 0
  optimizer: Adam
  base_lr: 0.001
  backbone_lr: 0.0001
  weight_decay: 0.0005
  scheduler: CosineAnnealingLR
  eta_min: 0.000001
  precision: 16

  # 关闭所有三元组损失
  triplet_weight: 0.0
  triplet_margin: 0.3
  triplet_start_epoch: 999
  triplet_warmup_epochs: 0
  use_temporal_apn_triplet: false

  checkpoint_dir: "/workspace/experiments-checkpoints/agedb_ce_only"
  log_interval: 10

augmentation:
  random_horizontal_flip: 0.5
  random_affine_degrees: 15
  color_jitter_brightness: 0.3
  color_jitter_contrast: 0.3
  color_jitter_saturation: 0.2
  color_jitter_hue: 0.1
  gaussian_blur_kernel: 3
  gaussian_blur_sigma: [0.1, 2.0]
  random_erasing_p: 0.5
  random_erasing_scale: [0.02, 0.15]
  random_erasing_ratio: [0.3, 3.3]
```

- [ ] **Step 2: 创建时间 APN 配置**

创建 `configs/config_agedb_apn.yaml`：

```yaml
# AgeDB 人脸数据集 - 时间 APN Triplet Loss
# 567 人，16,488 张图片，跨年龄（0-96 岁）

data:
  root_dir: "/workspace/AgeDB"
  dataset_type: "agedb"
  min_samples_per_identity: 5
  image_size: 224
  num_workers: 8

model:
  backbone: resnet101
  pretrained: true
  bottleneck_dim: 512
  use_mlp_bottleneck: true
  dropout: 0.4
  arcface_s: 30.0
  arcface_m: 0.35
  use_se_block: false
  label_smoothing: 0.1

training:
  batch_size: 64
  accumulation_steps: 1
  epochs: 200
  freeze_until_epoch: 0
  optimizer: Adam
  base_lr: 0.001
  backbone_lr: 0.0001
  weight_decay: 0.0005
  scheduler: CosineAnnealingLR
  eta_min: 0.000001
  precision: 16

  # 时间 APN 三元组损失
  triplet_weight: 0.2
  triplet_margin: 0.3
  triplet_start_epoch: 80
  triplet_warmup_epochs: 20
  use_temporal_apn_triplet: true

  checkpoint_dir: "/workspace/experiments-checkpoints/agedb_apn"
  log_interval: 10

augmentation:
  random_horizontal_flip: 0.5
  random_affine_degrees: 15
  color_jitter_brightness: 0.3
  color_jitter_contrast: 0.3
  color_jitter_saturation: 0.2
  color_jitter_hue: 0.1
  gaussian_blur_kernel: 3
  gaussian_blur_sigma: [0.1, 2.0]
  random_erasing_p: 0.5
  random_erasing_scale: [0.02, 0.15]
  random_erasing_ratio: [0.3, 3.3]
```

- [ ] **Step 3: 提交**

```bash
git add configs/config_agedb_ce.yaml configs/config_agedb_apn.yaml
git commit -m "feat: 新增 AgeDB 配置文件（CE Baseline + 时间 APN）

- config_agedb_ce.yaml: 纯交叉熵 80 轮
- config_agedb_apn.yaml: CE + 时间 APN 200 轮，80 轮引入"
```

---

### Task 4: 测试数据加载

**Files:**
- Create: `tests/test_agedb_loader.py`

- [ ] **Step 1: 创建测试脚本**

创建 `tests/test_agedb_loader.py`：

```python
"""测试 AgeDB 数据加载器"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.prepare_agedb import AgeDBDataset, prepare_agedb_dataloaders


def test_agedb_dataset_creation():
    """测试数据集创建"""
    config = {
        'data': {
            'root_dir': '/workspace/AgeDB',
            'min_samples_per_identity': 5
        },
        'training': {'batch_size': 32},
        'augmentation': {}
    }

    train_dataset = AgeDBDataset(config, split='train', return_time=True)
    test_dataset = AgeDBDataset(config, split='test', return_time=True,
                               identity_map=train_dataset.identity_map)

    assert len(train_dataset) > 0, "训练集为空"
    assert len(test_dataset) > 0, "测试集为空"
    assert train_dataset.num_identities > 0, "个体数为 0"

    print(f"✓ 训练集: {len(train_dataset)} 张图片, {train_dataset.num_identities} 个个体")
    print(f"✓ 测试集: {len(test_dataset)} 张图片")

    # 测试返回格式
    image, label, age = train_dataset[0]
    print(f"✓ 样本: label={label}, age={age}, image_shape={image.size if hasattr(image, 'size') else 'N/A'}")


def test_agedb_dataloaders():
    """测试 DataLoader 创建"""
    config = {
        'data': {
            'root_dir': '/workspace/AgeDB',
            'min_samples_per_identity': 5,
            'num_workers': 0
        },
        'training': {'batch_size': 32},
        'augmentation': {}
    }

    train_loader, test_loader, num_ids, _ = prepare_agedb_dataloaders(
        config, return_time=True
    )

    assert len(train_loader) > 0
    assert len(test_loader) > 0
    assert num_ids > 0

    print(f"✓ DataLoader: train={len(train_loader)} batches, test={len(test_loader)} batches")

    # 测试 batch 返回
    batch = next(iter(train_loader))
    assert len(batch) == 3, f"期望返回 (image, label, age)，实际返回 {len(batch)} 个元素"
    images, labels, ages = batch
    print(f"✓ Batch: images={images.shape}, labels={labels.shape}, ages={ages.shape}")


if __name__ == '__main__':
    print("=" * 50)
    print("测试 AgeDB 数据加载器")
    print("=" * 50)

    test_agedb_dataset_creation()
    print()
    test_agedb_dataloaders()

    print()
    print("=" * 50)
    print("✓ 所有测试通过!")
```

- [ ] **Step 2: 运行测试**

```bash
cd /workspace/Fuchuang && python tests/test_agedb_loader.py
```

预期输出：
```
==================================================
测试 AgeDB 数据加载器
==================================================
加载 AgeDB train 数据集: ~11000 张图片, ~566 个个体
加载 AgeDB test 数据集: ~5000 张图片, 566 个个体
✓ 训练集: 11xxx 张图片, 566 个个体
✓ 测试集: 5xxx 张图片
✓ 样本: label=0, age=25.0, image_shape=PIL.Image.Image

✓ DataLoader: train=xxx batches, test=xxx batches
✓ Batch: images=torch.Size([32, 3, 224, 224]), labels=torch.Size([32]), ages=torch.Size([32])

==================================================
✓ 所有测试通过!
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_agedb_loader.py
git commit -m "test: 新增 AgeDB 数据加载器测试

- 测试数据集创建和 DataLoader 创建
- 验证返回格式和 batch 大小"
```

---

### Task 5: 端到端验证

**Files:**
- 无新文件

- [ ] **Step 1: 快速验证训练脚本**

```bash
cd /workspace/Fuchuang
python train_turtle.py --config configs/config_agedb_ce.yaml --epochs 2 2>&1 | tail -30
```

> 注意：这里我们只跑 2 个 epoch 来验证数据管道是否正常工作。实际训练用完整 epoch 数。

预期输出应包含：
- `>>> 使用 AgeDB 人脸数据集`
- `加载 AgeDB train 数据集: xxx 张图片, xxx 个个体`
- `加载 AgeDB test 数据集: xxx 张图片`
- 正常的训练日志

- [ ] **Step 2: 提交最终 commit**

```bash
git add -A
git commit -m "feat: AgeDB 人脸数据集适配完成

- 新增数据适配层 data/prepare_agedb.py
- 训练脚本支持数据集选择器
- 新增 2 个配置文件（CE Baseline + 时间 APN）
- 通过数据加载测试和端到端验证"
```

---

## 训练命令

```bash
# Baseline CE
python train_turtle.py --config configs/config_agedb_ce.yaml

# 时间 APN
python train_turtle.py --config configs/config_agedb_apn.yaml

# 从 checkpoint 恢复
python train_turtle.py --config configs/config_agedb_apn.yaml \
    --resume /workspace/experiments-checkpoints/agedb_apn/best_model1.pth
```
