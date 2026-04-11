# 龟类数据集训练系统实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为龟类数据集（Turtle Dataset）添加COCO格式数据加载器和训练脚本，支持ResNet50 + ArcFace训练。

**Architecture:** 新增COCO格式数据集加载器（TurtleDataset），完全复用现有模型架构和训练逻辑，通过配置文件驱动数据集切换。

**Tech Stack:** Python 3.x, PyTorch, torchvision, PyYAML, Pillow, COCO JSON格式

---

## 文件结构

```
新增文件:
├── configs/config_turtle.yaml          # 龟类数据集配置
├── data/turtle_dataset.py              # COCO格式数据集加载器
├── data/prepare_turtle_data.py         # 数据准备函数
├── tests/test_turtle_dataset.py        # 数据集加载器测试
└── train_turtle.py                     # 龟类训练脚本

复用文件（不修改）:
├── models/backbone.py
├── models/bottleneck.py
├── models/arcface.py
├── utils/metrics.py
├── utils/utils.py
└── data/dataset.py                     # 仅复用 get_train_transform/get_val_transform
```

---

### Task 1: 创建龟类数据集配置文件

**Files:**
- Create: `configs/config_turtle.yaml`

- [ ] **Step 1: 创建配置文件**

创建 `configs/config_turtle.yaml`，内容如下：

```yaml
data:
  root_dir: "E:/fuchuang/turtlehead-dataset/Turtel_dataset"
  splits_dir: "dataset_splits"
  dataset_name: "dataset_E_ge5years_count"
  train_json: "train.json"
  test_json: "test.json"
  image_size: 112
  num_workers: 0
  min_samples_per_identity: 5

model:
  backbone: resnet50
  pretrained: true
  bottleneck_dim: 512
  dropout: 0.4
  arcface_s: 30.0
  arcface_m: 0.35
  label_smoothing: 0.1

training:
  batch_size: 8
  accumulation_steps: 4
  epochs: 30
  freeze_until_epoch: 10
  optimizer: Adam
  base_lr: 0.001
  backbone_lr: 0.0001
  weight_decay: 0.0005
  scheduler: CosineAnnealingLR
  eta_min: 0.000001
  precision: 16
  checkpoint_dir: "checkpoints_turtle"
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

**注意**: 
- `root_dir` 使用正斜杠避免Windows转义问题
- 去掉了 `image_dir` 字段，COCO JSON的`path`已包含完整相对路径

- [ ] **Step 2: 验证配置文件可加载**

运行: `python -c "from utils.utils import load_config; print(load_config('configs/config_turtle.yaml'))"`

预期: 输出完整的配置字典

- [ ] **Step 3: 提交**

```bash
git add configs/config_turtle.yaml
git commit -m "feat: 添加龟类数据集配置文件"
```

---

### Task 2: 创建 COCO 格式数据集加载器

**Files:**
- Create: `data/turtle_dataset.py`
- Test: `tests/test_turtle_dataset.py`

- [ ] **Step 1: 创建 TurtleDataset 类**

创建 `data/turtle_dataset.py`，内容如下：

```python
"""
龟类数据集COCO格式加载器
支持读取 train.json / test.json 并按个体ID分配标签
"""
import os
import json
from PIL import Image
from torch.utils.data import Dataset


class TurtleDataset(Dataset):
    """龟类数据集COCO格式加载器"""
    
    def __init__(self, config, split='train', transform=None, identity_map=None):
        """
        Args:
            config: 配置字典
            split: 'train' 或 'test'
            transform: 数据增强transform
            identity_map: 身份映射字典（test集共享train集的映射）
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # 构建JSON路径
        splits_dir = config['data']['splits_dir']
        dataset_name = config['data']['dataset_name']
        json_file = config['data'][f'{split}_json']
        json_path = os.path.join(splits_dir, dataset_name, json_file)
        
        # 加载COCO标注
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 解析images数组
        self.image_list = []
        
        if identity_map is not None:
            # test集：共享train集的identity_map
            self.identity_map = identity_map
            self.label_counter = len(identity_map)
        else:
            # train集：创建新的identity_map
            self.identity_map = {}
            self.label_counter = 0
        
        # 第一遍：统计每个个体的图片数
        identity_counts = {}
        for image_info in coco_data['images']:
            image_path = image_info['path']
            identity_id = image_path.split('/')[0]  # 顶级目录即个体ID
            
            identity_counts[identity_id] = identity_counts.get(identity_id, 0) + 1
        
        # 过滤：只保留图片数 >= min_samples 的个体
        min_samples = config['data'].get('min_samples_per_identity', 1)
        valid_identities = {
            identity_id for identity_id, count in identity_counts.items()
            if count >= min_samples
        }
        
        # 第二遍：构建image_list（仅包含有效个体）
        for image_info in coco_data['images']:
            image_path = image_info['path']
            identity_id = image_path.split('/')[0]
            
            # 跳过图片数不足的个体
            if identity_id not in valid_identities:
                continue
            
            # 构建完整图片路径（直接拼接 root_dir + path）
            full_path = os.path.join(
                config['data']['root_dir'],
                image_path
            )
            
            # 分配标签
            if identity_id not in self.identity_map:
                self.identity_map[identity_id] = self.label_counter
                self.label_counter += 1
            
            label = self.identity_map[identity_id]
            self.image_list.append({
                'path': full_path,
                'identity_id': identity_id,
                'label': label,
                'date': image_info.get('date', None)
            })
        
        # 统计信息
        self.num_identities = len(self.identity_map)
        print(f"加载 {split} 数据集: {len(self.image_list)} 张图片, "
              f"{self.num_identities} 个个体")
        
        # 验证test集的identity_map一致性
        if split == 'test' and identity_map is not None:
            missing = set(self.identity_map.keys()) - set(identity_map.keys())
            if missing:
                print(f"警告: 测试集中存在训练集未出现的个体: {missing}")
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        sample = self.image_list[idx]
        
        # 读取图片
        image = Image.open(sample['path']).convert('RGB')
        
        # 应用transform
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']
    
    def get_identity_list(self):
        """返回所有个体ID列表"""
        return list(self.identity_map.keys())
```

- [ ] **Step 2: 创建数据集加载器测试**

创建 `tests/test_turtle_dataset.py`，内容如下：

```python
"""
龟类数据集加载器测试
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import load_config
from data.turtle_dataset import TurtleDataset


def test_turtle_dataset():
    """测试龟类数据集加载器"""
    print("=" * 60)
    print("测试龟类数据集加载器")
    print("=" * 60)
    
    # 加载配置
    config = load_config('configs/config_turtle.yaml')
    print(f"\n✓ 配置文件加载成功")
    print(f"  数据集: {config['data']['dataset_name']}")
    print(f"  根目录: {config['data']['root_dir']}")
    
    # 创建训练集（不指定transform，仅测试数据加载）
    print("\n--- 创建训练集 ---")
    train_dataset = TurtleDataset(config, split='train', transform=None)
    
    print(f"✓ 训练集创建成功")
    print(f"  图片数: {len(train_dataset)}")
    print(f"  个体数: {train_dataset.num_identities}")
    print(f"  个体列表: {train_dataset.get_identity_list()[:10]}...")  # 显示前10个
    
    # 测试获取样本
    print("\n--- 测试样本读取 ---")
    image, label = train_dataset[0]
    print(f"✓ 样本读取成功")
    print(f"  图片类型: {type(image)}")
    print(f"  标签: {label}")
    print(f"  个体ID: {train_dataset.image_list[0]['identity_id']}")
    
    # 创建测试集（共享train集的identity_map）
    print("\n--- 创建测试集（共享identity_map）---")
    test_dataset = TurtleDataset(
        config, 
        split='test', 
        transform=None,
        identity_map=train_dataset.identity_map
    )
    
    print(f"✓ 测试集创建成功")
    print(f"  图片数: {len(test_dataset)}")
    print(f"  个体数: {test_dataset.num_identities}")
    
    # 验证identity_map一致性
    train_ids = set(train_dataset.identity_map.keys())
    test_ids = set(test_dataset.identity_map.keys())
    assert test_ids.issubset(train_ids), "测试集个体必须是训练集个体的子集"
    print(f"✓ identity_map一致性验证通过")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_turtle_dataset()
```

- [ ] **Step 3: 运行测试验证数据集加载器**

运行: `python tests/test_turtle_dataset.py`

预期输出:
```
============================================================
测试龟类数据集加载器
============================================================

✓ 配置文件加载成功
  数据集: dataset_E_ge5years_count
  根目录: E:/fuchuang/turtlehead-dataset/Turtel_dataset

--- 创建训练集 ---
加载 train 数据集: XXX 张图片, XX 个个体
✓ 训练集创建成功
  图片数: XXX
  个体数: XX
  ...

--- 测试样本读取 ---
✓ 样本读取成功
  ...

--- 创建测试集（共享identity_map）---
加载 test 数据集: XXX 张图片, XX 个个体
✓ 测试集创建成功
  ...
✓ identity_map一致性验证通过

============================================================
所有测试通过！
============================================================
```

- [ ] **Step 4: 提交**

```bash
git add data/turtle_dataset.py tests/test_turtle_dataset.py
git commit -m "feat: 添加龟类COCO数据集加载器及测试"
```

---

### Task 3: 创建数据准备函数

**Files:**
- Create: `data/prepare_turtle_data.py`

- [ ] **Step 1: 创建数据准备函数**

创建 `data/prepare_turtle_data.py`，内容如下：

```python
"""
龟类数据集数据准备函数
创建 DataLoader 实例
"""
import torch
from torch.utils.data import DataLoader

from data.turtle_dataset import TurtleDataset
from data.dataset import get_train_transform, get_val_transform


def prepare_turtle_dataloaders(config):
    """
    准备龟类数据集的DataLoader
    
    Returns:
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
        num_identities: 身份数量
        train_dataset: 训练集Dataset实例
    """
    # 创建训练集
    train_dataset = TurtleDataset(
        config, 
        split='train',
        transform=get_train_transform(config)
    )
    
    # 创建测试集（共享train集的identity_map）
    test_dataset = TurtleDataset(
        config,
        split='test',
        transform=get_val_transform(config),
        identity_map=train_dataset.identity_map
    )
    
    num_identities = train_dataset.num_identities
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # 保证每个batch大小一致，最后不足batch_size的样本被丢弃
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

- [ ] **Step 2: 验证导入无误**

运行: `python -c "from data.prepare_turtle_data import prepare_turtle_dataloaders; print('导入成功')"`

预期: 输出 "导入成功"

- [ ] **Step 3: 提交**

```bash
git add data/prepare_turtle_data.py
git commit -m "feat: 添加龟类数据准备函数"
```

---

### Task 4: 创建龟类训练脚本

**Files:**
- Create: `train_turtle.py`

- [ ] **Step 1: 创建训练脚本**

创建 `train_turtle.py`，内容如下：

```python
"""
龟类个体识别训练脚本
基于 ResNet50 + ArcFace，使用COCO格式标注
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models.backbone import ResNet50Backbone
from models.bottleneck import Bottleneck
from models.arcface import ArcFace
from data.prepare_turtle_data import prepare_turtle_dataloaders
from data.dataset import get_train_transform, get_val_transform
from utils.utils import load_config, set_seed, get_device, ensure_dir
from utils.metrics import compute_all_metrics


class ChimpFaceModel(nn.Module):
    def __init__(self, config, num_identities):
        super().__init__()
        model_cfg = config['model']
        self.backbone = ResNet50Backbone(
            pretrained=model_cfg['pretrained'],
            freeze_until_layer=3
        )
        self.bottleneck = Bottleneck(
            in_features=2048,
            bottleneck_dim=model_cfg['bottleneck_dim'],
            dropout=model_cfg['dropout']
        )
        self.arcface = ArcFace(
            in_features=model_cfg['bottleneck_dim'],
            num_classes=num_identities,
            s=model_cfg['arcface_s'],
            m=model_cfg['arcface_m']
        )

    def forward(self, images, labels):
        features = self.backbone(images)
        features = self.bottleneck(features)
        output = self.arcface(features, labels)
        return output, features

    def unfreeze_backbone(self):
        self.backbone.unfreeze_all()


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    features_list, labels_list = [], []
    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        feats = model.backbone(images)
        feats = model.bottleneck(feats)
        features_list.append(feats.cpu())
        labels_list.append(labels)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return compute_all_metrics(features, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_turtle.yaml')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed)
    device = get_device()
    print(f"使用设备: {device}")
    print(f"配置文件: {args.config}")

    # 准备数据
    train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(config)
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # 模型、优化器、调度器
    model = ChimpFaceModel(config, num_identities).to(device)
    optimizer = torch.optim.Adam([
        {'params': model.backbone.layer4.parameters(), 'lr': config['training']['backbone_lr']},
        {'params': model.bottleneck.parameters(), 'lr': config['training']['base_lr']},
        {'params': model.arcface.parameters(), 'lr': config['training']['base_lr']}
    ], weight_decay=config['training']['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs'],
        eta_min=config['training']['eta_min']
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=config['model']['label_smoothing'])
    scaler = torch.amp.GradScaler('cuda')
    checkpoint_dir = config['training']['checkpoint_dir']
    ensure_dir(checkpoint_dir)

    best_acc = 0.0
    accumulation_steps = config['training']['accumulation_steps']

    print(f"\n开始训练: {config['training']['epochs']} epochs")
    print(f"Batch size: {config['training']['batch_size']}, Accumulation: {accumulation_steps}")

    for epoch in range(config['training']['epochs']):
        # === 训练 ===
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(config['training']['precision'] == 16)):
                output, features = model(images, labels)
                loss = criterion(output, labels) / accumulation_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.4f}'})

        # === 验证 ===
        metrics = evaluate(model, test_loader, device)
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        acc = metrics['accuracy']
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

        # === 保存最佳模型 ===
        if acc > best_acc:
            best_acc = acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'config': config
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  ↳ 保存最佳模型! Acc={best_acc:.4f}")

        # 阶段 2: 解冻 backbone
        if epoch + 1 == config['training'].get('freeze_until_epoch', 10):
            print(">>> 解冻 backbone layer4...")
            model.unfreeze_backbone()
            optimizer.param_groups[0]['lr'] = config['training']['base_lr'] * 0.1

    print(f"\n训练完成! 最佳 Accuracy: {best_acc:.4f}")
    print(f"模型保存在: {os.path.join(checkpoint_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 验证脚本语法**

运行: `python -m py_compile train_turtle.py`

预期: 无输出（语法正确）

- [ ] **Step 3: 提交**

```bash
git add train_turtle.py
git commit -m "feat: 添加龟类训练脚本"
```

---

### Task 5: 端到端集成测试

**Files:**
- Modify: `train_turtle.py` (验证运行)

- [ ] **Step 1: 运行1个epoch验证完整性**

运行: `python train_turtle.py --config configs/config_turtle.yaml`

预期:
- 成功加载配置
- 成功创建数据集和DataLoader
- 成功创建模型
- 完成第1个epoch的训练和验证
- 输出类似: `Epoch 1: Loss=X.XXXX, Acc=X.XXXX, LR=X.XXXXXX`

**注意**: 如果显存不足，可以:
1. 减小 `batch_size` (如改为4)
2. 增大 `accumulation_steps` (如改为8)

- [ ] **Step 2: 验证检查点保存**

检查文件是否存在: `checkpoints_turtle/best_model.pth`

运行: `python -c "import torch; ckpt = torch.load('checkpoints_turtle/best_model.pth'); print(f'Epoch: {ckpt[\"epoch\"]}, Accuracy: {ckpt[\"accuracy\"]}')"`

预期: 输出epoch和accuracy信息

- [ ] **Step 3: 提交最终版本**

```bash
git add checkpoints_turtle/.gitkeep
git commit -m "feat: 完成龟类数据集训练系统集成"
```

---

## 自审检查

### 1. 规范覆盖

- ✅ 配置文件创建
- ✅ COCO格式数据集加载器
- ✅ identity_map共享机制
- ✅ min_samples过滤逻辑
- ✅ 数据准备函数
- ✅ 训练脚本
- ✅ 测试代码

### 2. 占位符扫描

无占位符、无TODO、所有代码均完整实现

### 3. 类型一致性

- 所有函数签名一致
- 配置键名一致（`config['data']`, `config['training']`, `config['model']`）
- 返回值类型一致

---

**计划版本**: v1.0  
**最后更新**: 2026-04-10
