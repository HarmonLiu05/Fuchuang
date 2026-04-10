# 龟类数据集训练系统设计文档

**日期**: 2026-04-10  
**状态**: 待审核  
**作者**: AI Assistant

---

## 1. 概述

### 1.1 目标

为现有的 Chimpanzee ArcFace 项目添加对龟类头部数据集（Turtle Dataset）的支持，使其能够使用 ResNet50 + ArcFace 架构进行龟类个体识别训练。

### 1.2 数据集信息

- **数据集名称**: Turtle Head Dataset - dataset_E_ge5years_count
- **路径**: `E:\fuchuang\turtlehead-dataset\Turtel_dataset`
- **格式**: COCO格式标注（train.json / test.json）
- **个体数**: 38个（时间跨度≥5年）
- **训练集**: 966张图片
- **测试集**: 438张图片
- **图片路径**: `images/t{ID}/xxx.JPG`

### 1.3 核心原则

- **最大化代码复用**: 完全复用现有模型架构和训练逻辑
- **配置文件驱动**: 通过YAML配置切换数据集
- **保持COCO格式**: 不转换为ImageFolder，保留标注信息

---

## 2. 架构设计

### 2.1 文件结构

```
新增文件:
├── configs/config_turtle.yaml          # 龟类数据集配置
├── data/turtle_dataset.py              # COCO格式数据集加载器
├── data/prepare_turtle_data.py         # 数据准备函数
└── train_turtle.py                     # 龟类训练脚本

复用文件（不修改）:
├── models/backbone.py                  # ResNet50 backbone
├── models/bottleneck.py                # 瓶颈层
├── models/arcface.py                   # ArcFace Head
├── utils/metrics.py                    # 评估指标
├── utils/utils.py                      # 工具函数
└── data/dataset.py                     # 数据增强transform
```

### 2.2 数据流

```
train.json (COCO)
    ↓
TurtleDataset.__init__()
    ├─ 解析 images 数组
    ├─ 从 path 字段提取个体ID (如 "images/t007/xxx.JPG" → "t007")
    ├─ 构建 {image_path, identity_id} 映射
    └─ 创建 identity → label 映射 (t007→0, t008→1, ...)
    ↓
TurtleDataset.__getitem__(idx)
    ├─ 读取图片 (完整路径 = root_dir + "/" + image_dir + "/" + path)
    ├─ 应用 transform (数据增强)
    └─ 返回 (image_tensor, label)
    ↓
DataLoader (batch_size=8, shuffle=True)
    ↓
ChimpFaceModel (ResNet50 + ArcFace)
    ↓
训练循环 (完全复用 train_split.py 逻辑)
```

---

## 3. 组件详细设计

### 3.1 配置文件 `configs/config_turtle.yaml`

```yaml
data:
  root_dir: "E:\\fuchuang\\turtlehead-dataset\\Turtel_dataset"
  splits_dir: "dataset_splits"
  dataset_name: "dataset_E_ge5years_count"
  train_json: "train.json"
  test_json: "test.json"
  image_size: 112
  num_workers: 0
  min_samples_per_identity: 5
```

**关键调整**:
- **去掉 `image_dir` 配置**: COCO JSON中`path`字段已包含完整相对路径（如`images/t007/xxx.JPG`），直接使用`root_dir + path`拼接
- `batch_size: 8`（原4）- 38个个体足够大
- `accumulation_steps: 4`（原8）- 保持等效batch_size=32
- `min_samples_per_identity: 5` - 过滤图片数少于5的个体
- `checkpoint_dir: "checkpoints_turtle"` - 独立的检查点目录

---

### 3.2 数据集加载器 `data/turtle_dataset.py`

#### 类定义

```python
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
        with open(json_path, 'r') as f:
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

**关键实现点**:
- **路径拼接**: 直接使用`root_dir + image_path`，不额外拼接`image_dir`
- **identity_map共享**: test集通过参数接收train集的identity_map，保证标签一致
- **min_samples过滤**: 在`__init__`末尾过滤掉图片数少于阈值的个体
- 从COCO JSON的`images[].path`字段提取个体ID（`path.split('/')[0]`）
- 支持transform（复用现有的get_train_transform/get_val_transform）
- 保留date字段供未来时间序列分析使用

---

### 3.3 数据准备函数 `data/prepare_turtle_data.py`

```python
def prepare_turtle_dataloaders(config):
    """
    准备龟类数据集的DataLoader
    
    Returns:
        train_loader: 训练集DataLoader
        test_loader: 测试集DataLoader
        num_identities: 身份数量
        train_dataset: 训练集Dataset实例
    """
    from data.turtle_dataset import TurtleDataset
    from data.dataset import get_train_transform, get_val_transform
    
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

---

### 3.4 训练脚本 `train_turtle.py`

与 `train_split.py` 的差异仅在于导入和数据准备部分：

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

# === 以下完全复用 train_split.py 的 ChimpFaceModel 类和训练逻辑 ===
# ... (ChimpFaceModel, evaluate, main 函数与 train_split.py 基本一致)
```

**代码复用率**: ~95%

**主要改动**:
1. 导入 `prepare_turtle_dataloaders` 而非 `prepare_dataloaders`
2. 配置文件默认路径改为 `configs/config_turtle.yaml`
3. 其余训练循环、模型定义、评估逻辑完全一致

---

## 4. 训练策略

### 4.1 分阶段训练

| 阶段 | Epoch | 训练内容 | 学习率 |
|------|-------|---------|--------|
| 阶段1 | 1-10 | backbone.layer4 + bottleneck + arcface | backbone: 1e-4, 其他: 1e-3 |
| 阶段2 | 11-30 | 全部参数微调 | 余弦衰减至1e-6 |

### 4.2 超参数对比

| 参数 | 大猩猩数据集 | 龟类数据集 | 原因 |
|------|------------|-----------|------|
| num_identities | 24 | 38 | 个体数量不同 |
| batch_size | 4-8 | 8 | 可以增大 |
| accumulation_steps | 8 | 4 | 保持等效batch_size=32 |
| min_samples_per_identity | 20 | 5 | 龟类个体图片数较少 |
| epochs | 30 | 30 | 保持一致，可根据结果调整 |

---

## 5. 使用方法

### 5.1 训练

```bash
python train_turtle.py --config configs/config_turtle.yaml
```

### 5.2 评估（复用现有 evaluate.py）

```bash
python evaluate.py --checkpoint checkpoints_turtle/best_model.pth --config configs/config_turtle.yaml
```

### 5.3 推理（复用现有 inference.py）

```bash
python inference.py --checkpoint checkpoints_turtle/best_model.pth --image turtle.jpg
```

---

## 6. 扩展性

### 6.1 切换数据集子集

只需修改 `config_turtle.yaml` 中的 `dataset_name`:

```yaml
# 切换到 dataset_D_ge3years_count (108个个体)
dataset_name: "dataset_D_ge3years_count"
```

### 6.2 未来优化方向

1. **利用bbox信息**: 从COCO标注中提取面部bounding box进行裁剪
2. **时间序列评估**: 利用`date`字段分析跨时间泛化能力
3. **数据增强调整**: 针对龟类特点调整增强策略

---

## 7. 注意事项

1. **路径问题**: Windows路径使用反斜杠，YAML配置中需要转义或使用正斜杠
2. **磁盘空间**: 检查点目录独立，不与其他实验冲突
3. **显存需求**: batch_size=8, accumulation_steps=4，显存需求与原配置相当
4. **COCO JSON解析**: 确保JSON文件格式一致，异常处理待完善

---

## 8. 测试计划

1. **数据集加载测试**: 验证COCO JSON解析正确性
2. **数据增强测试**: 确认transform应用于龟类图片
3. **训练流程测试**: 运行1-2个epoch验证完整性
4. **评估指标测试**: 验证Accuracy、TAR@FAR计算正确

---

**文档版本**: v1.0  
**最后更新**: 2026-04-10
