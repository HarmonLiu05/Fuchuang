# AgeDB 人脸数据集适配设计

**日期**: 2026-04-12
**分支**: feature/face-migration
**目标**: 将训练框架迁移到 AgeDB 数据集，验证时间 APN 在跨年龄人脸场景的有效性

## 1. 数据集概览

| 指标 | 值 |
|------|-----|
| 人数 | 567 |
| 图片数 | 16,488 |
| 每人图片数 ≥ 2 | 566 人 |
| 年龄跨度 ≥ 5 年 | 566 人 |
| 最大年龄跨度 | 90 年（Zsa Zsa Gabor, 6-96 岁） |

**文件名格式**: `ID_Name_Age_gender.jpg`
- `ID`: 数字编号
- `Name`: 人名（identity 标识）
- `Age`: 年龄（作为"时间"信息，等价于龟类的拍摄日期）
- `gender`: m/f

## 2. 架构设计

### 2.1 新增文件

```
Fuchuang/
├── data/
│   └── prepare_agedb.py          # AgeDB 数据加载器（新增）
├── configs/
│   ├── config_agedb_ce.yaml      # AgeDB 纯交叉熵（新增）
│   ├── config_agedb_dist.yaml    # AgeDB 距离三元组（新增）
│   └── config_agedb_apn.yaml     # AgeDB 时间APN（新增）
```

### 2.2 修改文件

| 文件 | 改动范围 | 行数变化 |
|------|---------|---------|
| `train_turtle.py` | 数据集选择器 | +10 行 |
| `losses/__init__.py` | 无需改动 | 0 |
| `arcface.py` | 无需改动 | 0 |

### 2.3 无需改动的部分

- ✅ Temporal APN Triplet Loss（时间跨度逻辑通用）
- ✅ Batch-Hard Triplet Loss（距离逻辑通用）
- ✅ ResNet backbone（ImageNet 预训练对人脸有效）
- ✅ ArcFace Head（本身就是为人脸识别设计的）
- ✅ 评估时 margin=0 的逻辑（已在上一个 commit 实现）

## 3. 数据适配层实现

### 3.1 `data/prepare_agedb.py`

**核心函数**: `prepare_agedb_dataloaders(config, return_time=False)`

**处理流程**:
1. 扫描 `/workspace/AgeDB/` 目录所有 `.jpg` 文件
2. 解析文件名提取: `name`, `age`, `gender`, `filepath`
3. 按人名分组，过滤图片数 < min_samples_per_identity 的个体
4. 每个个体按年龄排序，前 70% → 训练集，后 30% → 测试集
5. 创建 `AgeDBDataset` 类（复用 `TurtleDataset` 接口）
6. 返回 `train_loader, test_loader, num_identities, train_dataset`

**时间映射**:
- 年龄直接作为数值时间戳（如 `age=25` → `time=25.0`）
- 与龟类数据集的 `_to_numeric_time` 逻辑兼容
- 无效年龄设为 `NaN`，与龟类处理一致

**AgeDBDataset 类设计**:
```python
class AgeDBDataset(Dataset):
    def __init__(self, config, split='train', transform=None, return_time=False):
        # 1. 扫描图片，解析文件名
        # 2. 按人名分组，过滤 min_samples
        # 3. 按年龄排序，7:3 划分
        # 4. 构建 image_list: [{'path', 'label', 'age'}]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(path).convert('RGB')
        if transform: image = transform(image)
        if return_time: return image, label, age
        return image, label
```

### 3.2 `train_turtle.py` 改动

在数据加载处添加选择器：

```python
# 原代码:
train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(config)

# 改为:
dataset_type = config['data'].get('dataset_type', 'turtle')
if dataset_type == 'agedb':
    from data.prepare_agedb import prepare_agedb_dataloaders
    train_loader, test_loader, num_identities, train_dataset = prepare_agedb_dataloaders(
        config, return_time=True
    )
else:
    train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(
        config, return_time=True
    )
```

### 3.3 配置文件设计

**`config_agedb_ce.yaml`** (纯交叉熵 Baseline):
```yaml
data:
  root_dir: "/workspace/AgeDB"
  dataset_type: "agedb"
  min_samples_per_identity: 5
  image_size: 224
  num_workers: 8

training:
  epochs: 80
  freeze_until_epoch: 0
  base_lr: 0.001
  backbone_lr: 0.0001
  batch_size: 64
  use_temporal_apn_triplet: false
  checkpoint_dir: "/workspace/experiments-checkpoints/agedb_ce_only"
```

**`config_agedb_dist.yaml`** (距离三元组):
```yaml
# 同上，但:
training:
  epochs: 200
  triplet_start_epoch: 80
  triplet_weight: 0.2
  triplet_margin: 0.3
  triplet_warmup_epochs: 20
  use_temporal_apn_triplet: false
  checkpoint_dir: "/workspace/experiments-checkpoints/agedb_dist_triplet"
```

**`config_agedb_apn.yaml`** (时间 APN):
```yaml
# 同 dist，但:
  use_temporal_apn_triplet: true
  checkpoint_dir: "/workspace/experiments-checkpoints/agedb_apn"
```

## 4. 训练命令

```bash
# Baseline CE
python train_turtle.py --config configs/config_agedb_ce.yaml

# 距离三元组
python train_turtle.py --config configs/config_agedb_dist.yaml

# 时间 APN
python train_turtle.py --config configs/config_agedb_apn.yaml
```

## 5. 预期实验结果

| 实验 | 损失配置 | Epoch | 预期 Acc_direct |
|------|---------|-------|----------------|
| Baseline CE | CE only | 80 | ~60-70% |
| 距离三元组 | CE + Batch-Hard | 200 | ~75-85% |
| 时间 APN | CE + Temporal-APN | 200 | ~80-90% |

年龄跨度比龟类数据集更大（最大 90 年），APN 效果应该更显著。

## 6. 错误处理

- 文件名格式错误 → 跳过该文件并打印警告
- 无法加载图片 → 跳过并计数
- 无效年龄 → 设为 NaN，与龟类处理一致
- min_samples 过滤后无有效个体 → 抛出 RuntimeError

## 7. 验证方式

```bash
# 快速测试数据加载
python -c "
from data.prepare_agedb import prepare_agedb_dataloaders
import yaml
config = yaml.safe_load(open('configs/config_agedb_ce.yaml'))
train_loader, test_loader, num_ids, _ = prepare_agedb_dataloaders(config, return_time=True)
print(f'训练集: {len(train_loader.dataset)}, 测试集: {len(test_loader.dataset)}, 个体数: {num_ids}')
"
```
