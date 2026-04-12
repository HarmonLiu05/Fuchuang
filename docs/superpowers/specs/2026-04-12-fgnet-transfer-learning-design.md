# FGNET 人脸跨年龄数据集迁移训练设计文档

**日期**: 2026-04-12  
**目标**: 在 FGNET 数据集上验证时间 APN 三元组损失的有效性  
**方案**: 最小改动方案 (方案A)

---

## 1. 设计目标

### 1.1 核心目标
在 FGNET 人脸数据集上验证**时间跨度感知 APN Triplet Loss** 是否比纯距离三元组损失更有效,与龟类数据集实验结果形成对比。

### 1.2 实验对比
| 实验 | 损失配置 | 说明 |
|------|---------|------|
| **Baseline** | 纯交叉熵 (CE) | 对照基准 |
| **距离三元组** | CE + Batch-Hard Triplet | 通用度量学习 |
| **时间 APN** | CE + Temporal-APN Triplet | 跨年龄时间鲁棒性 |

### 1.3 成功标准
- 时间 APN 的 Acc_direct 高于距离三元组
- 验证时间信息在人脸场景的增益是否与龟类场景一致

---

## 2. 数据集说明

### 2.1 FGNET 数据集
- **人数**: 82 个个体
- **照片数**: 1,002 张
- **年龄范围**: 0-69 岁
- **每人照片**: 6-18 张 (平均 12.22 张)
- **特征**: 纵向老化数据集,同一个人有多个年龄段照片

### 2.2 年龄标注来源
使用 `age_groundtruth.csv` 文件:
```csv
SampleID;Age
001A02.JPG;2
001A05.JPG;5
...
```

### 2.3 数据划分策略
**按时间划分** (与龟类数据集一致):
- 每个人的照片按年龄从小到大排序
- **前 70% 旧照片 → 训练集**
- **后 30% 新照片 → 测试集**

**意义**: 模拟真实跨年龄识别场景 - 用历史数据训练,识别未来照片

**示例** (个体 001, 15 张照片):
```
年龄序列: 2, 5, 8, 10, 14, 16, 18, 19, 22, 28, 29, 33, 40, 43, 43
训练集: 2, 5, 8, 10, 14, 16, 18, 19, 22, 28  (10张)
测试集: 29, 33, 40, 43, 43  (5张)
```

---

## 3. 架构设计

### 3.1 整体架构
```
FGNET 数据集
    ↓
[新增] prepare_fgnet.py  ← 数据集适配层
    ↓
[修改] train_turtle.py  ← 数据集选择器 (5行代码)
    ↓
[复用] 现有代码
    ├── models/ (backbone, bottleneck, arcface)
    ├── losses/ (triplet, temporal_apn_triplet)
    ├── samplers/ (time_aware_sampler)
    └── utils/ (metrics, utils)
```

### 3.2 改动范围

#### ✅ 需要新增的文件
1. **`data/prepare_fgnet.py`** - FGNET 数据集加载器
2. **`configs/config_fgnet_baseline.yaml`** - Baseline CE 配置
3. **`configs/config_fgnet_dist_triplet.yaml`** - 距离三元组配置
4. **`configs/config_fgnet_temporal_apn.yaml`** - 时间 APN 配置

#### 🔧 需要修改的文件
1. **`train_turtle.py`** - 添加数据集类型选择器 (约 5 行代码)

#### ❌ 无需改动的文件
- `models/` - 模型架构完全通用
- `losses/` - 三元组损失逻辑与数据集无关
- `samplers/` - 时间感知采样器通用
- `utils/` - 工具函数通用

---

## 4. 详细设计

### 4.1 FGNET 数据集适配层

#### 文件: `data/prepare_fgnet.py`

**核心功能**:
1. 解析 FGNET 目录结构
2. 读取 `age_groundtruth.csv` 获取年龄标注
3. 按 70/30 时间划分训练集/测试集
4. 生成与 `TurtleDataset` 兼容的数据结构

**实现方案**:

```python
class FGNETDataset(Dataset):
    """FGNET 人脸数据集加载器"""
    
    def __init__(self, config, split='train', transform=None, return_time=False):
        # 1. 扫描 images/ 目录获取所有 JPG 文件
        # 2. 读取 age_groundtruth.csv 构建 {filename: age} 映射
        # 3. 按个体ID分组,每个个体的照片按年龄排序
        # 4. 70/30 划分: 前70% → train, 后30% → test
        # 5. 构建 image_list: [{'path': ..., 'label': ..., 'age': ...}]
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # 返回: (image, label, age) 如果 return_time=True
        # 否则: (image, label)
```

**关键设计点**:
- `age` 字段作为 `time` 信息传入 Triplet Loss
- 身份标签从 0 开始编号 (0-81, 共 82 个个体)
- 与 `TurtleDataset` 保持相同的 `__getitem__` 返回格式

#### 数据准备函数:

```python
def prepare_fgnet_dataloaders(config, return_time=False, use_time_aware_sampler=False):
    """
    准备 FGNET 数据集的 DataLoader
    
    Returns:
        train_loader, test_loader, num_identities (82), train_dataset
    """
    # 1. 创建训练集和测试集
    # 2. 使用 FGNETBatchSampler (复用 TimeAwareBatchSampler)
    # 3. 返回 DataLoader 实例
```

### 4.2 训练脚本适配

#### 文件: `train_turtle.py`

**修改位置**: `main()` 函数中的数据准备部分

**修改前**:
```python
train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(
    config, return_time=use_time_info, use_time_aware_sampler=use_sampler
)
```

**修改后**:
```python
# 数据集选择器
dataset_type = config['data'].get('dataset_type', 'turtle')

if dataset_type == 'fgnet':
    from data.prepare_fgnet import prepare_fgnet_dataloaders
    train_loader, test_loader, num_identities, train_dataset = prepare_fgnet_dataloaders(
        config, return_time=use_time_info, use_time_aware_sampler=use_sampler
    )
else:
    from data.prepare_turtle_data import prepare_turtle_dataloaders
    train_loader, test_loader, num_identities, train_dataset = prepare_turtle_dataloaders(
        config, return_time=use_time_info, use_time_aware_sampler=use_sampler
    )
```

**改动量**: 仅 5 行代码

### 4.3 配置文件设计

#### 基础配置差异

| 配置项 | 龟类数据集 | FGNET 数据集 | 说明 |
|--------|-----------|-------------|------|
| `batch_size` | 64 | 32 | FGNET 数据集小,降低 batch |
| `epochs` | 200 | 150 | 数据少,收敛快 |
| `triplet_start_epoch` | 80 | 50 | 更早引入 Triplet |
| `triplet_warmup_epochs` | 20 | 15 | 缩短 warmup |
| `num_workers` | 4 | 0 | Windows 本地训练 |
| `min_samples_per_identity` | 5 | 3 | FGNET 最少 6 张,设 3 保证全部使用 |
| `image_size` | 224 | 224 | 保持一致 |

#### 三个实验配置

**1. Baseline CE**:
```yaml
training:
  batch_size: 32
  epochs: 150
  use_temporal_apn_triplet: false
  triplet_start_epoch: 999  # 不启用
```

**2. 距离三元组**:
```yaml
training:
  batch_size: 32
  epochs: 150
  triplet_start_epoch: 50
  triplet_warmup_epochs: 15
  use_temporal_apn_triplet: false
```

**3. 时间 APN**:
```yaml
training:
  batch_size: 32
  epochs: 150
  triplet_start_epoch: 50
  triplet_warmup_epochs: 15
  use_temporal_apn_triplet: true
```

---

## 5. 数据流设计

### 5.1 训练数据流

```
FGNET/images/*.JPG  ──┐
                      ↓
age_groundtruth.csv → prepare_fgnet.py → FGNETDataset
                                              ↓
                                    (image, label, age)
                                              ↓
                              ┌───────────────┴───────────────┐
                              ↓                               ↓
                    CrossEntropyLoss              TemporalAPNTripletLoss
                    (分类损失)                     (度量学习损失)
                              ↓                               ↓
                              └───────────────┬───────────────┘
                                              ↓
                                      Total Loss = CE + λ × Triplet
                                              ↓
                                      Backward + Optimizer
```

### 5.2 时间 APN 在 FGNET 中的工作示例

**Batch 内样本** (假设 batch_size=32, 每个个体 4 张):

```
个体 001 (4张): 年龄 2, 5, 14, 18
个体 002 (4张): 年龄 3, 7, 15, 20
个体 003 (4张): 年龄 18, 25, 38, 47
...

对于 Anchor: 个体001_年龄5
  → Positive 选择: 个体001_年龄18 (时间跨度最大: |5-18|=13年)
  → Negative 选择: 个体003_年龄18 (时间跨度最小: |5-18|=13年,但不同个体)
```

**关键点**: 
- 年龄差距越大,Positive 越"难",学到的特征越时间鲁棒
- Negative 选年龄相近的不同个体,让模型学会区分相似年龄的不同人

---

## 6. 实验执行计划

### 6.1 实验顺序

```bash
# 实验 1: Baseline CE
python train_turtle.py --config configs/config_fgnet_baseline.yaml

# 实验 2: 距离三元组
python train_turtle.py --config configs/config_fgnet_dist_triplet.yaml

# 实验 3: 时间 APN
python train_turtle.py --config configs/config_fgnet_temporal_apn.yaml
```

### 6.2 预计训练时间

| 实验 | Epoch | 每轮时间 | 总时间 (GTX 1080Ti) |
|------|-------|---------|-------------------|
| Baseline | 150 | ~30秒 | ~75分钟 |
| 距离三元组 | 150 | ~35秒 | ~88分钟 |
| 时间 APN | 150 | ~40秒 | ~100分钟 |

### 6.3 评估指标

每个实验完成后运行:
```bash
python evaluate.py --checkpoint checkpoints_fgnet/best_model.pth \
    --config configs/config_fgnet_temporal_apn.yaml
```

**输出指标**:
- `acc_direct`: 直接分类准确率
- `accuracy`: 检索 Accuracy (基于特征距离)
- `TAR@FAR=0.1%`: 真接受率 @ 假接受率 0.1%
- `Rank-1`: Top-1 检索准确率

---

## 7. 目录结构设计

```
E:\fuchuang\FGNET\
├── FGNET\                          ← 原始 FGNET 数据集
│   ├── images\                     (1002 张 JPG)
│   ├── points\                     (68 特征点)
│   └── age_annotations\
│       └── kara2015_ageannotations\
│           └── age_groundtruth.csv
│
└── Fuchuang\                       ← 训练代码仓库
    ├── data/
    │   ├── prepare_turtle_data.py      # 现有
    │   ├── prepare_fgnet.py            # [新增] FGNET 适配层
    │   └── ...
    ├── configs/
    │   ├── config_fgnet_baseline.yaml       # [新增]
    │   ├── config_fgnet_dist_triplet.yaml   # [新增]
    │   └── config_fgnet_temporal_apn.yaml   # [新增]
    ├── checkpoints_fgnet/              # [新增] FGNET 模型保存目录
    │   ├── baseline/
    │   ├── dist_triplet/
    │   └── temporal_apn/
    ├── results/
    │   └── fgnet_experiments/          # [新增] 实验结果记录
    │       └── comparison.md
    └── train_turtle.py                 # [修改] +5行数据集选择器
```

---

## 8. 风险与应对

### 8.1 数据量过小
- **风险**: FGNET 仅 1002 张图,可能过拟合
- **应对**: 
  - 强数据增强 (翻转、仿射、颜色抖动、随机擦除)
  - 早停策略 (监控验证集性能)
  - 降低模型复杂度 (可切换到 ResNet50)

### 8.2 身份标签不一致
- **风险**: 训练集和测试集的个体标签映射不一致
- **应对**: 
  - 测试集共享训练集的 `identity_map`
  - 划分时确保每个个体在 train/test 都有样本

### 8.3 时间信息格式不兼容
- **风险**: `age` (浮点数) 与龟类的 `date` (字符串) 格式不同
- **应对**: 
  - `age` 直接作为数值时间戳
  - 修改 `_to_numeric_time` 函数处理浮点数输入

---

## 9. 预期成果

### 9.1 实验结果对比表

| 实验 | Epoch | Acc_direct | Rank-1 | TAR@FAR=0.1% | 说明 |
|------|-------|-----------|--------|--------------|------|
| Baseline CE | 150 | TBD | TBD | TBD | 对照基准 |
| 距离三元组 | 150 | TBD | TBD | TBD | 预期 +5-10% |
| 时间 APN | 150 | TBD | TBD | TBD | 预期再 +1-2% |

### 9.2 与龟类实验对比

| 维度 | 龟类 (107个体) | FGNET (82个体) | 分析 |
|------|---------------|---------------|------|
| 距离三元组增益 | +4.92% | TBD | 对比跨物种效果 |
| 时间 APN 增益 | +0.52% (vs 距离) | TBD | 验证人脸场景优势 |

### 9.3 论文贡献
- 首次验证时间 APN 在人脸场景的有效性
- 对比龟类 vs 人脸的跨物种迁移学习
- 为时间鲁棒特征学习提供新证据

---

## 10. 后续扩展

### 10.1 可能的后续实验
- [ ] 在 CACD 数据集 (2000人, 16万张) 上大规模验证
- [ ] 双 Triplet 联合损失 (距离 + 时间)
- [ ] 跨数据集泛化 (龟类→人脸→其他物种)

### 10.2 代码重构机会
如果后续需要支持更多数据集,可以考虑:
- 重构为统一数据集基类 (方案B)
- 添加配置文件验证
- 增加实验管理工具 (如 MLflow)

---

**设计文档完成,待用户审查后进入实现阶段**
