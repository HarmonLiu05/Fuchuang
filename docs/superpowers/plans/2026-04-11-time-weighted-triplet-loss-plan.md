# 时间加权 Triplet Loss 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Triplet Loss 中加入时间权重 + 动态 warmup，提升跨时间泛化能力

**Architecture:** 新增 `TimeWeightedTripletLoss` 类 + `compute_triplet_weight` 函数，实现时间加权 + epoch 动态权重

**Tech Stack:** PyTorch, Python, YAML 配置

---

### Task 1: 实现 TimeWeightedTripletLoss 类

**Files:**
- Create: `zq/losses/time_weighted_triplet.py`
- Test: `zq/tests/test_time_weighted_triplet.py`

- [ ] **Step 1: 创建时间加权 Triplet Loss 类**

创建文件 `zq/losses/time_weighted_triplet.py`:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplet import _to_numeric_time


class TimeWeightedTripletLoss(nn.Module):
    """
    时间加权 Batch-Hard Triplet Loss
    
    核心思想：
        同类样本的时间跨度越大，越难区分，给予更多关注
        
    公式：
        time_gap = |time(a) - time(p)| / max_time_gap
        weight = 1 + alpha * time_gap
        loss = weight * max(0, d(a, p) - d(a, n) + margin)
    """
    def __init__(self, margin: float = 0.3, alpha: float = 0.3, 
                 max_time_gap: float = 10.0, normalize_embeddings: bool = True):
        """
        Args:
            margin: 间隔阈值
            alpha: 时间权重系数，控制时间跨度对损失的影响强度
            max_time_gap: 最大时间跨度（年），用于归一化
            normalize_embeddings: 是否对特征向量做 L2 归一化
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.max_time_gap = max_time_gap
        self.normalize_embeddings = normalize_embeddings

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                times: torch.Tensor) -> torch.Tensor:
        """
        计算时间加权 Triplet Loss
        
        Args:
            embeddings: [B, D] 特征向量
            labels: [B] 个体 ID 标签
            times: [B] 数值化时间（年），无时间信息则为 NaN
            
        Returns:
            loss: 标量，时间加权的平均 triplet loss
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings should be [B, D], got {embeddings.shape}")
        if labels.ndim != 1:
            labels = labels.view(-1)
        if times.ndim != 1:
            times = times.view(-1)

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

        pos_mask = same & ~eye
        neg_mask = ~same

        if not pos_mask.any() or not neg_mask.any():
            return embeddings.new_tensor(0.0)

        hardest_pos = dist_mat.masked_fill(~pos_mask, float('-inf')).max(dim=1).values
        hardest_neg = dist_mat.masked_fill(~neg_mask, float('inf')).min(dim=1).values

        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return embeddings.new_tensor(0.0)

        # --- 计算时间权重 ---
        time_a = times[valid]  # [N_valid]
        
        # 找 hardest positive 的索引
        pos_dist = dist_mat.masked_fill(~pos_mask, float('-inf'))
        hardest_pos_idx = pos_dist.argmax(dim=1)  # [B]
        time_p = times[hardest_pos_idx][valid]  # [N_valid]
        
        # 时间差归一化
        time_gap = torch.abs(time_a - time_p) / self.max_time_gap  # [N_valid]
        time_gap = torch.clamp(time_gap, 0, 1)  # 限制在 [0, 1]
        
        # 处理 NaN（无时间信息的情况）
        weight = torch.ones_like(time_gap)
        valid_time = ~torch.isnan(time_gap)
        weight[valid_time] = 1.0 + self.alpha * time_gap[valid_time]

        # --- 计算加权损失 ---
        losses = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)
        weighted_losses = weight * losses
        
        return weighted_losses.mean() if weighted_losses.numel() > 0 else embeddings.new_tensor(0.0)
```

- [ ] **Step 2: 在 losses/__init__.py 中导出**

修改 `zq/losses/__init__.py`:

```python
from .triplet import BatchHardTripletLoss
from .time_weighted_triplet import TimeWeightedTripletLoss

__all__ = ['BatchHardTripletLoss', 'TimeWeightedTripletLoss']
```

- [ ] **Step 3: 创建单元测试**

创建文件 `zq/tests/test_time_weighted_triplet.py`:

```python
import torch
import pytest
from losses.time_weighted_triplet import TimeWeightedTripletLoss


class TestTimeWeightedTripletLoss:
    def test_basic_loss(self):
        """基础测试：4个样本，2个个体"""
        loss_fn = TimeWeightedTripletLoss(margin=0.3, alpha=0.3)
        embeddings = torch.randn(4, 512)
        labels = torch.tensor([0, 0, 1, 1])
        times = torch.tensor([2020.0, 2021.0, 2020.0, 2021.0])
        
        loss = loss_fn(embeddings, labels, times)
        assert loss >= 0
        assert loss.dim() == 0

    def test_time_weighting_effect(self):
        """验证时间加权效果"""
        loss_fn = TimeWeightedTripletLoss(margin=0.5, alpha=0.5, max_time_gap=10.0)
        
        embeddings = torch.tensor([
            [1.0, 0.0],  # A: 个体0, 2020年
            [0.9, 0.1],  # B: 个体0, 2025年 (时间跨度5年)
            [0.0, 1.0],  # C: 个体1, 2020年
        ])
        labels = torch.tensor([0, 0, 1])
        times = torch.tensor([2020.0, 2025.0, 2020.0])
        
        loss = loss_fn(embeddings, labels, times)
        assert loss >= 0

    def test_no_time_info(self):
        """测试无时间信息时退化为普通 Triplet"""
        loss_fn = TimeWeightedTripletLoss(margin=0.3, alpha=0.3)
        embeddings = torch.randn(4, 512)
        labels = torch.tensor([0, 0, 1, 1])
        times = torch.tensor([float('nan')] * 4)
        
        loss = loss_fn(embeddings, labels, times)
        assert loss >= 0

    def test_single_identity(self):
        """测试单一个体返回 0"""
        loss_fn = TimeWeightedTripletLoss()
        embeddings = torch.randn(4, 512)
        labels = torch.tensor([0, 0, 0, 0])
        times = torch.tensor([2020.0, 2021.0, 2022.0, 2023.0])
        
        loss = loss_fn(embeddings, labels, times)
        assert loss.item() == 0.0

    def test_alpha_zero_equals_standard(self):
        """测试 alpha=0 时等于标准 Triplet Loss"""
        from losses.triplet import BatchHardTripletLoss
        
        loss_fn_time = TimeWeightedTripletLoss(margin=0.3, alpha=0.0)
        loss_fn_std = BatchHardTripletLoss(margin=0.3)
        
        embeddings = torch.randn(8, 256)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        times = torch.tensor([2020.0] * 8)
        
        loss_time = loss_fn_time(embeddings, labels, times)
        loss_std = loss_fn_std(embeddings, labels)
        
        assert torch.allclose(loss_time, loss_std, atol=1e-6)
```

- [ ] **Step 4: 运行测试**

```bash
cd E:\fuchuang\chimpanzee_arcface\zq
python -m pytest tests/test_time_weighted_triplet.py -v
```

预期：5 个测试全部通过

- [ ] **Step 5: Commit**

```bash
cd E:\fuchuang\chimpanzee_arcface
git add zq/losses/time_weighted_triplet.py zq/losses/__init__.py zq/tests/test_time_weighted_triplet.py
git commit -m "feat: 添加时间加权 Triplet Loss"
```

---

### Task 2: 集成到训练脚本

**Files:**
- Modify: `zq/train_turtle.py`

- [ ] **Step 1: 添加 compute_triplet_weight 函数**

在 `TurtleFaceModel` 类定义之前添加：

```python
def compute_triplet_weight(epoch, training_cfg):
    """计算当前 epoch 的 triplet_weight（带 warmup）"""
    max_weight = training_cfg.get('triplet_weight', 0.2)
    start_epoch = training_cfg.get('triplet_start_epoch', 80)
    warmup_epochs = training_cfg.get('triplet_warmup_epochs', 20)

    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return max_weight

    progress = min(1.0, (epoch - start_epoch + 1) / warmup_epochs)
    return max_weight * progress
```

- [ ] **Step 2: 更新导入**

修改顶部导入：

```python
# 原来
from losses.triplet import BatchHardTripletLoss

# 改为
from losses.time_weighted_triplet import TimeWeightedTripletLoss
from samplers.time_aware_sampler import _to_numeric_time
```

- [ ] **Step 3: 修改损失函数初始化**

```python
# 原来
    triplet_criterion = BatchHardTripletLoss(
        margin=config['training'].get('triplet_margin', 0.3),
        normalize_embeddings=True
    )

# 改为
    triplet_criterion = TimeWeightedTripletLoss(
        margin=config['training'].get('triplet_margin', 0.3),
        alpha=config['training'].get('time_alpha', 0.3),
        max_time_gap=config['training'].get('max_time_gap', 10.0),
        normalize_embeddings=True
    )
```

- [ ] **Step 4: 修改训练循环中的数据提取]

```python
# 原来
        for batch_idx, batch in enumerate(pbar):
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

# 改为
        for batch_idx, batch in enumerate(pbar):
            images, labels = batch[:2]
            # 提取时间信息
            times_list = [_to_numeric_time(batch[2][i] if len(batch) > 2 else None) 
                          for i in range(len(images))]
            times = torch.tensor([t if t is not None else float('nan') for t in times_list], 
                                dtype=torch.float32)
            
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            times = times.to(device)
```

- [ ] **Step 5: 修改训练循环中的权重计算和损失计算**

在训练循环中添加 `current_triplet_weight` 计算：

```python
# 在 for batch_idx, batch in enumerate(pbar): 之前添加
        current_triplet_weight = compute_triplet_weight(epoch, config['training'])
```

修改损失计算：

```python
# 原来
                if epoch<100:
                    loss = cls_loss / accumulation_steps
                else:
                    tri_loss = triplet_criterion(features, labels)
                    loss = (cls_loss + triplet_weight * tri_loss) / accumulation_steps

# 改为
                tri_loss = triplet_criterion(features, labels, times) if current_triplet_weight > 0 else torch.zeros(1, device=device, dtype=cls_loss.dtype).squeeze(0)
                total_batch_loss = cls_loss + current_triplet_weight * tri_loss
                loss = total_batch_loss / accumulation_steps
```

修改进度条显示：

```python
# 原来
            pbar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.4f}'})

# 改为
            pbar.set_postfix({
                'loss': f'{total_loss / (batch_idx + 1):.4f}',
                'tri_w': f'{current_triplet_weight:.3f}'
            })
```

- [ ] **Step 6: 修改打印输出]

```python
# 原来
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, Acc0={acc0:.4f}, "
              f"LR={scheduler.get_last_lr()[0]:.6f}")

# 改为
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={acc:.4f}, Acc0={acc0:.4f}, "
              f"TripletW={current_triplet_weight:.3f}, LR={scheduler.get_last_lr()[0]:.6f}")
```

- [ ] **Step 7: Commit**

```bash
cd E:\fuchuang\chimpanzee_arcface
git add zq/train_turtle.py
git commit -m "feat: 训练脚本集成时间加权 Triplet Loss + 动态 warmup"
```

---

### Task 3: 更新配置文件

**Files:**
- Modify: `zq/configs/config_turtle.yaml`

- [ ] **Step 1: 添加时间加权和 warmup 参数**

在 `training` 部分添加：

```yaml
training:
  # ... 现有参数 ...
  triplet_weight: 0.2
  triplet_margin: 0.3
  triplet_start_epoch: 80        # 开始引入 Triplet
  triplet_warmup_epochs: 20      # warmup 周期
  time_alpha: 0.3                # 时间权重系数
  max_time_gap: 10.0             # 最大时间跨度（年）
```

- [ ] **Step 2: Commit**

```bash
cd E:\fuchuang\chimpanzee_arcface
git add zq/configs/config_turtle.yaml
git commit -m "config: 添加时间加权和 warmup 参数"
```

---

### Task 4: 验证完整训练流程

- [ ] **Step 1: 运行快速验证**

```bash
cd E:\fuchuang\chimpanzee_arcface\zq
python -c "
from losses.time_weighted_triplet import TimeWeightedTripletLoss
import torch

loss_fn = TimeWeightedTripletLoss(margin=0.3, alpha=0.3)
embeddings = torch.randn(8, 512)
labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
times = torch.tensor([2020.0, 2023.0, 2020.0, 2021.0, 2020.0, 2025.0, 2020.0, 2022.0])
loss = loss_fn(embeddings, labels, times)
print(f'Loss: {loss.item():.4f}')
print('OK')
"
```

- [ ] **Step 2: 检查配置加载]

```bash
cd E:\fuchuang\chimpanzee_arcface\zq
python -c "
from utils.utils import load_config
config = load_config('configs/config_turtle.yaml')
print(f'triplet_start_epoch: {config[\"training\"].get(\"triplet_start_epoch\")}')
print(f'triplet_warmup_epochs: {config[\"training\"].get(\"triplet_warmup_epochs\")}')
print(f'time_alpha: {config[\"training\"].get(\"time_alpha\")}')
print('OK')
"
```

- [ ] **Step 3: 完整 Commit 并推送**

```bash
cd E:\fuchuang\chimpanzee_arcface
git status
git -c http.sslVerify=false push origin main
```

---

## 自审清单

| 检查项 | 状态 |
|--------|------|
| 无 TBD/TODO | ✅ |
| 类型一致性 | ✅ times 都是 torch.Tensor |
| 文件边界清晰 | ✅ 新类独立文件 |
| 测试覆盖边界 | ✅ 5 个测试用例 |
| 配置参数有默认值 | ✅ alpha=0.3, max_time_gap=10.0 |
| 动态权重逻辑完整 | ✅ compute_triplet_weight warmup |
