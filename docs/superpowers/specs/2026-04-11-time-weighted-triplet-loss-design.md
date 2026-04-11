# 时间加权 Triplet Loss 设计文档

## 背景

当前训练脚本在 100 epoch 后加入 Triplet Loss，但所有样本对的权重相同，没有考虑时间跨度。
跨时间样本（如同一只龟 2020 年 vs 2023 年的照片）更难区分，应该给予更多关注。

## 目标

在 Triplet Loss 中加入时间权重，让时间跨度大的样本对损失贡献更大，提升跨时间泛化能力。

## 核心设计

### 公式

```
传统 Triplet Loss:
    loss = max(0, d(a, p) - d(a, n) + margin)

时间加权 Triplet Loss:
    time_gap = |time(a) - time(p)| / max_time_gap  # 归一化到 [0, 1]
    weight = 1 + α × time_gap                       # α 控制时间权重强度
    loss = weight × max(0, d(a, p) - d(a, n) + margin)
```

### 参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| α (time_alpha) | 0.3 | 时间权重系数，参考跨年龄人脸论文中的辅助损失权重 |
| max_time_gap | 10.0 年 | 最大时间跨度归一化因子 |
| margin | 0.3 | 保持不变 |
| triplet_weight | 0.2 | 最大值，动态 warmup |
| triplet_start_epoch | 80 | 开始引入 Triplet 的 epoch |
| triplet_warmup_epochs | 20 | warmup 周期（80→100 epoch 线性增长） |

### 动态权重策略

```python
# triplet_weight 随 epoch 变化
epoch < 80:   triplet_weight = 0.0     # 纯分类
epoch 80-99:  triplet_weight = 0.0 → 0.2  # 线性 warmup
epoch >= 100: triplet_weight = 0.2     # 最大值

# time_alpha 固定 0.3，不动态变化
# 最终 Triplet 影响: triplet_weight × (1 + time_alpha × time_gap)
```

### 为什么这样设计？

- 前 80 epoch：模型学习基本分类，特征空间粗糙
- 80-100 epoch：特征逐渐稳定，逐渐引入 Triplet 细化特征
- 100+ epoch：Triplet 全开，时间加权发挥作用

### 权重范围（epoch >= 100 时）

```
α = 0.3 时:
  time_gap = 0    → weight = 1.0  (同年照片，无加权)
  time_gap = 0.5  → weight = 1.15 (相隔5年，加权15%)
  time_gap = 1.0  → weight = 1.3  (相隔10年，加权30%)

最终 Triplet 影响:
  同年: 0.2 × 1.0 = 0.20
  5年:  0.2 × 1.15 = 0.23
  10年: 0.2 × 1.3 = 0.26
```

### 为什么 α = 0.3？

参考论文 *The identity-level angular triplet loss for cross-age face recognition* (2021)：
- Triplet Loss 相对 CrossEntropy 的权重 λ = 0.1
- 你的项目中 `triplet_weight = 0.2`，已经比论文大
- 时间加权是辅助信号，α=0.3 时最大影响 ≈ 0.2 × 1.3 = 0.26 倍，不会主导训练

## 文件变更

| 文件 | 变更 |
|------|------|
| `zq/losses/triplet.py` | 添加 `TimeWeightedTripletLoss` 类 |
| `zq/train_turtle.py` | 导入新 Loss，从 batch 提取时间信息 |
| `zq/configs/config_turtle.yaml` | 添加 `time_alpha: 0.3` 配置 |

## 边界情况

| 情况 | 处理 |
|------|------|
| 没有时间信息 | weight = 1.0，退化为普通 Triplet |
| 单一个体 batch | loss = 0（已有处理） |
| 时间解析失败 | weight = 1.0 |

## 预期效果

- 跨时间泛化能力 +3-5%
- 训练速度影响 < 5%
- 不影响推理代码
