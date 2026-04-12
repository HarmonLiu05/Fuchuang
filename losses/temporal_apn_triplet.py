import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAPNTripletLoss(nn.Module):
    """
    时间跨度感知 APN Triplet Loss

    核心思想：
        - Anchor (A): 当前样本
        - Positive (P): 同类中时间跨度最远的个体（最难区分的同类，时间差异最大）
        - Negative (N): 异类中时间跨度最近的个体（最容易混淆的异类，时间差异最小）

    策略：
        - 对每个 Anchor，在 batch 内搜索：
          · Positive: 同类样本中，|time(a) - time(p)| 最大的样本
          · Negative: 异类样本中，|time(a) - time(n)| 最小的样本
        - 若多个样本时间跨度相同，选择 embedding 距离最难的作为 tie-breaker

    公式：
        loss = max(0, d(a, p) - d(a, n) + margin)

    优势：
        - 强制模型学习时间鲁棒性：即使时间跨度很大，同类样本也应靠近
        - 防止模型被时间相近的异类样本混淆
        - 更贴近真实场景：同一只龟在不同年份的外观变化可能很大
    """
    def __init__(self, margin: float = 0.3, normalize_embeddings: bool = True):
        """
        Args:
            margin: 间隔阈值，强制正负样本距离差至少为 margin
            normalize_embeddings: 是否对特征向量做 L2 归一化
        """
        super().__init__()
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                times: torch.Tensor) -> torch.Tensor:
        """
        计算时间跨度感知 APN Triplet Loss

        Args:
            embeddings: [B, D] 特征向量，B 为 batch size，D 为特征维度
            labels: [B] 每个样本的个体 ID 标签
            times: [B] 数值化时间（年），无时间信息则为 NaN

        Returns:
            loss: 标量，平均 triplet loss
        """
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings should be [B, D], got {embeddings.shape}")
        if labels.ndim != 1:
            labels = labels.view(-1)
        if times.ndim != 1:
            times = times.view(-1)

        B = embeddings.size(0)

        # --- Step 1: 特征归一化 ---
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # --- Step 2: 计算 pairwise 距离矩阵 ---
        dist_mat = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        # --- Step 3: 构建正负样本掩码 ---
        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        eye = torch.eye(B, dtype=torch.bool, device=labels.device)
        pos_mask = same & ~eye  # 正样本：同类且不是自己
        neg_mask = ~same  # 负样本：不同类

        if not pos_mask.any() or not neg_mask.any():
            return embeddings.new_tensor(0.0)

        # --- Step 4: 计算时间差距矩阵 ---
        # time_diff[i][j] = |time[i] - time[j]|
        time_a = times.unsqueeze(1)  # [B, 1]
        time_b = times.unsqueeze(0)  # [1, B]
        time_diff = torch.abs(time_a - time_b)  # [B, B]

        # 处理 NaN 时间：设为 -1 作为标记
        valid_time_mask = ~torch.isnan(time_diff)
        time_diff = time_diff.masked_fill(~valid_time_mask, -1.0)

        # --- Step 5: 对每个 Anchor，选择 Temporal Positive / Negative ---
        # Positive: 同类中时间差距最大的
        # Negative: 异类中时间差距最小的

        selected_pos_idx = torch.full((B,), -1, dtype=torch.long, device=labels.device)
        selected_neg_idx = torch.full((B,), -1, dtype=torch.long, device=labels.device)

        for i in range(B):
            # 找 positive candidates（同类且非自己）
            pos_candidates = pos_mask[i].clone()
            # 只考虑有时间信息的样本
            pos_candidates = pos_candidates & valid_time_mask[i]

            if pos_candidates.any():
                # 选时间差距最大的
                pos_time_diff = time_diff[i].masked_fill(~pos_candidates, -1.0)
                max_time_diff = pos_time_diff.max()

                # Tie-breaker: 如果多个样本时间差距相同，选 embedding 距离最远的（hardest）
                candidates_with_max_time = (pos_time_diff == max_time_diff)
                pos_dist = dist_mat[i].masked_fill(~candidates_with_max_time, -1.0)
                selected_pos_idx[i] = pos_dist.argmax()

            # 找 negative candidates（异类）
            neg_candidates = neg_mask[i].clone()
            # 只考虑有时间信息的样本
            neg_candidates = neg_candidates & valid_time_mask[i]

            if neg_candidates.any():
                # 选时间差距最小的（但必须 > 0）
                neg_time_diff = time_diff[i].masked_fill(~neg_candidates, float('inf'))
                min_time_diff = neg_time_diff.min()

                # Tie-breaker: 如果多个样本时间差距相同，选 embedding 距离最近的（hardest）
                candidates_with_min_time = (neg_time_diff == min_time_diff)
                neg_dist = dist_mat[i].masked_fill(~candidates_with_min_time, float('inf'))
                selected_neg_idx[i] = neg_dist.argmin()

        # --- Step 6: 筛选有效样本 ---
        valid = (selected_pos_idx >= 0) & (selected_neg_idx >= 0)
        if not valid.any():
            return embeddings.new_tensor(0.0)

        # --- Step 7: 计算 Triplet Loss ---
        anchor_indices = torch.nonzero(valid, as_tuple=True)[0]
        pos_indices = selected_pos_idx[anchor_indices]
        neg_indices = selected_neg_idx[anchor_indices]

        # d(a, p) 和 d(a, n)
        dist_ap = dist_mat[anchor_indices, pos_indices]  # [N_valid]
        dist_an = dist_mat[anchor_indices, neg_indices]  # [N_valid]

        losses = F.relu(dist_ap - dist_an + self.margin)

        return losses.mean() if losses.numel() > 0 else embeddings.new_tensor(0.0)
