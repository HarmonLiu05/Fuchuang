import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Loss

    核心思想：
        - 每个样本作为 Anchor，从 batch 中挑选：
          · Hardest Positive：同类个体中距离最远的样本（最难区分的同类）
          · Hardest Negative：不同类个体中距离最近的样本（最容易混淆的异类）
        - 目标：让同类样本靠近，异类样本远离，且保持 margin 间隔

    公式：
        loss = max(0, d(anchor, hardest_positive) - d(anchor, hardest_negative) + margin)

    优势：
        - 相比普通 Triplet Loss，Batch-Hard 策略让模型聚焦在最难的样本对上
        - 一个 batch 内即可构造所有可能的 triplet，无需显式采样
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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算 Batch-Hard Triplet Loss

        Args:
            embeddings: [B, D] 特征向量，B 为 batch size，D 为特征维度（如 512）
            labels: [B] 每个样本的个体 ID 标签

        Returns:
            loss: 标量，整个 batch 的平均 triplet loss
        """
        # --- 输入校验 ---
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings should be [B, D], got {embeddings.shape}")
        if labels.ndim != 1:
            labels = labels.view(-1)

        # --- Step 1: 特征归一化 ---
        # 将 512 维特征缩放到单位超球面上（L2 范数=1）
        # 归一化后距离计算只考虑向量方向，不受模长影响
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # --- Step 2: 计算 pairwise 距离矩阵 ---
        # dist_mat[i][j] = 样本i 与 样本j 的欧氏距离
        # 输出形状: [B, B]
        dist_mat = torch.cdist(embeddings, embeddings, p=2)

        # --- Step 3: 构建正负样本掩码 ---
        # same[i][j] = True 表示样本 i 和 j 是同一个体
        same = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]

        # eye[i][j] = True 表示 i == j（样本自己和自己）
        eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

        # pos_mask[i][j] = True 表示 j 是 i 的正样本（同类且不是自己）
        pos_mask = same & ~eye

        # neg_mask[i][j] = True 表示 j 是 i 的负样本（不同类）
        neg_mask = ~same

        # --- 边界情况 1: batch 中无法构成 triplet ---
        # 例如 batch 中只有一个个体，或者每个个体只有一张照片
        if not pos_mask.any() or not neg_mask.any():
            return embeddings.new_tensor(0.0)

        # --- Step 4: 找 Hardest Positive / Negative ---
        # 对每个样本 i，在 batch 中找：
        #   · hardest_pos[i]: 同类中距离最远的（max）
        #   · hardest_neg[i]: 异类中距离最近的（min）
        # 技巧：用 masked_fill 把不相关位置设为极值，再用 max/min 筛选

        # 把非正样本位置填 -inf，取 max 后只剩最远同类
        hardest_pos = dist_mat.masked_fill(~pos_mask, float('-inf')).max(dim=1).values

        # 把非负样本位置填 +inf，取 min 后只剩最近异类
        hardest_neg = dist_mat.masked_fill(~neg_mask, float('inf')).min(dim=1).values

        # --- Step 5: 筛选有效样本 ---
        # valid[i] = True 表示样本 i 在 batch 中既有正样本也有负样本
        valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
        if not valid.any():
            return embeddings.new_tensor(0.0)

        # --- Step 6: 计算 Triplet Loss ---
        # 公式: max(0, hardest_pos - hardest_neg + margin)
        # F.relu 等价于 max(0, x)，确保 loss >= 0
        losses = F.relu(hardest_pos[valid] - hardest_neg[valid] + self.margin)

        # 返回有效样本的平均 loss
        return losses.mean() if losses.numel() > 0 else embeddings.new_tensor(0.0)
