import torch
import torch.nn as nn
import torch.nn.functional as F


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
