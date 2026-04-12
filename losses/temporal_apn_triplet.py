import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAPNTripletLoss(nn.Module):
    """
    时间跨度感知 APN Triplet Loss (精确到月)

    核心思想：
        - Anchor (A): 当前样本
        - Positive (P): 同类中【时间跨度最远】的样本
                      如果时间跨度相同，选【特征距离最远】的样本（Tie-breaker）
        - Negative (N): 异类中【时间跨度最近】的样本
                      如果时间跨度相同，选【特征距离最近】的样本（Tie-breaker）

    时间计算规则：
        - 只精确到【年-月】（忽略日、时、分）
        - 2010:07:02 与 2010:07:20 视为同一时间，跨度为 0
    """
    def __init__(self, margin: float = 0.3, normalize_embeddings: bool = True):
        super().__init__()
        self.margin = margin
        self.normalize_embeddings = normalize_embeddings

    def _truncate_to_month(self, times: torch.Tensor) -> torch.Tensor:
        """
        将连续时间截断到月份（忽略日、时、分）
        例如：2010.25 (4月) -> 2010.25; 2010.28 (4月) -> 2010.25
        """
        years = torch.floor(times)
        # 假设一年12个月，将小数部分映射到月份
        months = torch.floor((times - years) * 12)
        # 返回每个月的中间值（代表该月）
        return years + (months + 0.5) / 12.0

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

        # --- Step 4: 截断时间并计算时间差距 ---
        # 先将时间截断到月份（忽略日、时、分），减少微小波动干扰
        valid_time = ~torch.isnan(times)
        times_processed = times.clone()
        
        # 先截断有效时间（避免 NaN 值被 _truncate_to_month 计算产生奇怪数值）
        if valid_time.any():
            times_processed[valid_time] = self._truncate_to_month(times_processed[valid_time])
        
        # 再将无效时间设为标记值
        times_processed[~valid_time] = -100.0

        # time_diff[i][j] = |time[i] - time[j]|
        time_a = times_processed.unsqueeze(1)  # [B, 1]
        time_b = times_processed.unsqueeze(0)  # [1, B]
        time_diff = torch.abs(time_a - time_b)  # [B, B]
        time_diff[~valid_time.unsqueeze(0).expand(B, B)] = -100.0  # 标记无效行列
        time_diff[:, ~valid_time] = -100.0  # 标记无效列

        # --- Step 5: 对每个 Anchor，选择 APN ---
        selected_pos_idx = torch.full((B,), -1, dtype=torch.long, device=labels.device)
        selected_neg_idx = torch.full((B,), -1, dtype=torch.long, device=labels.device)
        
        # 标记是否成功选择了样本
        valid_selection = torch.zeros(B, dtype=torch.bool, device=labels.device)

        for i in range(B):
            # === Positive: 时间跨度最远 -> 距离最远 ===
            pos_candidates = pos_mask[i].clone() & valid_time[i]
            if pos_candidates.any():
                # 1. 找最大时间差
                pos_td = time_diff[i].masked_fill(~pos_candidates, -100.0)
                max_td = pos_td.max()
                
                # 2. 找出所有具有该最大时间差的样本
                max_td_candidates = (pos_td == max_td)
                
                # 3. 在候选样本中，找特征距离最远的
                pos_dist = dist_mat[i].masked_fill(~max_td_candidates, -100.0)
                selected_pos_idx[i] = pos_dist.argmax()
                valid_selection[i] = True

            # === Negative: 时间跨度最近 -> 距离最近 ===
            neg_candidates = neg_mask[i].clone() & valid_time[i]
            if neg_candidates.any():
                # 1. 找最小时间差
                neg_td = time_diff[i].masked_fill(~neg_candidates, 100.0)
                min_td = neg_td.min()
                
                # 2. 找出所有具有该最小时间差的样本
                min_td_candidates = (neg_td == min_td)
                
                # 3. 在候选样本中，找特征距离最近的
                neg_dist = dist_mat[i].masked_fill(~min_td_candidates, 100.0)
                selected_neg_idx[i] = neg_dist.argmin()
                valid_selection[i] = True
        # --- Step 6: 筛选有效样本 ---
        valid = valid_selection
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
