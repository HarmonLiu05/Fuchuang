import numpy as np
import torch


def compute_accuracy(similarity, labels):
    """
    计算集合内部 Rank-1 准确率（Self-Similarity）。
    用于评估特征聚类效果。
    """
    sim_np = similarity.numpy().copy()
    labels_np = labels.numpy()

    np.fill_diagonal(sim_np, -1.0)

    nearest_idx = np.argmax(sim_np, axis=1)
    pred_labels = labels_np[nearest_idx]

    accuracy = np.mean(pred_labels == labels_np)
    return float(accuracy)


def compute_rank1_accuracy(query_features, query_labels, gallery_features, gallery_labels):
    """
    计算 Rank-1 识别率（Identification Accuracy）。
    在 Gallery（底库/训练集）中寻找与 Query（测试集/野外集）样本最相似的图片，
    判断是否为同一个体。
    """
    # 计算相似度矩阵 (Query x Gallery)
    similarity = torch.mm(query_features, gallery_features.t()).cpu().numpy()
    
    # 找到每个 Query 在 Gallery 中相似度最高的索引
    nearest_gallery_indices = np.argmax(similarity, axis=1)
    
    # 获取 Gallery 中对应样本的标签
    pred_labels = gallery_labels[nearest_gallery_indices]
    
    # 计算准确率
    accuracy = np.mean(query_labels == pred_labels)
    return float(accuracy)


def compute_tar_at_far(features, labels, target_far=0.001):
    similarity_matrix = torch.mm(features, features.t())
    
    sim = similarity_matrix.numpy().copy()
    labels_np = labels.numpy()
    
    same_identity = np.equal(labels_np[:, None], labels_np[None, :])
    np.fill_diagonal(sim, -1.0)
    
    positive_scores = sim[same_identity]
    negative_scores = sim[~same_identity]
    
    thresholds = np.linspace(-1, 1, 1000)
    results = []
    
    for threshold in thresholds:
        far = np.mean(negative_scores >= threshold)
        tar = np.mean(positive_scores >= threshold)
        results.append({'threshold': threshold, 'far': far, 'tar': tar})
    
    best_match = min(results, key=lambda x: abs(x['far'] - target_far))
    return best_match['tar'], best_match['threshold']


def compute_all_metrics(features, labels):
    similarity = torch.mm(features, features.t())
    accuracy = compute_accuracy(similarity, labels)
    tar_at_far, threshold = compute_tar_at_far(features, labels, target_far=0.001)
    
    return {
        'accuracy': accuracy,
        'tar_at_far_0.1': tar_at_far,
        'threshold': threshold
    }
