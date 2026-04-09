import numpy as np
import torch


def compute_accuracy(similarity, labels):
    sim_np = similarity.numpy().copy()
    labels_np = labels.numpy()
    
    np.fill_diagonal(sim_np, -1.0)
    
    nearest_idx = np.argmax(sim_np, axis=1)
    pred_labels = labels_np[nearest_idx]
    
    accuracy = np.mean(pred_labels == labels_np)
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
