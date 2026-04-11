import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from torch.utils.data import Sampler



def _to_numeric_time(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
            try:
                dt = datetime.strptime(text, fmt)
                return float(dt.year) + (dt.timetuple().tm_yday - 1) / 366.0
            except ValueError:
                continue
        digits = ''.join(ch for ch in text if ch.isdigit())
        if len(digits) >= 4:
            return float(digits[:4])
    return None


class TimeAwareBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, num_instances: int = 4, drop_last: bool = True):
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.drop_last = drop_last

        self.index_dic: Dict[int, List[int]] = defaultdict(list)
        self.time_dic: Dict[int, List[Optional[float]]] = defaultdict(list)

        for idx, sample in enumerate(dataset.image_list):
            label = sample['label']
            self.index_dic[label].append(idx)
            self.time_dic[label].append(_to_numeric_time(sample.get('date')))

        self.labels = [label for label, idxs in self.index_dic.items() if len(idxs) > 0]
        self.length = max(1, len(self.labels) // self.num_pids_per_batch)

    def _sample_k_for_identity(self, label: int) -> List[int]:
        idxs = self.index_dic[label]
        times = self.time_dic[label]

        if len(idxs) <= self.num_instances:
            return random.choices(idxs, k=self.num_instances)

        valid_pairs = [(idx, t) for idx, t in zip(idxs, times) if t is not None]
        if len(valid_pairs) >= self.num_instances:
            sorted_pairs = sorted(valid_pairs, key=lambda x: x[1])
            chosen = []
            left, right = 0, len(sorted_pairs) - 1
            while len(chosen) < self.num_instances and left <= right:
                chosen.append(sorted_pairs[left][0])
                if len(chosen) < self.num_instances and right != left:
                    chosen.append(sorted_pairs[right][0])
                left += 1
                right -= 1
            random.shuffle(chosen)
            return chosen[:self.num_instances]

        return random.sample(idxs, self.num_instances)

    def __iter__(self):
        labels = self.labels.copy()
        random.shuffle(labels)

        batch = []
        for label in labels:
            batch.extend(self._sample_k_for_identity(label))
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return self.length
