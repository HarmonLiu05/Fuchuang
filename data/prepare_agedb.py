"""
AgeDB 人脸数据集加载器
支持按年龄（时间）划分训练/测试集，返回时间信息用于 APN Triplet Loss

文件名格式: ID_Name_Age_gender.jpg
- identity = Name（人名）
- time_info = Age（年龄作为"时间"信息）
"""
import os
import re
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch


class AgeDBDataset(Dataset):
    """AgeDB 数据集加载器"""

    def __init__(self, config, split='train', transform=None, return_time=False, identity_map=None):
        self.config = config
        self.split = split
        self.transform = transform
        self.return_time = return_time

        root_dir = config['data']['root_dir']
        if not os.path.exists(root_dir):
            raise RuntimeError(f"AgeDB 目录不存在: {root_dir}")

        # 第一遍：扫描所有图片，按人名分组
        person_images = defaultdict(list)
        pattern = re.compile(r'^\d+_([^_]+)_(\d+)_([mf])\.jpg$')

        for filename in os.listdir(root_dir):
            if not filename.endswith('.jpg'):
                continue
            match = pattern.match(filename)
            if not match:
                continue

            name = match.group(1)
            try:
                age = int(match.group(2))
            except ValueError:
                continue

            filepath = os.path.join(root_dir, filename)
            person_images[name].append({
                'path': filepath,
                'age': age,
                'filename': filename
            })

        # 过滤：只保留图片数 >= min_samples 的个体
        min_samples = config['data'].get('min_samples_per_identity', 5)
        valid_people = {
            name: imgs for name, imgs in person_images.items()
            if len(imgs) >= min_samples
        }

        if not valid_people:
            raise RuntimeError(f"没有符合条件的个体（min_samples={min_samples}）")

        # 第二遍：按年龄排序，7:3 划分训练/测试
        if identity_map is not None:
            self.identity_map = identity_map
            self.label_counter = len(identity_map)
        else:
            self.identity_map = {}
            self.label_counter = 0

        self.image_list = []
        train_count = 0
        test_count = 0

        for name in sorted(valid_people.keys()):
            imgs = sorted(valid_people[name], key=lambda x: x['age'])
            total = len(imgs)
            split_idx = max(1, int(total * 0.7))

            if self.split == 'train':
                selected = imgs[:split_idx]
            else:
                selected = imgs[split_idx:]

            if name not in self.identity_map:
                self.identity_map[name] = self.label_counter
                self.label_counter += 1

            label = self.identity_map[name]
            for img in selected:
                self.image_list.append({
                    'path': img['path'],
                    'label': label,
                    'age': float(img['age']),
                    'identity_name': name
                })

            if self.split == 'train':
                train_count += len(selected)
            else:
                test_count += len(selected)

        self.num_identities = len(self.identity_map)
        print(f"加载 AgeDB {split} 数据集: {len(self.image_list)} 张图片, "
              f"{self.num_identities} 个个体")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        sample = self.image_list[idx]

        try:
            image = Image.open(sample['path']).convert('RGB')
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"无法加载图片 {sample['path']}: {e}")

        if self.transform:
            image = self.transform(image)

        if self.return_time:
            return image, sample['label'], sample['age']

        return image, sample['label']


def prepare_agedb_dataloaders(config, return_time=False):
    """
    准备 AgeDB 数据集的 DataLoader

    Args:
        config: 配置字典
        return_time: 是否返回时间信息（用于 APN Triplet Loss）

    Returns:
        train_loader, test_loader, num_identities, train_dataset
    """
    from data.dataset import get_train_transform, get_val_transform

    train_dataset = AgeDBDataset(
        config,
        split='train',
        transform=get_train_transform(config),
        return_time=return_time
    )

    test_dataset = AgeDBDataset(
        config,
        split='test',
        transform=get_val_transform(config),
        identity_map=train_dataset.identity_map,
        return_time=return_time
    )

    num_identities = train_dataset.num_identities

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 0),
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader, num_identities, train_dataset
