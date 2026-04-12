"""
FGNET 人脸数据集加载器
支持年龄标注读取和时间划分训练集/测试集
"""
import os
import re
import pandas as pd
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from data.dataset import get_train_transform, get_val_transform


class FGNETDataset(Dataset):
    """FGNET 人脸数据集加载器"""

    def __init__(self, config, split='train', transform=None, return_time=False):
        """
        Args:
            config: 配置字典
            split: 'train' 或 'test'
            transform: 数据增强 transform
            return_time: 是否返回年龄信息 (用于时间 APN)
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.return_time = return_time

        # 数据集根目录
        self.fgnet_root = config['data']['fgnet_root']
        self.images_dir = os.path.join(self.fgnet_root, 'images')
        self.age_file = os.path.join(
            self.fgnet_root,
            config['data'].get('age_annotation_path', 'age_annotations/kara2015_ageannotations/age_groundtruth.csv')
        )

        # 读取年龄标注
        self.age_map = self._load_age_map()

        # 构建图片列表并按个体分组
        self.image_list = []
        self.identity_map = {}
        self.label_counter = 0

        self._build_image_list()

        # 按时间划分训练集/测试集
        self._split_dataset()

        print(f"加载 FGNET {split} 数据集: {len(self.image_list)} 张图片, "
              f"{self.num_identities} 个个体")

    def _load_age_map(self):
        """读取 age_groundtruth.csv 文件"""
        age_map = {}
        if not os.path.exists(self.age_file):
            raise FileNotFoundError(f"找不到年龄标注文件: {self.age_file}")

        # 读取 CSV (分号分隔)
        df = pd.read_csv(self.age_file, sep=';')

        for _, row in df.iterrows():
            filename = row['SampleID'].strip()
            age = int(row['Age'])
            age_map[filename] = age

        print(f"✓ 读取 {len(age_map)} 个年龄标注")
        return age_map

    def _build_image_list(self):
        """扫描 images 目录构建完整图片列表"""
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"找不到图片目录: {self.images_dir}")

        # 获取所有 JPG 文件
        image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.JPG'))
        ])

        # 按个体分组
        individuals = defaultdict(list)

        for img_file in image_files:
            # 提取个体ID: 001A02.JPG -> 001
            match = re.match(r'(\d{3})[aA]', img_file)
            if not match:
                continue

            individual_id = match.group(1)

            # 获取年龄
            if img_file not in self.age_map:
                print(f"警告: {img_file} 无年龄标注,跳过")
                continue

            age = self.age_map[img_file]

            individuals[individual_id].append({
                'filename': img_file,
                'path': os.path.join(self.images_dir, img_file),
                'age': age
            })

        # 过滤: 只保留图片数 >= min_samples 的个体
        min_samples = self.config['data'].get('min_samples_per_identity', 3)
        valid_individuals = {
            ind_id: imgs for ind_id, imgs in individuals.items()
            if len(imgs) >= min_samples
        }

        # 按年龄排序并分配标签
        for ind_id in sorted(valid_individuals.keys()):
            images = valid_individuals[ind_id]
            # 按年龄排序
            images.sort(key=lambda x: x['age'])

            # 分配身份标签
            if ind_id not in self.identity_map:
                self.identity_map[ind_id] = self.label_counter
                self.label_counter += 1

            label = self.identity_map[ind_id]

            # 添加到总列表
            for img_info in images:
                img_info['label'] = label
                img_info['identity_id'] = ind_id

            self.image_list.extend(images)

        self.num_identities = len(self.identity_map)
        print(f"✓ 加载 {len(self.image_list)} 张图片, {self.num_identities} 个个体")

    def _split_dataset(self):
        """按时间划分训练集和测试集 (70/30)"""
        # 按个体分组
        individuals = defaultdict(list)
        for img_info in self.image_list:
            individuals[img_info['identity_id']].append(img_info)

        # 使用公共方法划分数据集
        self.image_list = self._split_by_time(individuals)

    def _split_by_time(self, individuals):
        """按时间划分数据集的公共逻辑"""
        result_list = []
        for ind_id in sorted(individuals.keys()):
            images = individuals[ind_id]
            n = len(images)
            split_idx = max(1, int(n * 0.7))

            if self.split == 'train':
                result_list.extend(images[:split_idx])
            else:
                result_list.extend(images[split_idx:])

        return result_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        sample = self.image_list[idx]

        # 读取图片
        try:
            image = Image.open(sample['path']).convert('RGB')
        except (FileNotFoundError, IOError) as e:
            raise RuntimeError(f"无法加载图片 {sample['path']}: {e}")

        # 应用 transform
        if self.transform:
            image = self.transform(image)

        if self.return_time:
            # 返回年龄作为时间信息
            return image, sample['label'], float(sample['age'])

        return image, sample['label']

    def get_identity_list(self):
        """返回所有个体 ID 列表"""
        return list(self.identity_map.keys())


def prepare_fgnet_dataloaders(config, return_time=False, use_time_aware_sampler=False):
    """
    准备 FGNET 数据集的 DataLoader

    Args:
        config: 配置字典
        return_time: 是否返回年龄信息
        use_time_aware_sampler: 是否使用时间感知采样器

    Returns:
        train_loader, test_loader, num_identities, train_dataset
    """
    # 创建训练集
    train_dataset = FGNETDataset(
        config,
        split='train',
        transform=get_train_transform(config),
        return_time=return_time
    )

    # 创建测试集 (共享 identity_map)
    test_dataset = FGNETDataset(
        config,
        split='test',
        transform=get_val_transform(config),
        return_time=return_time
    )

    num_identities = train_dataset.num_identities

    # 创建 DataLoader
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 0)
    pin_memory = torch.cuda.is_available()

    if use_time_aware_sampler:
        from samplers.time_aware_sampler import TimeAwareBatchSampler
        sampler = TimeAwareBatchSampler(
            dataset=train_dataset,
            batch_size=batch_size,
            num_instances=4,
            drop_last=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        print(">>> 使用 TimeAwareBatchSampler")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader, num_identities, train_dataset
