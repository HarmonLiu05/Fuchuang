import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class ChimpanzeeDataset(Dataset):
    def __init__(self, root_dir, annotation_file, image_dir, transform=None, 
                 min_samples_per_identity=20, identities=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_dir)
        self.transform = transform
        
        self.samples = self._parse_annotations(
            os.path.join(root_dir, annotation_file),
            min_samples_per_identity,
            identities
        )
        
        self.identity_to_idx = {name: i for i, name in enumerate(self.identities)}
        
    def _parse_annotations(self, ann_file, min_samples, specified_identities):
        identity_samples = defaultdict(list)
        
        with open(ann_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                filename = parts[0]
                identity = self._extract_identity_from_filename(filename)
                
                if identity:
                    identity_samples[identity].append(filename)
        
        if specified_identities:
            self.identities = specified_identities
        else:
            self.identities = [
                name for name, samples in identity_samples.items()
                if len(samples) >= min_samples
            ]
        
        self.identity_to_idx = {name: i for i, name in enumerate(self.identities)}
        
        samples = []
        for identity in self.identities:
            for filename in identity_samples[identity]:
                samples.append({
                    'filename': filename,
                    'identity': identity,
                    'label': self.identity_to_idx[identity]
                })
        
        return samples
    
    def _extract_identity_from_filename(self, filename):
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        if len(parts) >= 1:
            return parts[0]
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.image_dir, sample['filename'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, sample['label']


from torchvision import transforms


def get_train_transform(config):
    """根据配置创建训练集增强"""
    aug = config.get('augmentation', {})
    
    return transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.RandomHorizontalFlip(p=aug.get('random_horizontal_flip', 0.5)),
        transforms.RandomAffine(
            degrees=aug.get('random_affine_degrees', 15),
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(
            brightness=aug.get('color_jitter_brightness', 0.3),
            contrast=aug.get('color_jitter_contrast', 0.3),
            saturation=aug.get('color_jitter_saturation', 0.2),
            hue=aug.get('color_jitter_hue', 0.1)
        ),
        transforms.GaussianBlur(
            kernel_size=aug.get('gaussian_blur_kernel', 3),
            sigma=aug.get('gaussian_blur_sigma', (0.1, 2.0))
        ),
        transforms.RandomErasing(
            p=aug.get('random_erasing_p', 0.5),
            scale=aug.get('random_erasing_scale', (0.02, 0.15)),
            ratio=aug.get('random_erasing_ratio', (0.3, 3.3)),
            value='random'
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transform(config):
    """创建验证集增强"""
    return transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
