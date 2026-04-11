"""
数据增强 transform 定义
"""
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
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=aug.get('random_erasing_p', 0.5),
            scale=aug.get('random_erasing_scale', (0.02, 0.15)),
            ratio=aug.get('random_erasing_ratio', (0.3, 3.3)),
            value='random'
        ),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def get_val_transform(config):
    """创建验证集增强"""
    return transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
