"""
划分训练集和测试集到两个文件夹
按个体划分：每个个体的部分样本进训练集，其余进测试集
"""
import os
import sys
import shutil
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import ChimpanzeeDataset
from utils.utils import load_config


def split_dataset_to_folders(config, output_root, train_ratio=0.8, seed=42):
    """
    将数据集划分为 train/test 两个文件夹
    """
    # 加载数据集
    full_dataset = ChimpanzeeDataset(
        root_dir=config['data']['root_dir'],
        annotation_file=config['data']['annotation_file'],
        image_dir=config['data']['image_dir'],
        min_samples_per_identity=config['data'].get('min_samples_per_identity', 20)
    )

    source_root = os.path.join(config['data']['root_dir'], config['data']['image_dir'])

    train_dir = os.path.join(output_root, 'train')
    test_dir = os.path.join(output_root, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"数据源: {source_root}")
    print(f"输出目录: {output_root}")
    print(f"划分比例: train={train_ratio}, test={1-train_ratio}")
    print(f"个体数: {len(full_dataset.identities)}")
    print(f"总样本: {len(full_dataset.samples)}")

    total_train = 0
    total_test = 0

    # 按个体划分
    for identity_name in full_dataset.identities:
        identity_samples = [
            s for s in full_dataset.samples
            if s['identity'] == identity_name
        ]

        train_samples, test_samples = train_test_split(
            identity_samples,
            train_size=train_ratio,
            random_state=seed
        )

        # 创建个体文件夹
        train_identity_dir = os.path.join(train_dir, identity_name)
        test_identity_dir = os.path.join(test_dir, identity_name)
        os.makedirs(train_identity_dir, exist_ok=True)
        os.makedirs(test_identity_dir, exist_ok=True)

        # 复制训练集
        for sample in train_samples:
            src = os.path.join(source_root, sample['filename'])
            dst = os.path.join(train_identity_dir, os.path.basename(sample['filename']))
            if os.path.exists(src):
                shutil.copy2(src, dst)
                total_train += 1

        # 复制测试集
        for sample in test_samples:
            src = os.path.join(source_root, sample['filename'])
            dst = os.path.join(test_identity_dir, os.path.basename(sample['filename']))
            if os.path.exists(src):
                shutil.copy2(src, dst)
                total_test += 1

    print(f"\n划分完成！")
    print(f"训练集: {total_train} 样本 → {train_dir}")
    print(f"测试集: {total_test} 样本 → {test_dir}")


if __name__ == '__main__':
    config = load_config('configs/config_local.yaml')
    output_root = '../chimpanzee_faces/datasets_cropped_chimpanzee_faces/data_CZoo_split'
    split_dataset_to_folders(config, output_root, train_ratio=0.8)
