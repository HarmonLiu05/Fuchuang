"""测试 AgeDB 数据加载器"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.prepare_agedb import AgeDBDataset, prepare_agedb_dataloaders


def test_agedb_dataset_creation():
    """测试数据集创建"""
    config = {
        'data': {
            'root_dir': '/workspace/AgeDB',
            'min_samples_per_identity': 5
        },
        'training': {'batch_size': 32},
        'augmentation': {}
    }

    train_dataset = AgeDBDataset(config, split='train', return_time=True)
    test_dataset = AgeDBDataset(config, split='test', return_time=True,
                               identity_map=train_dataset.identity_map)

    assert len(train_dataset) > 0, "训练集为空"
    assert len(test_dataset) > 0, "测试集为空"
    assert train_dataset.num_identities > 0, "个体数为 0"

    print(f"✓ 训练集: {len(train_dataset)} 张图片, {train_dataset.num_identities} 个个体")
    print(f"✓ 测试集: {len(test_dataset)} 张图片")

    # 测试返回格式
    image, label, age = train_dataset[0]
    print(f"✓ 样本: label={label}, age={age}, image_shape={image.size if hasattr(image, 'size') else 'N/A'}")


def test_agedb_dataloaders():
    """测试 DataLoader 创建"""
    config = {
        'data': {
            'root_dir': '/workspace/AgeDB',
            'min_samples_per_identity': 5,
            'num_workers': 0,
            'image_size': 112
        },
        'training': {'batch_size': 32},
        'augmentation': {}
    }

    train_loader, test_loader, num_ids, _ = prepare_agedb_dataloaders(
        config, return_time=True
    )

    assert len(train_loader) > 0
    assert len(test_loader) > 0
    assert num_ids > 0

    print(f"✓ DataLoader: train={len(train_loader)} batches, test={len(test_loader)} batches")

    # 测试 batch 返回
    batch = next(iter(train_loader))
    assert len(batch) == 3, f"期望返回 (image, label, age)，实际返回 {len(batch)} 个元素"
    images, labels, ages = batch
    print(f"✓ Batch: images={images.shape}, labels={labels.shape}, ages={ages.shape}")


if __name__ == '__main__':
    print("=" * 50)
    print("测试 AgeDB 数据加载器")
    print("=" * 50)

    test_agedb_dataset_creation()
    print()
    test_agedb_dataloaders()

    print()
    print("=" * 50)
    print("✓ 所有测试通过!")
