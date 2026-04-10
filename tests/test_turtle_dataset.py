"""
龟类数据集加载器测试
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import load_config
from data.turtle_dataset import TurtleDataset


def test_turtle_dataset():
    """测试龟类数据集加载器"""
    print("=" * 60)
    print("测试龟类数据集加载器")
    print("=" * 60)
    
    # 加载配置
    config = load_config('configs/config_turtle.yaml')
    print(f"\n✓ 配置文件加载成功")
    print(f"  数据集: {config['data']['dataset_name']}")
    print(f"  根目录: {config['data']['root_dir']}")
    
    # 创建训练集（不指定transform，仅测试数据加载）
    print("\n--- 创建训练集 ---")
    train_dataset = TurtleDataset(config, split='train', transform=None)
    
    print(f"✓ 训练集创建成功")
    print(f"  图片数: {len(train_dataset)}")
    print(f"  个体数: {train_dataset.num_identities}")
    print(f"  个体列表: {train_dataset.get_identity_list()[:10]}...")  # 显示前10个
    
    # 测试获取样本
    print("\n--- 测试样本读取 ---")
    image, label = train_dataset[0]
    print(f"✓ 样本读取成功")
    print(f"  图片类型: {type(image)}")
    print(f"  标签: {label}")
    print(f"  个体ID: {train_dataset.image_list[0]['identity_id']}")
    
    # 创建测试集（共享train集的identity_map）
    print("\n--- 创建测试集（共享identity_map）---")
    test_dataset = TurtleDataset(
        config, 
        split='test', 
        transform=None,
        identity_map=train_dataset.identity_map
    )
    
    print(f"✓ 测试集创建成功")
    print(f"  图片数: {len(test_dataset)}")
    print(f"  个体数: {test_dataset.num_identities}")
    
    # 验证identity_map一致性
    train_ids = set(train_dataset.identity_map.keys())
    test_ids = set(test_dataset.identity_map.keys())
    assert test_ids.issubset(train_ids), "测试集个体必须是训练集个体的子集"
    print(f"✓ identity_map一致性验证通过")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == '__main__':
    test_turtle_dataset()
