"""
龟类数据集COCO格式加载器
支持读取 train.json / test.json 并按个体ID分配标签
"""
import os
import json
from PIL import Image
from torch.utils.data import Dataset


class TurtleDataset(Dataset):
    """龟类数据集COCO格式加载器"""
    
    def __init__(self, config, split='train', transform=None, identity_map=None):
        """
        Args:
            config: 配置字典
            split: 'train' 或 'test'
            transform: 数据增强transform
            identity_map: 身份映射字典（test集共享train集的映射）
        """
        self.config = config
        self.split = split
        self.transform = transform
        
        # 构建JSON路径
        root_dir = config['data']['root_dir']
        splits_dir = config['data']['splits_dir']
        dataset_name = config['data']['dataset_name']
        json_file = config['data'][f'{split}_json']
        json_path = os.path.join(root_dir, splits_dir, dataset_name, json_file)
        
        # 加载COCO标注
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 解析images数组
        self.image_list = []
        
        if identity_map is not None:
            # test集：共享train集的identity_map
            self.identity_map = identity_map
            self.label_counter = len(identity_map)
        else:
            # train集：创建新的identity_map
            self.identity_map = {}
            self.label_counter = 0
        
        # 第一遍：统计每个个体的图片数
        identity_counts = {}
        for image_info in coco_data['images']:
            image_path = image_info['path']
            # path格式: images/t007/xxx.JPG，个体ID是第二级目录
            path_parts = image_path.replace('\\', '/').split('/')
            identity_id = path_parts[1] if len(path_parts) > 1 else path_parts[0]
            
            identity_counts[identity_id] = identity_counts.get(identity_id, 0) + 1
        
        # 过滤：只保留图片数 >= min_samples 的个体
        min_samples = config['data'].get('min_samples_per_identity', 1)
        valid_identities = {
            identity_id for identity_id, count in identity_counts.items()
            if count >= min_samples
        }
        
        # 第二遍：构建image_list（仅包含有效个体）
        for image_info in coco_data['images']:
            image_path = image_info['path']
            # path格式: images/t007/xxx.JPG，个体ID是第二级目录
            path_parts = image_path.replace('\\', '/').split('/')
            identity_id = path_parts[1] if len(path_parts) > 1 else path_parts[0]
            
            # 跳过图片数不足的个体
            if identity_id not in valid_identities:
                continue
            
            # 构建完整图片路径（直接拼接 root_dir + path）
            full_path = os.path.join(
                config['data']['root_dir'],
                image_path
            )
            
            # 分配标签
            if identity_id not in self.identity_map:
                self.identity_map[identity_id] = self.label_counter
                self.label_counter += 1
            
            label = self.identity_map[identity_id]
            self.image_list.append({
                'path': full_path,
                'identity_id': identity_id,
                'label': label,
                'date': image_info.get('date', None)
            })
        
        # 统计信息
        self.num_identities = len(self.identity_map)
        print(f"加载 {split} 数据集: {len(self.image_list)} 张图片, "
              f"{self.num_identities} 个个体")
        
        # 验证test集的identity_map一致性
        if split == 'test' and identity_map is not None:
            missing = set(self.identity_map.keys()) - set(identity_map.keys())
            if missing:
                print(f"警告: 测试集中存在训练集未出现的个体: {missing}")
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        sample = self.image_list[idx]
        
        # 读取图片
        image = Image.open(sample['path']).convert('RGB')
        
        # 应用transform
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']
    
    def get_identity_list(self):
        """返回所有个体ID列表"""
        return list(self.identity_map.keys())
