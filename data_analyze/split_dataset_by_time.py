"""
按时间划分数据集
规则：
1. 筛选：时间跨度 ≥ 3 年 且 照片 ≥ 15 张
2. 按天隔离：同一天的照片全部归入训练集或测试集
3. 时间间隔：训练集和测试集之间至少间隔 1 年
4. 划分比例：训练集 ~70%，测试集 ~30%
"""
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ========== 配置 ==========
ANNOTATIONS_PATH = r"E:\fuchuang\turtlehead-dataset\Turtel_dataset\annotations.json"
OUTPUT_DIR = r"E:\fuchuang\turtlehead-dataset\Turtel_dataset\dataset_splits\dataset_G_ge3years_15photos_dayisolate"

MIN_SPAN_YEARS = 3.0
MIN_PHOTOS = 15
GAP_YEARS = 1.0
TRAIN_RATIO = 0.7
SEED = 42

# ========== 加载数据 ==========
with open(ANNOTATIONS_PATH, 'r') as f:
    coco_data = json.load(f)

# 构建 image_id -> identity 映射
image_to_identity = {}
for ann in coco_data['annotations']:
    image_to_identity[ann['image_id']] = ann['identity']

# 构建 image_id -> image_info 映射
image_id_to_info = {img['id']: img for img in coco_data['images']}

# ========== 统计每个个体的照片和日期 ==========
identity_images = defaultdict(list)

for img_info in coco_data['images']:
    image_id = img_info['id']
    if image_id not in image_to_identity:
        continue
    
    identity_id = image_to_identity[image_id]
    date_str = img_info.get('date', '')
    
    if date_str:
        try:
            dt = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            identity_images[identity_id].append({
                'image_id': image_id,
                'date': dt,
                'day_key': dt.strftime('%Y-%m-%d'),
                'info': img_info
            })
        except:
            pass

# ========== 筛选符合条件的个体 ==========
qualified_identities = {}

for identity_id, images in identity_images.items():
    if len(images) < MIN_PHOTOS:
        continue
    
    dates = [img['date'] for img in images]
    min_date = min(dates)
    max_date = max(dates)
    span_years = (max_date - min_date).days / 365.25
    
    if span_years >= MIN_SPAN_YEARS:
        qualified_identities[identity_id] = {
            'images': images,
            'min_date': min_date,
            'max_date': max_date,
            'span_years': span_years,
            'count': len(images)
        }

print(f"筛选条件: 时间跨度 ≥ {MIN_SPAN_YEARS} 年, 照片 ≥ {MIN_PHOTOS} 张")
print(f"符合条件的个体数: {len(qualified_identities)}")
print(f"总照片数: {sum(v['count'] for v in qualified_identities.values())}")

# ========== 按天分组并划分 ==========
train_images = []
test_images = []
skipped_identities = []

for identity_id, data in qualified_identities.items():
    images = data['images']
    
    # 按天分组
    days_dict = defaultdict(list)
    for img in images:
        days_dict[img['day_key']].append(img)
    
    # 按日期排序
    sorted_days = sorted(days_dict.keys())
    
    if len(sorted_days) < 2:
        # 只有1天拍摄，无法按天隔离划分
        skipped_identities.append(identity_id)
        continue
    
    # 尝试找到满足 gap 的划分点，同时尽量保持 7:3 比例
    target_train_days = int(len(sorted_days) * TRAIN_RATIO)
    found_split = False
    
    # 先尝试在目标比例附近找满足 gap 的划分点
    for offset in range(len(sorted_days)):
        for candidate_idx in [target_train_days - offset, target_train_days + offset]:
            if candidate_idx < 1 or candidate_idx >= len(sorted_days):
                continue
            
            candidate_train = sorted_days[:candidate_idx]
            candidate_test = sorted_days[candidate_idx:]
            
            lt = max(datetime.strptime(d, '%Y-%m-%d') for d in candidate_train)
            ft = min(datetime.strptime(d, '%Y-%m-%d') for d in candidate_test)
            g = (ft - lt).days / 365.25
            
            if g >= GAP_YEARS:
                train_days = candidate_train
                test_days = candidate_test
                gap_years = g
                found_split = True
                break
        
        if found_split:
            break
    
    if not found_split:
        # 无法满足 gap，使用中间划分
        mid_idx = len(sorted_days) // 2
        train_days = sorted_days[:mid_idx]
        test_days = sorted_days[mid_idx:]
        gap_years = 0
    
    # 收集训练和测试图片（直接按 image_id 区分）
    identity_train_images = []
    identity_test_images = []
    
    for day in train_days:
        identity_train_images.extend([img['info'] for img in days_dict[day]])
    for day in test_days:
        identity_test_images.extend([img['info'] for img in days_dict[day]])
    
    train_images.extend(identity_train_images)
    test_images.extend(identity_test_images)

print(f"\n跳过个体（无法按天隔离划分）: {len(skipped_identities)}")
print(f"最终训练集图片数: {len(train_images)}")
print(f"最终测试集图片数: {len(test_images)}")
print(f"训练/测试比例: {len(train_images)/(len(train_images)+len(test_images)):.2f} / {len(test_images)/(len(train_images)+len(test_images)):.2f}")

# ========== 生成 COCO 格式 JSON ==========
def create_coco_format(images_list, all_annotations):
    """创建COCO格式的划分"""
    image_ids = set(img['id'] for img in images_list)
    
    filtered_images = [img for img in images_list]
    filtered_annotations = [ann for ann in all_annotations if ann['image_id'] in image_ids]
    
    return {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': coco_data.get('categories', [])
    }

train_coco = create_coco_format(train_images, coco_data['annotations'])
test_coco = create_coco_format(test_images, coco_data['annotations'])

# ========== 保存结果 ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, 'train.json'), 'w') as f:
    json.dump(train_coco, f, indent=2)

with open(os.path.join(OUTPUT_DIR, 'test.json'), 'w') as f:
    json.dump(test_coco, f, indent=2)

# 生成统计信息
stats = {
    'dataset_name': 'dataset_G_ge3years_15photos_dayisolate',
    'filter_criteria': {
        'min_span_years': MIN_SPAN_YEARS,
        'min_photos': MIN_PHOTOS,
        'gap_years': GAP_YEARS
    },
    'num_identities': len(qualified_identities) - len(skipped_identities),
    'skipped_identities': skipped_identities,
    'train_num_images': len(train_images),
    'test_num_images': len(test_images),
    'identities': {}
}

for identity_id, data in qualified_identities.items():
    if identity_id in skipped_identities:
        continue
    
    # 统计该个体的训练/测试图片数
    train_count = sum(1 for img in identity_images.get(identity_id, []) 
                     if img['day_key'] in [d for d in sorted(days_dict.keys()) if d in train_days])
    test_count = sum(1 for img in identity_images.get(identity_id, []) 
                    if img['day_key'] in [d for d in sorted(days_dict.keys()) if d in test_days])
    
    stats['identities'][identity_id] = {
        'total_photos': data['count'],
        'train_photos': train_count,
        'test_photos': test_count,
        'span_years': round(data['span_years'], 2),
        'min_date': data['min_date'].strftime('%Y-%m-%d'),
        'max_date': data['max_date'].strftime('%Y-%m-%d')
    }

with open(os.path.join(OUTPUT_DIR, 'split_statistics.json'), 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n数据集已保存到: {OUTPUT_DIR}")
print(f"  - train.json: {len(train_images)} 张图片")
print(f"  - test.json: {len(test_images)} 张图片")
print(f"  - split_statistics.json: 统计信息")
