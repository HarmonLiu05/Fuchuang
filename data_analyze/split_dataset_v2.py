"""
按时间划分数据集 - 简洁版
规则：
1. 筛选：时间跨度 ≥ 3 年 且 照片 ≥ 15 张
2. 按天隔离：同一天的照片全部归入训练集或测试集
3. 时间间隔：训练集和测试集之间至少间隔 1 年
4. 划分比例：训练集 ~70%，测试集 ~30%
"""
import json
import os
from datetime import datetime
from collections import defaultdict

# ========== 配置 ==========
ANNOTATIONS_PATH = r"E:\fuchuang\turtlehead-dataset\Turtel_dataset\annotations.json"
OUTPUT_DIR = r"E:\fuchuang\turtlehead-dataset\Turtel_dataset\dataset_splits\dataset_G_ge3years_15photos_dayisolate"

MIN_SPAN_YEARS = 3.0
MIN_PHOTOS = 15
GAP_YEARS = 1.0
TRAIN_RATIO = 0.7

# ========== 加载数据 ==========
with open(ANNOTATIONS_PATH, 'r') as f:
    coco_data = json.load(f)

# 构建 image_id -> identity 和 image_id -> info 映射
image_to_identity = {ann['image_id']: ann['identity'] for ann in coco_data['annotations']}
image_info_map = {img['id']: img for img in coco_data['images']}

# ========== 按个体和日期分组 ==========
# identity -> { day_key -> [image_ids] }
identity_day_images = defaultdict(lambda: defaultdict(list))

for ann in coco_data['annotations']:
    image_id = ann['image_id']
    identity = ann['identity']
    img_info = image_info_map.get(image_id)
    
    if not img_info or not img_info.get('date'):
        continue
    
    try:
        dt = datetime.strptime(img_info['date'], '%Y:%m:%d %H:%M:%S')
        day_key = dt.strftime('%Y-%m-%d')
        identity_day_images[identity][day_key].append(image_id)
    except:
        continue

# ========== 筛选符合条件的个体 ==========
qualified_identities = {}

for identity, day_dict in identity_day_images.items():
    all_image_ids = []
    all_dates = []
    
    for day_key, img_ids in day_dict.items():
        all_image_ids.extend(img_ids)
        all_dates.append(datetime.strptime(day_key, '%Y-%m-%d'))
    
    if len(all_image_ids) < MIN_PHOTOS:
        continue
    
    span_years = (max(all_dates) - min(all_dates)).days / 365.25
    
    if span_years >= MIN_SPAN_YEARS:
        qualified_identities[identity] = {
            'day_dict': day_dict,
            'sorted_days': sorted(day_dict.keys()),
            'total_photos': len(all_image_ids),
            'span_years': span_years,
            'min_date': min(all_dates),
            'max_date': max(all_dates)
        }

print(f"筛选条件: 时间跨度 ≥ {MIN_SPAN_YEARS} 年, 照片 ≥ {MIN_PHOTOS} 张")
print(f"符合条件的个体数: {len(qualified_identities)}")
print(f"总照片数: {sum(v['total_photos'] for v in qualified_identities.values())}")

# ========== 按天划分 ==========
train_image_ids = set()
test_image_ids = set()
skipped = []

for identity, data in qualified_identities.items():
    sorted_days = data['sorted_days']
    
    if len(sorted_days) < 2:
        skipped.append(identity)
        continue
    
    # 寻找满足 gap 且接近目标比例的划分点
    target_idx = int(len(sorted_days) * TRAIN_RATIO)
    found = False
    
    for offset in range(len(sorted_days)):
        for idx in [target_idx - offset, target_idx + offset]:
            if idx < 1 or idx >= len(sorted_days):
                continue
            
            train_days = sorted_days[:idx]
            test_days = sorted_days[idx:]
            
            last_train = datetime.strptime(train_days[-1], '%Y-%m-%d')
            first_test = datetime.strptime(test_days[0], '%Y-%m-%d')
            gap = (first_test - last_train).days / 365.25
            
            if gap >= GAP_YEARS:
                for day in train_days:
                    train_image_ids.update(data['day_dict'][day])
                for day in test_days:
                    test_image_ids.update(data['day_dict'][day])
                found = True
                break
        
        if found:
            break
    
    if not found:
        # 无法满足 gap，中间划分
        mid = len(sorted_days) // 2
        for day in sorted_days[:mid]:
            train_image_ids.update(data['day_dict'][day])
        for day in sorted_days[mid:]:
            test_image_ids.update(data['day_dict'][day])

# ========== 生成 COCO 格式 ==========
def create_split_coco(image_ids_set):
    images = [image_info_map[iid] for iid in image_ids_set if iid in image_info_map]
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in image_ids_set]
    return {
        'images': images,
        'annotations': annotations,
        'categories': coco_data.get('categories', [])
    }

train_coco = create_split_coco(train_image_ids)
test_coco = create_split_coco(test_image_ids)

# ========== 保存 ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, 'train.json'), 'w') as f:
    json.dump(train_coco, f)

with open(os.path.join(OUTPUT_DIR, 'test.json'), 'w') as f:
    json.dump(test_coco, f)

# 统计信息
print(f"\n训练集: {len(train_image_ids)} 张图片")
print(f"测试集: {len(test_image_ids)} 张图片")
total = len(train_image_ids) + len(test_image_ids)
print(f"比例: {len(train_image_ids)/total:.2f} / {len(test_image_ids)/total:.2f}")

# 生成个体统计
stats = {
    'dataset_name': 'dataset_G_ge3years_15photos_dayisolate',
    'criteria': {'min_span_years': MIN_SPAN_YEARS, 'min_photos': MIN_PHOTOS, 'gap_years': GAP_YEARS},
    'num_identities': len(qualified_identities) - len(skipped),
    'skipped_identities': skipped,
    'train_images': len(train_image_ids),
    'test_images': len(test_image_ids),
    'identities': {}
}

for identity, data in qualified_identities.items():
    if identity in skipped:
        continue
    
    train_count = sum(len(data['day_dict'][d]) for d in data['sorted_days'] 
                     if any(iid in train_image_ids for iid in data['day_dict'][d]))
    test_count = sum(len(data['day_dict'][d]) for d in data['sorted_days'] 
                    if any(iid in test_image_ids for iid in data['day_dict'][d]))
    
    stats['identities'][identity] = {
        'total': data['total_photos'],
        'train': train_count,
        'test': test_count,
        'span_years': round(data['span_years'], 2)
    }

with open(os.path.join(OUTPUT_DIR, 'split_statistics.json'), 'w') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print(f"\n保存到: {OUTPUT_DIR}")
