"""
分析数据集个体分布和划分规则
"""
import json
from datetime import datetime
from collections import defaultdict

# 加载标注文件
ANNOTATIONS_PATH = r"E:\fuchuang\turtlehead-dataset\Turtel_dataset\annotations.json"

with open(ANNOTATIONS_PATH, 'r') as f:
    coco_data = json.load(f)

# 构建 image_id -> identity 映射
image_to_identity = {}
for ann in coco_data['annotations']:
    image_to_identity[ann['image_id']] = ann['identity']

# 1. 统计每个个体的照片数量和时间范围
identity_stats = defaultdict(lambda: {
    'count': 0,
    'dates': [],
    'images': []
})

for img_info in coco_data['images']:
    image_id = img_info['id']
    if image_id not in image_to_identity:
        continue
    
    identity_id = image_to_identity[image_id]
    stats = identity_stats[identity_id]
    stats['count'] += 1
    stats['images'].append(img_info)
    
    # 解析日期
    date_str = img_info.get('date', '')
    if date_str:
        try:
            dt = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
            stats['dates'].append(dt)
        except:
            pass

# 2. 计算每个个体的时间跨度
print("=" * 80)
print("数据集个体统计分析")
print("=" * 80)
print(f"\n总个体数: {len(identity_stats)}")
print(f"总照片数: {sum(s['count'] for s in identity_stats.values())}")

# 计算时间跨度
for identity_id, stats in identity_stats.items():
    if stats['dates']:
        stats['min_date'] = min(stats['dates'])
        stats['max_date'] = max(stats['dates'])
        stats['span_days'] = (stats['max_date'] - stats['min_date']).days
        stats['span_years'] = stats['span_days'] / 365.25
    else:
        stats['span_years'] = 0

# 3. 不同筛选条件下的统计
print("\n" + "=" * 80)
print("筛选条件分析")
print("=" * 80)

filters = [
    ("所有个体", lambda x: True),
    ("照片 ≥ 10 张", lambda x: x['count'] >= 10),
    ("照片 ≥ 15 张", lambda x: x['count'] >= 15),
    ("照片 ≥ 20 张", lambda x: x['count'] >= 20),
    ("照片 ≥ 30 张", lambda x: x['count'] >= 30),
    ("时间跨度 ≥ 3 年", lambda x: x['span_years'] >= 3),
    ("时间跨度 ≥ 5 年", lambda x: x['span_years'] >= 5),
    ("时间跨度 ≥ 3 年 且 照片 ≥ 15 张", lambda x: x['span_years'] >= 3 and x['count'] >= 15),
    ("时间跨度 ≥ 3 年 且 照片 ≥ 20 张", lambda x: x['span_years'] >= 3 and x['count'] >= 20),
    ("时间跨度 ≥ 5 年 且 照片 ≥ 15 张", lambda x: x['span_years'] >= 5 and x['count'] >= 15),
]

for name, condition in filters:
    filtered = {k: v for k, v in identity_stats.items() if condition(v)}
    if filtered:
        counts = [v['count'] for v in filtered.values()]
        spans = [v['span_years'] for v in filtered.values() if v['span_years'] > 0]
        print(f"\n{name}:")
        print(f"  个体数: {len(filtered)}")
        print(f"  总照片: {sum(counts)}")
        print(f"  照片数: 最小={min(counts)}, 最大={max(counts)}, 平均={sum(counts)/len(counts):.1f}")
        if spans:
            print(f"  时间跨度: 最小={min(spans):.2f}年, 最大={max(spans):.2f}年, 平均={sum(spans)/len(spans):.2f}年")

# 4. 按时间跨度分布
print("\n" + "=" * 80)
print("时间跨度分布（所有个体）")
print("=" * 80)
span_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 100)]
for min_y, max_y in span_ranges:
    count = sum(1 for v in identity_stats.values() if min_y <= v['span_years'] < max_y)
    if count > 0:
        print(f"  {min_y:.0f}-{max_y:.0f} 年: {count} 个个体")

# 5. 按照片数量分布
print("\n" + "=" * 80)
print("照片数量分布（所有个体）")
print("=" * 80)
count_ranges = [(0, 10), (10, 20), (20, 30), (30, 50), (50, 80), (80, 120), (120, 200), (200, 1000)]
for min_c, max_c in count_ranges:
    count = sum(1 for v in identity_stats.values() if min_c <= v['count'] < max_c)
    if count > 0:
        print(f"  {min_c}-{max_c} 张: {count} 个个体")

# 6. 找出照片最多的前20个个体
print("\n" + "=" * 80)
print("照片最多的前20个个体")
print("=" * 80)
sorted_by_count = sorted(identity_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
for identity_id, stats in sorted_by_count:
    print(f"  {identity_id}: {stats['count']} 张, 时间跨度 {stats['span_years']:.2f} 年")

# 7. 按天隔离分析：统计每个个体有多少个不同的拍摄日期
print("\n" + "=" * 80)
print("按天隔离可行性分析（照片 ≥ 15 且 时间跨度 ≥ 3 年的个体）")
print("=" * 80)

qualified = {k: v for k, v in identity_stats.items() 
             if v['count'] >= 15 and v['span_years'] >= 3}

for identity_id, stats in qualified.items():
    # 按天分组
    dates_by_day = defaultdict(int)
    for dt in stats['dates']:
        day_key = dt.strftime('%Y-%m-%d')
        dates_by_day[day_key] += 1
    
    stats['unique_days'] = len(dates_by_day)
    stats['max_photos_per_day'] = max(dates_by_day.values())
    stats['photos_per_day'] = dict(dates_by_day)

# 统计
unique_days_list = [v['unique_days'] for v in qualified.values()]
max_per_day_list = [v['max_photos_per_day'] for v in qualified.values()]

print(f"\n符合条件的个体数: {len(qualified)}")
print(f"\n拍摄天数分布:")
print(f"  最少: {min(unique_days_list)} 天")
print(f"  最多: {max(unique_days_list)} 天")
print(f"  平均: {sum(unique_days_list)/len(unique_days_list):.1f} 天")
print(f"\n单日最多照片数:")
print(f"  最少: {min(max_per_day_list)} 张")
print(f"  最多: {max(max_per_day_list)} 张")
print(f"  平均: {sum(max_per_day_list)/len(max_per_day_list):.1f} 张")

# 找出单日拍摄照片特别多的个体（可能导致时间泄漏风险）
high_risk = {k: v for k, v in qualified.items() if v['max_photos_per_day'] > 20}
print(f"\n单日拍摄 >20 张照片的个体（高风险）: {len(high_risk)} 个")
for identity_id, stats in sorted(high_risk.items(), key=lambda x: x[1]['max_photos_per_day'], reverse=True)[:10]:
    print(f"  {identity_id}: 最多一天 {stats['max_photos_per_day']} 张, 总 {stats['count']} 张, {stats['unique_days']} 天")

# 8. 输出详细列表
print("\n" + "=" * 80)
print("详细个体列表（照片 ≥ 15 且 时间跨度 ≥ 3 年）")
print("=" * 80)
print(f"{'个体ID':<10} {'照片数':<8} {'时间跨度':<10} {'拍摄天数':<10} {'单日最多':<10}")
print("-" * 60)
for identity_id, stats in sorted(qualified.items(), key=lambda x: x[1]['count'], reverse=True):
    print(f"{identity_id:<10} {stats['count']:<8} {stats['span_years']:.2f}年     {stats['unique_days']:<10} {stats['max_photos_per_day']:<10}")
