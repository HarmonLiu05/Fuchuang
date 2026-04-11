"""
分析图片质量问题
"""
import json
from collections import Counter, defaultdict
from datetime import datetime

ANNOTATIONS_PATH = r"E:\fuchuang\turtlehead-dataset\Turtel_dataset\annotations.json"

with open(ANNOTATIONS_PATH, 'r') as f:
    coco = json.load(f)

print("=" * 60)
print("图片质量分析")
print("=" * 60)

# 1. 分辨率一致性
resolutions = [(img['width'], img['height']) for img in coco['images']]
res_counts = Counter(resolutions)
print(f"\n1. 分辨率一致性")
print(f"   不同分辨率数量: {len(res_counts)} 种")
print(f"   主分辨率: {res_counts.most_common(1)[0]} (占 {res_counts.most_common(1)[0][1]/len(coco['images'])*100:.1f}%)")
print(f"   其他分辨率: {sum(c for r,c in res_counts.most_common()[1:])} 张")

# 2. 个体样本量均衡性
identity_counts = Counter(ann['identity'] for ann in coco['annotations'])
counts = list(identity_counts.values())
print(f"\n2. 个体样本量")
print(f"   最少: {min(counts)} 张")
print(f"   最多: {max(counts)} 张")
print(f"   平均: {sum(counts)/len(counts):.1f} 张")
print(f"   标准差: {(sum((x-sum(counts)/len(counts))**2 for x in counts)/len(counts))**0.5:.1f}")
print(f"   样本 <10 张的个体: {sum(1 for c in counts if c < 10)} 个")
print(f"   样本 >100 张的个体: {sum(1 for c in counts if c > 100)} 个")

# 3. 时间分布
dates = []
for img in coco['images']:
    if img.get('date'):
        try:
            dates.append(datetime.strptime(img['date'], '%Y:%m:%d %H:%M:%S'))
        except:
            pass

if dates:
    print(f"\n3. 时间跨度")
    print(f"   最早: {min(dates).strftime('%Y-%m-%d')}")
    print(f"   最晚: {max(dates).strftime('%Y-%m-%d')}")
    print(f"   总跨度: {(max(dates)-min(dates)).days/365.25:.1f} 年")
    
    # 按年统计
    year_counts = Counter(d.year for d in dates)
    print(f"\n   按年分布:")
    for year, count in sorted(year_counts.items()):
        print(f"     {year}: {count} 张")

# 4. 单日拍摄密度
day_counts = Counter(d.strftime('%Y-%m-%d') for d in dates)
high_density_days = {day: count for day, count in day_counts.items() if count > 20}
print(f"\n4. 单日拍摄密度")
print(f"   单日 >20 张的天数: {len(high_density_days)} 天")
print(f"   单日最多: {max(day_counts.values())} 张")
if high_density_days:
    print(f"   高密度日期示例:")
    for day, count in sorted(high_density_days.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"     {day}: {count} 张")

# 5. 个体时间覆盖
identity_dates = defaultdict(list)
for img, ann in zip(coco['images'], coco['annotations']):
    if img.get('date'):
        try:
            dt = datetime.strptime(img['date'], '%Y:%m:%d %H:%M:%S')
            identity_dates[ann['identity']].append(dt)
        except:
            pass

short_spans = []
long_spans = []
for identity, dts in identity_dates.items():
    if len(dts) >= 2:
        span = (max(dts) - min(dts)).days / 365.25
        if span < 1:
            short_spans.append((identity, span, len(dts)))
        elif span > 7:
            long_spans.append((identity, span, len(dts)))

print(f"\n5. 个体时间覆盖")
print(f"   跨度 <1 年的个体: {len(short_spans)} 个")
print(f"   跨度 >7 年的个体: {len(long_spans)} 个")

print(f"\n" + "=" * 60)
print("总结：主要质量问题")
print("=" * 60)
print("1. 分辨率不一致（5472x3648 占 88%，但有 10+ 种其他分辨率）")
print("2. 个体样本量严重不均衡（最少 1 张，最多 182 张）")
print("3. 161 个个体样本 <10 张，训练不足")
print("4. 241 个个体时间跨度 <1 年，无法评估跨时间泛化")
print("5. 单日高密度拍摄可能导致数据泄漏（同一天最多 X 张）")
