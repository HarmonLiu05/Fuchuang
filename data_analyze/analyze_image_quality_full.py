"""
图片质量分析脚本
分析：光照、模糊、Position 分布、分辨率等
输出：analysis_report.json + dataset_analysis.png
"""
import cv2
import numpy as np
import os
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt

# ========== 配置 ==========
ANNOTATIONS_PATH = r"E:\fuchuang\turtlehead-dataset\Turtel_dataset\annotations.json"
IMAGES_ROOT = r"E:\fuchuang\turtlehead-dataset\Turtel_dataset"
OUTPUT_REPORT = r"E:\fuchuang\chimpanzee_arcface\results\analysis_report.json"
OUTPUT_CHART = r"E:\fuchuang\chimpanzee_arcface\results\dataset_analysis.png"

# 抽样分析（加速）
SAMPLE_SIZE = 2000  # 只分析 2000 张随机抽样

os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)

# ========== 加载 COCO 数据 ==========
print("加载标注文件...")
with open(ANNOTATIONS_PATH, 'r') as f:
    coco = json.load(f)

# 构建映射
image_id_to_ann = {ann['image_id']: ann for ann in coco['annotations']}
image_id_to_img = {img['id']: img for img in coco['images']}

print(f"共 {len(coco['images'])} 张图片")

# 随机抽样
import random
if len(coco['images']) > SAMPLE_SIZE:
    sampled_images = random.sample(coco['images'], SAMPLE_SIZE)
    print(f"随机抽样 {SAMPLE_SIZE} 张进行分析")
else:
    sampled_images = coco['images']

# ========== 分析函数 ==========
def analyze_lighting(img_path):
    """分析光照"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    return {
        "mean_brightness": float(v_channel.mean()),
        "std_brightness": float(v_channel.std()),
        "is_overexposed": bool(v_channel.mean() > 220),
        "is_underexposed": bool(v_channel.mean() < 30),
        "contrast": float(l_channel.std()),
    }


def analyze_sharpness(img_path):
    """分析模糊度"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return {
        "laplacian_var": float(lap_var),
        "is_blurry": bool(lap_var < 50)  # 降低阈值到 50
    }


# ========== 主分析流程 ==========
results = []
processed_count = 0
error_count = 0

# Position 统计
position_counter = defaultdict(int)
identity_positions = defaultdict(list)

print("\n开始分析图片质量...")

for img_info in sampled_images:
    img_id = img_info['id']
    ann = image_id_to_ann.get(img_id)
    if not ann:
        continue
    
    identity = ann.get('identity', 'unknown')
    position = ann.get('position', 'unknown')
    position_counter[position] += 1
    identity_positions[identity].append(position)
    
    # 构建图片路径 - 使用 path 字段（处理后的图片）
    img_path = os.path.join(IMAGES_ROOT, img_info.get('path', ''))
    
    if not os.path.exists(img_path):
        error_count += 1
        continue
    
    lighting = analyze_lighting(img_path)
    sharpness = analyze_sharpness(img_path)
    
    if lighting and sharpness:
        results.append({
            "image_id": img_id,
            "identity": identity,
            "position": position,
            "resolution": f"{img_info['width']}x{img_info['height']}",
            **lighting,
            **sharpness
        })
        processed_count += 1
    
    if processed_count % 500 == 0:
        print(f"  已分析 {processed_count} 张...")

print(f"\n分析完成: {processed_count} 张成功, {error_count} 张失败")

# ========== 保存报告 ==========
with open(OUTPUT_REPORT, 'w') as f:
    json.dump(results, f, indent=2)
print(f"报告保存到: {OUTPUT_REPORT}")

# ========== 可视化 ==========
print("\n生成可视化图表...")

df = results  # 直接用字典列表
import pandas as pd
df = pd.DataFrame(df)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 亮度分布
axes[0,0].hist(df["mean_brightness"], bins=50, color='skyblue', edgecolor='black')
axes[0,0].axvline(x=220, color='r', linestyle='--', label='Overexposed')
axes[0,0].axvline(x=30, color='g', linestyle='--', label='Underexposed')
axes[0,0].set_title("Brightness Distribution")
axes[0,0].set_xlabel("Mean Brightness")
axes[0,0].legend()

# 2. 个体亮度方差（前20个变化最大的个体）
id_brightness_std = df.groupby("identity")["mean_brightness"].std().sort_values(ascending=False).head(20)
axes[0,1].barh(range(len(id_brightness_std)), id_brightness_std.values, color='salmon')
axes[0,1].set_yticks(range(len(id_brightness_std)))
axes[0,1].set_yticklabels(id_brightness_std.index)
axes[0,1].set_title("Top 20 Individuals with Brightness Variation")
axes[0,1].set_xlabel("Brightness Std")

# 3. 清晰度分布
axes[0,2].hist(df["laplacian_var"], bins=50, color='lightgreen', edgecolor='black')
axes[0,2].axvline(x=50, color='r', linestyle='--', label='模糊阈值')
axes[0,2].set_title("Sharpness Distribution (Laplacian Variance)")
axes[0,2].set_xlabel("Laplacian Variance")
axes[0,2].legend()

# 4. 曝光问题统计
overexp = df["is_overexposed"].sum()
underexp = df["is_underexposed"].sum()
normal = len(df) - overexp - underexp
axes[1,0].bar(["Normal", "Overexposed", "Underexposed"], [normal, overexp, underexp], 
              color=['lightgreen', 'orange', 'red'])
axes[1,0].set_title("Exposure Quality")

# 5. 模糊图片按个体分布
blurry_by_id = df[df["is_blurry"]].groupby("identity").size().sort_values(ascending=False).head(15)
if len(blurry_by_id) > 0:
    axes[1,1].barh(range(len(blurry_by_id)), blurry_by_id.values, color='coral')
    axes[1,1].set_yticks(range(len(blurry_by_id)))
    axes[1,1].set_yticklabels(blurry_by_id.index)
    axes[1,1].set_title("Top 15 Individuals with Blurry Images")
else:
    axes[1,1].text(0.5, 0.5, "No blurry images", ha='center', va='center')

# 6. Position 分布
pos_counts = df["position"].value_counts()
axes[1,2].pie(pos_counts.values, labels=pos_counts.index, autopct='%1.1f%%')
axes[1,2].set_title("Position Distribution")

plt.tight_layout()
plt.savefig(OUTPUT_CHART, dpi=150, bbox_inches='tight')
print(f"图表保存到: {OUTPUT_CHART}")

# ========== 打印摘要 ==========
print("\n" + "=" * 60)
print("图片质量分析摘要")
print("=" * 60)
print(f"总分析图片数: {len(df)}")
print(f"\n亮度: 平均={df['mean_brightness'].mean():.1f}, 标准差={df['mean_brightness'].std():.1f}")
print(f"过曝图片: {df['is_overexposed'].sum()} 张 ({df['is_overexposed'].mean()*100:.1f}%)")
print(f"欠曝图片: {df['is_underexposed'].sum()} 张 ({df['is_underexposed'].mean()*100:.1f}%)")
print(f"\n清晰度: 平均={df['laplacian_var'].mean():.1f}, 中位数={df['laplacian_var'].median():.1f}")
print(f"模糊图片: {df['is_blurry'].sum()} 张 ({df['is_blurry'].mean()*100:.1f}%)")
print(f"\nPosition 分布:")
for pos, count in pos_counts.items():
    print(f"  {pos}: {count} 张 ({count/len(df)*100:.1f}%)")
