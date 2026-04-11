"""
从训练日志提取 Epoch 和 Acc0 数据并绘图
"""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_FILE = r"E:\fuchuang\chimpanzee_arcface\11111.txt"
OUTPUT_FILE = r"E:\fuchuang\chimpanzee_arcface\results\training_curve.png"

# 解析日志
epochs = []
acc0_list = []
acc_list = []
loss_list = []

with open(LOG_FILE, 'r', encoding='utf-8') as f:
    content = f.read()

# 匹配 Epoch X: Loss=..., Acc=..., Acc0=..., LR=...
pattern = r'Epoch (\d+): Loss=([\d.]+), Acc=([\d.]+), Acc0=([\d.]+), LR=([\d.]+)'
matches = re.findall(pattern, content)

for match in matches:
    epoch, loss, acc, acc0, lr = match
    epochs.append(int(epoch))
    loss_list.append(float(loss))
    acc_list.append(float(acc))
    acc0_list.append(float(acc0))

print(f"解析到 {len(epochs)} 个 epoch 数据")
print(f"最终 Acc0: {acc0_list[-1]:.4f}, 最佳 Acc0: {max(acc0_list):.4f}")

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Acc0 曲线
ax1.plot(epochs, acc0_list, 'b-', linewidth=2, label='Accuracy0')
ax1.axhline(y=max(acc0_list), color='r', linestyle='--', alpha=0.7, 
            label=f'Best Acc0={max(acc0_list):.4f} (Epoch {epochs[acc0_list.index(max(acc0_list))]})')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy0', fontsize=12)
ax1.set_title('Training Curve: Accuracy0 vs Epoch', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Loss 曲线
ax2.plot(epochs, loss_list, 'g-', linewidth=2, label='Training Loss')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training Curve: Loss vs Epoch', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"图表保存到: {OUTPUT_FILE}")
