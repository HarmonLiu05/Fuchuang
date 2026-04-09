# 🦍 大猩猩个体识别系统 (Chimpanzee ArcFace)

基于 ResNet50 + ArcFace 的大猩猩面部个体识别模型，支持本地 GPU 训练和评估。

## 功能

- ✅ **个体识别**：输入大猩猩面部图像，输出个体 ID
- ✅ **混合精度训练**：支持 FP16，节省显存
- ✅ **分阶段训练**：先冻结 backbone，后微调整体
- ✅ **数据增强**：翻转、仿射、颜色抖动、高斯模糊、随机擦除
- ✅ **评估指标**：Accuracy、TAR@FAR、Rank-1
- ✅ **开集/闭集识别**：支持未知个体检测

## 项目结构

```
chimpanzee_arcface/
├── configs/                    # 配置文件
│   ├── config_local.yaml      # 本地训练配置
│   ├── config_autodl.yaml     # AutoDL 配置
│   └── test_config.yaml       # 测试配置
├── data/
│   ├── dataset.py             # ChimpanzeeDataset 类
│   └── prepare_data.py        # 数据预处理和划分
├── models/
│   ├── backbone.py            # ResNet50 backbone
│   ├── bottleneck.py          # 512 维瓶颈层
│   └── arcface.py             # ArcFace Head
├── utils/
│   ├── metrics.py             # 评估指标计算
│   └── utils.py               # 工具函数
├── tests/
│   ├── test_dataset.py        # 数据集测试
│   ├── test_models.py         # 模型测试
│   └── test_metrics.py        # 评估指标测试
├── train.py                   # 训练脚本
├── evaluate.py                # 评估脚本
├── inference.py               # 推理脚本
├── requirements.txt           # 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

下载 [Chimpanzee Faces 数据集](https://github.com/KordingLab/PrimateFace)，放到项目上级目录：

```
fuchuang/
├── chimpanzee_faces/           # 数据集
│   └── datasets_cropped_chimpanzee_faces/
│       └── data_CZoo/
└── chimpanzee_arcface/         # 本项目
```

### 3. 训练模型

```bash
python train.py --config configs/config_local.yaml
```

### 4. 评估模型

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

### 5. 推理识别

```bash
# 闭集识别 (已知个体)
python inference.py --checkpoint checkpoints/best_model.pth --image chimp.jpg

# 开集识别 (未知个体输出 "Unknown")
python inference.py --checkpoint checkpoints/best_model.pth --image chimp.jpg --open_set --threshold 0.5
```

## 模型架构

```
输入图像 (112×112×3)
    ↓
ResNet50 Backbone (ImageNet 预训练)
    ↓
Bottleneck: Linear(2048→512) + BatchNorm + Dropout(0.4) + L2 归一化
    ↓
ArcFace Head: s=30.0, m=0.35, 24 分类
    ↓
输出个体 ID
```

## 训练策略

| 阶段 | Epoch | 训练内容 | 学习率 |
|------|-------|---------|--------|
| 阶段 1 | 1-10 | backbone.layer4 + bottleneck + arcface | backbone: 1e-4, 其他: 1e-3 |
| 阶段 2 | 11-30 | 全部参数微调 | 余弦衰减 |

## 评估结果 (24 个个体)

| 指标 | 结果 | 说明 |
|------|------|------|
| **Accuracy (Rank-1)** | **91.7%** | 预测最像的个体，猜对的概率 |
| **TAR@FAR=0.1%** | **62.0%** | 误识率 0.1% 时的真阳性率 |
| **Threshold** | **0.59** | 判定为同个体的相似度阈值 |

## 数据集

- **名称**: Chimpanzee Faces (C-Zoo)
- **来源**: [PrimateFace](https://github.com/KordingLab/PrimateFace)
- **个体数**: 24 个
- **图片数**: 2109 张
- **划分**: 训练集 1677 张，验证集 432 张

## 依赖

```
torch>=2.0.0
torchvision>=0.15.0
pyyaml>=6.0
numpy>=1.24.0
Pillow>=9.0.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
faiss-cpu>=1.7.0
```

## License

MIT
