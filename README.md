# EEGNet 学习计划 - braindecode PyTorch 教程

这是一个循序渐进的 braindecode (PyTorch EEGNet) 学习计划，通过 7 个 Python 文件从基础到实战完整掌握 EEG 解码技术。

## 学习路径

### 1. PyTorch 张量基础 (`01_tensor_basics.py`)
- PyTorch 张量创建和操作
- 形状变换（reshape, permute, unsqueeze）
- GPU 加速和自动求导
- EEG 数据的张量表示

### 2. EEG 数据加载与预处理 (`02_eeg_data.py`)
- MNE-Python 基础
- braindecode 数据结构（BaseDataset, BaseConcatDataset）
- 预处理流程（滤波、降采样、参考）
- MOABB 公开数据集加载

### 3. WindowDataset 与数据管道 (`03_dataset_window.py`)
- Trialwise vs Cropped 解码策略
- DataLoader 构建
- 数据标准化和增强
- 完整的数据管道示例

### 4. EEGNet 模型详解 (`04_eegnet_model.py`)
- EEGNet 架构原理
- 分离卷积（时间卷积 + 深度卷积 + 分离卷积）
- 手动实现 EEGNet（理解每一层）
- braindecode 的 EEGNet 实现
- 模型保存与加载

### 5. 训练循环与监控 (`05_training_loop.py`)
- 手动训练循环
- 损失函数、优化器、学习率调度器
- 早停机制
- 训练可视化
- braindecode 的 EEGClassifier
- 评估指标（准确率、精确率、召回率、F1）

### 6. 其他经典模型 (`06_advanced_models.py`)
- ShallowConvNet
- DeepConvNet
- ATCNet
- TCN
- 模型比较和选择指南
- 模型集成
- 迁移学习

### 7. 实战项目：运动想象 EEG 解码 (`07_project.py`)
- 使用 MOABB 加载 BCI Competition IV 2a 数据集
- 完整的端到端管道
- 交叉验证
- 多模型比较
- 结果可视化和统计分析

## 安装依赖

使用 uv 安装依赖：

```bash
uv sync
```

或手动安装：

```bash
pip install -e .
```

## 运行顺序

按照文件编号顺序学习：

```bash
python 01_tensor_basics.py
python 02_eeg_data.py
python 03_dataset_window.py
python 04_eegnet_model.py
python 05_training_loop.py
python 06_advanced_models.py
python 07_project.py
```

## 项目结构

```
EEGNet/
├── pyproject.toml              # 项目配置和依赖
├── 01_tensor_basics.py         # PyTorch 张量基础
├── 02_eeg_data.py              # EEG 数据加载与预处理
├── 03_dataset_window.py        # WindowDataset 与数据管道
├── 04_eegnet_model.py          # EEGNet 模型详解
├── 05_training_loop.py         # 训练循环与监控
├── 06_advanced_models.py       # 其他经典模型
├── 07_project.py               # 实战项目
└── README.md                   # 本文件
```

## 学习要点

1. **循序渐进**：从 PyTorch 基础开始，逐步深入到 EEG 解码
2. **理论结合实践**：每个概念都有对应的代码示例
3. **手动实现**：通过手动实现 EEGNet 深入理解架构
4. **实战项目**：完整的端到端项目，从数据加载到模型评估
5. **多模型比较**：了解不同模型的特点和适用场景

## 参考资源

- [braindecode 官方文档](https://braindecode.org/stable/)
- [braindecode GitHub](https://github.com/braindecode/braindecode)
- [MNE-Python 文档](https://mne.tools/stable/)
- [MOABB 文档](https://moabb.neurotechx.com/)
- [EEGNet 论文](https://arxiv.org/abs/1611.08024)

## 注意事项

- 首次运行 `07_project.py` 时会下载 BCI Competition IV 2a 数据集，可能需要较长时间
- 如果没有网络，代码会自动使用模拟数据进行演示
- 建议使用 GPU 加速训练（如果有）
- 每个文件都可以独立运行，方便复习特定主题

## 许可证

MIT License
