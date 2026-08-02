"""
03_dataset_window.py - WindowDataset 与数据管道

学习目标：
- 深入理解 braindecode 的 WindowDataset
- 掌握 Trialwise 和 Cropped 两种解码策略
- 构建完整的 DataLoader 数据管道
- 理解数据增强的基本方法

braindecode 的数据管道设计遵循 PyTorch 的 Dataset/DataLoader 模式，
同时针对 EEG 数据的特性进行了优化。
"""

import numpy as np
import mne
import torch
from torch.utils.data import DataLoader
import pandas as pd

from braindecode.datasets import RawDataset, BaseConcatDataset
from braindecode.preprocessing import (
    # 从 events（即 MNE annotations/事件标记）创建窗口，而非从 epochs 创建
    create_windows_from_events,
    Preprocessor,  # 预处理器类，用于定义具体的预处理操作（如滤波、重采样等）
    preprocess,    # 预处理执行函数，将预处理器列表批量应用到数据集上
)

print("=" * 60)
print("WindowDataset 与数据管道")
print("=" * 60)

# ============================================================
# 1. 创建模拟 EEG 数据
# ============================================================
print("\n1. 创建模拟 EEG 数据")
print("-" * 40)

sfreq = 250
n_channels = 22
ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)

# 生成 5 分钟的模拟数据
duration = 300  # 秒
n_samples = int(duration * sfreq)
data = np.random.randn(n_channels, n_samples) * 1e-6
raw = mne.io.RawArray(data, info, verbose=False)

# 添加事件标记（每 4 秒一个，4 类）
event_interval = 4.0  # 秒
n_events = int(duration / event_interval)
onsets = np.arange(n_events) * event_interval
durations = [0.0] * n_events
# 4 类运动想象: left hand, right hand, feet, rest
descriptions_list = [1, 2, 3, 4] * (n_events // 4) + [1, 2, 3, 4][:n_events % 4]
annotations = mne.Annotations(onsets, durations, [str(d) for d in descriptions_list])
raw = raw.set_annotations(annotations)

print(f"Raw 数据: {len(raw.ch_names)} 通道, {raw.times[-1]:.1f} 秒")
print(f"事件数量: {len(onsets)}, 类别: {set(descriptions_list)}")

# ============================================================
# 2. 预处理
# ============================================================
print("\n2. 预处理")
print("-" * 40)
"""
Braindecode 内部的判断逻辑是：
- fn 是 可调用对象 （函数）→ apply_on_array 才有意义， True 走数组路径， False 走对象路径
- fn 是 字符串 → 走 getattr(raw, fn) 路径，本质上就是在 raw 对象上调用方法，等价于 apply_on_array=False 的行为
所以当你传 apply_on_array=True 配合字符串 fn 时，braindecode 认为这是一个"不匹配"的组合，自动把它修正为 False 并提醒你。
"""
preprocessors_list = [
    Preprocessor("filter", l_freq=4.0, h_freq=38.0, verbose=False,apply_on_array=False),
    Preprocessor("resample", sfreq=128, verbose=False,apply_on_array=False),
]

# 创建 RawDataset，再包装进 BaseConcatDataset
raw_dataset = RawDataset(raw, pd.Series({"subject": 0, "session": "train"}))
dataset = BaseConcatDataset([raw_dataset])
# preprocess() 会直接修改数据集内部的 raw 对象
preprocess(dataset, preprocessors_list)
"""
preprocess() 之后的 dataset 仍然是 BaseConcatDataset ，它不是 PyTorch 风格的 Dataset。 
BaseConcatDataset 只是一个原始数据的容器，内部存的还是 MNE 的 Raw 对象，不支持 __getitem__ 和 __len__ 的 PyTorch 协议。
要变成 PyTorch 兼容的 Dataset，需要通过 create_windows_from_events() 将连续的 raw 数据切分成固定大小的窗口，
生成 WindowsDataset ——这才是实现了 PyTorch Dataset 接口的对象。
整个流程是：
原始 raw 数据
    ↓
BaseConcatDataset (数据容器，非 PyTorch Dataset)
    ↓ preprocess()
BaseConcatDataset (预处理后，仍是容器)
    ↓ create_windows_from_events()
WindowsDataset (PyTorch Dataset ✅)
    ↓ DataLoader
可以直接喂给模型训练
"""
print(f"预处理完成，采样率: {dataset.datasets[0].raw.info['sfreq']} Hz")
"""
dataset.datasets[0].raw.info['sfreq']
   │       │      │   │     │
   │       │      │   │     └── 采样率 (128 Hz)
   │       │      │   └── MNE 的 Info 对象
   │       │      └── RawDataset 内部持有的 MNE Raw 对象
   │       └── 容器中第 1 个子数据集（即 raw_dataset）
   └── BaseConcatDataset（数据容器）
"""

# ============================================================
# 3. Trialwise Decoding（试次级解码）
# ============================================================
print("\n3. Trialwise Decoding（试次级解码）")
print("-" * 40)

# Trialwise: 每个 trial 提取一个固定长度的窗口，产生一个预测
# trial_stop_offset_samples=512: trial 持续 4 秒 (512 samples @ 128Hz)
windows_trialwise = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=512,
    window_size_samples=512,  # 4 秒 @ 128Hz
    window_stride_samples=512,  # 无重叠
    preload=True,
)

print(f"Trialwise 窗口数: {len(windows_trialwise)}")
if len(windows_trialwise) > 0:
    X, y, idx = windows_trialwise[0]
    print(f"  单个样本: X.shape={X.shape}, y={y}")
    print(f"  X 含义: (channels, time) = ({X.shape[0]}, {X.shape[1]})")

# ============================================================
# 4. Cropped Decoding（裁剪解码）
# ============================================================
print("\n4. Cropped Decoding（裁剪解码）")
print("-" * 40)

# Cropped: 使用滑动窗口，一个 trial 产生多个重叠的窗口
# 每个窗口都有独立的标签，训练时产生更多样本
# trial_stop_offset_samples=512: trial 持续 4 秒，滑动窗口可产生多个样本
windows_cropped = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=512,
    window_size_samples=256,  # 2 秒窗口
    window_stride_samples=64,  # 0.5 秒步长（75% 重叠）
    preload=True,
)

print(f"Cropped 窗口数: {len(windows_cropped)}")
if len(windows_cropped) > 0:
    X, y, idx = windows_cropped[0]
    print(f"  单个样本: X.shape={X.shape}, y={y}")
    print(f"  窗口数比 Trialwise 多: {len(windows_cropped) / max(len(windows_trialwise), 1):.1f} 倍")

# ============================================================
# 5. 构建 DataLoader
# ============================================================
print("\n5. 构建 DataLoader")
print("-" * 40)

# 选择一种窗口策略
windows_dataset = windows_trialwise if len(windows_trialwise) > 0 else windows_cropped

# 划分训练集和验证集
n_total = len(windows_dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train

train_dataset = torch.utils.data.Subset(windows_dataset, range(n_train))
val_dataset = torch.utils.data.Subset(windows_dataset, range(n_train, n_total))

print(f"总样本数: {n_total}")
print(f"训练集: {n_train}, 验证集: {n_val}")

# 创建 DataLoader
batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # macOS 上建议用 0
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
)

print(f"\nTrainLoader: {len(train_loader)} 批次")
print(f"ValLoader: {len(val_loader)} 批次")

# 检查一个批次的数据
for X_batch, y_batch, idx_batch in train_loader:
    print(f"\n批次数据:")
    print(f"  X: {X_batch.shape} (batch, channels, time)")
    print(f"  y: {y_batch.shape} (batch,)")
    print(f"  类别分布: {torch.bincount(y_batch.long())}")
    break

# ============================================================
# 6. 数据标准化
# ============================================================
print("\n6. 数据标准化")
print("-" * 40)

# EEG 数据标准化很重要，常用方法：
# 1. Z-score 标准化
# 2. Min-Max 归一化
# 3. 按通道标准化

# 方法 1: 全局 Z-score
X_sample = X_batch.float()
mean = X_sample.mean()
std = X_sample.std()
X_normalized = (X_sample - mean) / (std + 1e-8)
print(f"全局 Z-score: mean={X_normalized.mean():.4f}, std={X_normalized.std():.4f}")

# 方法 2: 按通道标准化
X_channel_norm = torch.zeros_like(X_sample)
for c in range(X_sample.shape[1]):
    ch_mean = X_sample[:, c, :].mean()
    ch_std = X_sample[:, c, :].std()
    X_channel_norm[:, c, :] = (X_sample[:, c, :] - ch_mean) / (ch_std + 1e-8)
print(f"按通道标准化: mean={X_channel_norm.mean():.4f}, std={X_channel_norm.std():.4f}")

# ============================================================
# 7. 数据增强
# ============================================================
print("\n7. 数据增强")
print("-" * 40)

# braindecode 提供了多种 EEG 特定的数据增强方法
from braindecode.augmentation import (
    FrequencyShift,
    ChannelsDropout,
    TimeReverse,
    SignFlip,
)
from braindecode.augmentation import Transform

# 创建增强管道
transforms = [
    FrequencyShift(probability=0.5, sfreq=128, max_delta_freq=2.0),
    ChannelsDropout(probability=0.3, p_drop=0.1),
    SignFlip(probability=0.5),
]

# 应用增强
X_augmented = X_sample.clone()
for transform in transforms:
    X_augmented = transform(X_augmented)

print(f"原始数据形状: {X_sample.shape}")
print(f"增强后数据形状: {X_augmented.shape}")
print(f"数据增强方法:")
for t in transforms:
    print(f"  - {t.__class__.__name__}(p={t.probability})")

# ============================================================
# 8. 完整的数据管道示例
# ============================================================
print("\n8. 完整的数据管道示例")
print("-" * 40)


def get_data_loaders(
    raw,
    batch_size=64,
    train_ratio=0.8,
    window_size=512,
    window_stride=512,
):
    """
    构建完整的数据管道

    参数:
        raw: MNE Raw 对象（带 annotations）
        batch_size: 批次大小
        train_ratio: 训练集比例
        window_size: 窗口大小（样本数）
        window_stride: 窗口步长（样本数）

    返回:
        train_loader, val_loader
    """
    # 1. 创建 RawDataset，再包装进 BaseConcatDataset
    raw_dataset = RawDataset(raw, pd.Series({"subject": 0, "session": "train"}))
    dataset = BaseConcatDataset([raw_dataset])

    # 2. 预处理
    preprocessors = [
        Preprocessor("filter", l_freq=4.0, h_freq=38.0, verbose=False),
        Preprocessor("resample", sfreq=128, verbose=False),
    ]
    preprocess(dataset, preprocessors)

    # 3. 创建窗口
    windows = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=512,
        window_size_samples=window_size,
        window_stride_samples=window_stride,
        preload=True,
    )

    # 4. 划分数据集
    n_total = len(windows)
    n_train = int(train_ratio * n_total)
    train_set = torch.utils.data.Subset(windows, range(n_train))
    val_set = torch.utils.data.Subset(windows, range(n_train, n_total))

    # 5. 创建 DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


# 使用示例
train_loader, val_loader = get_data_loaders(raw.copy(), batch_size=32)
print(f"训练集批次数: {len(train_loader)}")
print(f"验证集批次数: {len(val_loader)}")

print("\n" + "=" * 60)
print("学习要点：")
print("1. Trialwise: 每个 trial 一个窗口，适合长 trial")
print("2. Cropped: 滑动窗口，产生更多样本，适合短 trial")
print("3. DataLoader 负责批处理、打乱、多进程加载")
print("4. 标准化对模型收敛很重要")
print("5. 数据增强可以提高模型泛化能力")
print("=" * 60)
