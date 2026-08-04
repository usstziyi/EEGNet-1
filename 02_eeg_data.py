"""
02_eeg_data.py - EEG 数据加载与预处理

学习目标：
- 使用 MNE-Python 加载和处理 EEG 数据
- 理解 braindecode 的 BaseConcatDataset 数据结构
- 掌握 EEG 数据的预处理流程（滤波、分段、标准化等）

braindecode 与 MNE-Python 紧密集成，利用 MNE 的数据结构进行预处理。
braindecode.datasets.MOABBDataset (包装器)
  ↓ 内部调用
moabb.datasets.* (具体数据集，如 BNCI2014001)
  ↓ 内部使用
mne (数据读取与处理)

你的代码
    │
    │  MOABBDataset("BNCI2014001", [1])
    ▼
braindecode.datasets.MOABBDataset
    │
    │  动态导入 moabb.datasets.BNCI2014001
    ▼
moabb.datasets.BNCI2014001.get_data(subjects=[1])
    │
    │  1. 检查本地缓存 → 没有就下载
    │  2. 读取 .gdf 文件
    ▼
mne.io.read_raw_gdf("~/mne_data/.../A01T.gdf", preload=True)
    │
    │  解析二进制 GDF 格式 → numpy 数组 + 元信息
    ▼
mne.io.Raw 对象
    │
    │  包装成 braindecode.datasets.BaseDataset
    ▼
braindecode.datasets.BaseConcatDataset
    │
    │  你拿到的 dataset 变量
    ▼
你的代码继续用 dataset.datasets[0].raw 访问
"""

import numpy as np
import mne
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_windows_from_events,
)

print("=" * 60)
print("EEG 数据加载与预处理")
print("=" * 60)

# ============================================================
# 1. 使用 MNE 创建模拟 EEG 数据
# ============================================================
print("\n1. 使用 MNE 创建模拟 EEG 数据")
print("-" * 40)

# 定义通道信息
sfreq = 250  # 采样率
ch_names = [f"EEG{i:03d}" for i in range(22)]  # 22 个 EEG 通道
ch_types = ["eeg"] * 22
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# 生成模拟数据（2 分钟）
duration_sec = 120
n_samples = int(duration_sec * sfreq)
data = np.random.randn(len(ch_names), n_samples) * 1e-6  # 微伏级别

# 创建 Raw 对象
raw = mne.io.RawArray(data, info)
print(f"Raw 对象: {raw}")
print(f"  - 通道数: {len(raw.ch_names)}")
print(f"  - 采样率: {raw.info['sfreq']} Hz")
print(f"  - 时长: {raw.times[-1]:.1f} 秒")

# ============================================================
# 2. 预处理流程
# ============================================================
print("\n2. 预处理流程")
print("-" * 40)

# 带通滤波 - EEG 常用频段: 0.5-40 Hz
raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=40.0)
print(f"带通滤波 (0.5-40 Hz): 完成")

# # 降采样 - 减少计算量
# raw_resampled = raw_filtered.copy().resample(sfreq=128)
# print(f"降采样至 128 Hz: {raw_resampled.info['sfreq']} Hz")

# 设置参考 - 使用平均参考
raw_reref = raw_filtered.copy().set_eeg_reference("average")
print(f"平均参考: 完成")

# ============================================================
# 3. 创建事件和 Epochs
# ============================================================
print("\n3. 创建事件和 Epochs")
print("-" * 40)

# 创建模拟事件（每 4 秒一个事件，共 2 类）
event_interval = int(4 * sfreq)  # 4 秒
events = []
for i in range(0, n_samples - event_interval, event_interval):
    event_id = 1 if i % (2 * event_interval) == 0 else 2
    events.append([i, 0, event_id])
events = np.array(events)  # shape (n_events, 3): 时间戳, 前状态持续时间, 事件类型
# 第 1 列 = 事件发生的采样点位置；
# 第 2 列 = 事件发生前那一刻，stim 通道（刺激通道）的值（几乎总是 0，通常忽略）；
# 第 3 列 = 事件类别标签 ID。

print(f"事件数量: {len(events)}")
print(f"事件类型: {np.unique(events[:, 2])}") # 事件列

# 创建 Epochs
tmin, tmax = 0.0, 2.0  # 每个 epoch 从事件开始后 0-2 秒
epochs = mne.Epochs(
    raw_reref,  # 预处理后的MNE Raw对象（经带通滤波、降采样、平均参考处理），作为epoch提取的数据源
    events,  # 事件数组，形状为(n_events, 3)，每行格式[事件采样点, 前stim通道值, 事件类型ID]，用于指定epoch起点
    event_id={"left_hand": 1, "right_hand": 2},  # 事件ID映射字典：key必须是字符串(事件名称)，value是events数组第3列的事件ID
    tmin=tmin,  # epoch相对于事件起点的开始时间（秒），此处tmin=0.0s，即从事件发生瞬间开始截取
    tmax=tmax,  # epoch相对于事件起点的结束时间（秒），此处tmax=2.0s，即截取事件后2秒的时间窗口
    baseline=None,  # 基线校正的时间区间（相对于事件起点），设为None表示不进行基线校正
    preload=True,  # 是否将所有epoch数据预加载到内存中，True表示预加载，便于后续快速访问
    verbose=False,  # 控制MNE日志输出详细程度，False为静默模式，不输出冗余信息
)
print(f"Epochs 对象: {epochs}")
print(f"  - epoch 数量: {len(epochs)}")
print(f"  - 每个 epoch 形状: {epochs.get_data().shape[1:]}")

# ============================================================
# 4. braindecode 数据集结构
# ============================================================
print("\n4. braindecode 数据集结构")
print("-" * 40)

from braindecode.datasets import RawDataset, BaseConcatDataset

# 将 MNE Raw 转换为 braindecode RawDataset
# RawDataset 需要 raw 和 description (一个 pandas Series)
import pandas as pd

"""
Subject（被试 / 人）
  └── Session（会话 / 天）
        └── Run（轮次 / 单次采集文件）
              └── Trial（试次 / 单个刺激事件）
"""
description = pd.Series(
    {
        "subject": 0,       # 被试编号
        "session": "train", # 会话类型
        "run": 0            # 轮次编号(包含多个试次trial)
    }
)
# 创建 RawDataset
dataset1 = RawDataset(raw_reref, pd.Series({'subject': 0, 'session': 'train', 'run': 0}))
dataset2 = RawDataset(raw_reref, pd.Series({"subject": 1, "session": "train", "run": 0}))
# 拼接多个数据集，成为 BaseConcatDataset
concat_dataset = BaseConcatDataset([dataset1, dataset2])

print(f"RawDataset: {dataset1}")
print(f"  - 数据长度: {len(dataset1)}")
print(f"BaseConcatDataset: {concat_dataset}")
print(f"  - 包含 {len(concat_dataset.datasets)} 个子数据集")

print(concat_dataset.datasets[0].description)
print(concat_dataset.datasets[1].description)


"""
Subject（被试 / 人）
  └── Session（会话 / 天）
        └── Run（轮次 / 单次采集文件）
              └── Trial（试次 / 单个刺激事件）
"""


# ============================================================
# 5. 使用 MOABB 加载公开数据集
# ============================================================
print("\n5. 使用 MOABB 加载公开数据集")
print("-" * 40)

# MOABB (Mother of All BCI Benchmarks) 提供了 150+ 公开 BCI 数据集
# 这里演示如何加载 Schirrmeister2017 数据集（需联网下载）
# 注意：首次运行会下载数据，可能需要较长时间

try:
    # 加载 Schirrmeister2017 数据集的一个被试
    dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[1])
    print(f"MOABB 数据集: {dataset}")
    print(f"  - 子数据集数量: {len(dataset.datasets)}")
    print(f"  - 第一个数据集: {dataset.datasets[0]}")

    print(dataset.datasets[0].description)
    print(dataset.datasets[1].description)

    """
    Subject（被试 / 人）
    └── Session（会话 / 天）
            └── Run（轮次 / 单次采集文件）
                └── Trial（试次 / 单个刺激事件）
    """

except Exception as e:
    print(f"MOABB 数据集加载失败（可能无网络）: {e}")
    print("提示：可以使用模拟数据继续学习")


# ============================================================
# 6. 创建 WindowDataset（滑动窗口分段）
# ============================================================
print("\n6. 创建 WindowDataset（滑动窗口分段）")
print("-" * 40)

# braindecode 支持两种解码策略：
# 1. Trialwise Decoding: 每个 trial 一个标签
# 2. Cropped Decoding: 滑动窗口，多个标签

# 使用模拟数据创建 WindowDataset
from braindecode.preprocessing import create_windows_from_events

# 首先给 raw 添加 annotations 作为事件标记
onsets = np.arange(0, len(raw_reref.times) - 4 * sfreq, 4 * sfreq) / sfreq
durations = [0.0] * len(onsets) # durations（每个事件的持续时间）
descriptions_list = ["left" if i % 2 == 0 else "right" for i in range(len(onsets))]
annotations = mne.Annotations(onsets, durations, descriptions_list)
print(annotations.description[:5])
raw_annotated = raw_reref.copy().set_annotations(annotations)
# set_annotations 默认是 in-place 操作 —— 直接修改 raw_reref 本身，并返回 self

# 用 BaseConcatDataset 包裹（create_windows_from_events 要求传入 BaseConcatDataset）
raw_dataset_annotated = BaseConcatDataset([RawDataset(
    raw_annotated,
    pd.Series({"subject": 0, "session": "train", "run": 0}),
)])

# 创建连续窗口
# 当前代码用的是 Cropped Decoding （步长 < 窗口大小，产生重叠窗口），这正是 braindecode 推荐的做法
windows_dataset = create_windows_from_events(
    raw_dataset_annotated,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=1000,  # 4 秒 @ 250Hz，与事件间隔一致
    window_size_samples=500,  # 2 秒 @ 250Hz
    window_stride_samples=250,  # 1 秒步长
    preload=True,
)
print(f"WindowsDataset: {windows_dataset}")
print(f"  - 窗口数量: {len(windows_dataset)}")
if len(windows_dataset) > 0:
    # x:(n_channels, n_times)EEG 窗口数据
    # y:(n_classes,)分类标签
    # crop_inds:(3,)窗口的元信息索引
    #   crop_inds[0]: 该窗口在 trial 内的序号（第几个滑动窗口）
    #   crop_inds[1]: 窗口在原始 Raw 数据中的 起始采样点
    #   crop_inds[2]: 窗口在原始 Raw 数据中的 结束采样点
    for i, (X, y, crop_inds) in enumerate(windows_dataset):
        print(f"  - 第{i}个窗口的形状: X={X.shape}, y={y}, crop_inds={crop_inds}")


# ============================================================
# 7. 预处理管道
# ============================================================
print("\n7. 预处理管道")
print("-" * 40)

# braindecode 提供了 Preprocessor 类来构建预处理管道
def to_microvolts(x):
    return x * 1e6

# 显式告诉 braindecode 这些是字符串指令，不需要自动修正
preprocessors = [
    Preprocessor("filter", l_freq=4.0, h_freq=38.0, apply_on_array=False),
    Preprocessor("resample", sfreq=128, apply_on_array=False),
    Preprocessor(to_microvolts),
]

print("预处理管道:")
for i, p in enumerate(preprocessors):
    print(f"  {i+1}. {p}")

print("\n" + "=" * 60)
print("学习要点：")
print("1. MNE-Python 是 braindecode 的数据基础")
print("2. BaseConcatDataset 是 braindecode 的核心数据结构")
print("3. 预处理包括：滤波、降采样、参考、分段")
print("4. WindowDataset 支持滑动窗口分段策略")
print("5. MOABB 提供了大量公开 BCI 数据集")
print("=" * 60)
