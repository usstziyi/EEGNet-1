"""
06_advanced_models.py - 其他模型 (ShallowConvNet, DeepConvNet, ATCNet)

学习目标：
- 了解 braindecode 中的其他经典模型
- 掌握 ShallowConvNet 和 DeepConvNet 的特点
- 理解不同模型的适用场景
- 学会模型选择和比较

braindecode 提供了 65+ 种模型，这里介绍几个经典的 EEG 解码模型。
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("其他经典 EEG 解码模型")
print("=" * 60)

# ============================================================
# 1. 模型概览
# ============================================================
print("\n1. 模型概览")
print("-" * 40)

print("""
braindecode 中的经典模型：

1. EEGNet (v1/v4)
   - 参数量: ~2,000-3,000
   - 特点: 轻量级，使用分离卷积
   - 适用: 实时 BCI，资源受限场景

2. ShallowConvNet
   - 参数量: ~6,000
   - 特点: 单层卷积，简单高效
   - 适用: 快速基线，小数据集

3. DeepConvNet
   - 参数量: ~130,000
   - 特点: 深层网络，多个卷积层
   - 适用: 大数据集，复杂任务

4. ATCNet
   - Attention Temporal Convolution
   - 中文：注意力时序卷积
   - 参数量: ~10,000
   - 特点: 结合注意力机制和 TCN
   - 适用: 需要时序建模的任务

5. TCN (Temporal Convolutional Network)
   - 参数量: 可变
   - 特点: 因果卷积，长序列建模
   - 适用: 长时间序列解码
""")

# ============================================================
# 2. ShallowConvNet
# ============================================================
print("\n2. ShallowConvNet")
print("-" * 40)

# ShallowConvNet （浅层 ConvNet）→ braindecode 命名为 ShallowFBCSPNet
from braindecode.models import ShallowFBCSPNet

# 创建模型
shallow_net = ShallowFBCSPNet(
    n_chans=22,          # 输入通道数（电极数），这里使用 22 通道 EEG
    n_outputs=4,         # 输出类别数，对应 4 种运动想象任务
    n_times=1000,        # 每个通道的时间采样点数，即时间序列长度
    final_conv_length="auto",  # 最终卷积层长度，"auto" 表示自动根据输入调整
    n_filters_time=40,   # 时间卷积滤波器数量，提取时间特征
    n_filters_spat=40,   # 空间卷积滤波器数量，提取空间特征
    pool_time_length=75, # 时间池化窗口长度，用于降采样
    pool_time_stride=15, # 时间池化步长，控制池化窗口的移动
)

print(f"ShallowConvNet:")
print(f"  参数量: {sum(p.numel() for p in shallow_net.parameters()):,}")

# 测试
dummy_input = torch.randn(8, 22, 1000)
output = shallow_net(dummy_input)
print(f"  输入: {dummy_input.shape}")
print(f"  输出: {output.shape}")

print("""
ShallowConvNet 架构：
- 时间卷积: (1, 25) -> 40 个滤波器
- 空间卷积: (C, 1) -> 40 个空间滤波器
- 平均池化: (1, 75), stride=15
- 分类: 平均 + 线性层

特点：
- 结构简单，只有一个卷积块
- 训练速度快
- 适合小数据集
""")

# ============================================================
# 3. DeepConvNet
# ============================================================
print("\n3. DeepConvNet")
print("-" * 40)

# DeepConvNet （深层 ConvNet）→ braindecode 命名为 Deep4Net
from braindecode.models import Deep4Net

# 创建模型
deep_net = Deep4Net(
    n_chans=22,          # 输入通道数（电极数），这里使用 22 通道 EEG
    n_outputs=4,         # 输出类别数，对应 4 种运动想象任务
    n_times=1000,        # 每个通道的时间采样点数，即时间序列长度
    stride_before_pool=False,  # 是否在池化前使用步长卷积进行降采样，False 表示使用正常卷积
)

print(f"DeepConvNet:")
print(f"  参数量: {sum(p.numel() for p in deep_net.parameters()):,}")

# 测试
output = deep_net(dummy_input)
print(f"  输入: {dummy_input.shape}")
print(f"  输出: {output.shape}")

print("""
DeepConvNet 架构：
- 4 个卷积块，每块包含：
  - Conv2d + BatchNorm + ELU + MaxPool + Dropout
- 滤波器数量: 25 -> 50 -> 100 -> 200
- 卷积核大小: 10 -> 3 -> 3 -> 3

特点：
- 深层网络，特征提取能力强
- 参数量较大
- 适合大数据集
""")

# ============================================================
# 4. ATCNet (Attention Temporal Convolutional Network)
# ============================================================
print("\n4. ATCNet")
print("-" * 40)

# Attention Temporal Convolution
# 中文：注意力时序卷积
# ATCNet （Attention Temporal Convolutional Network）是结合了 注意力机制 和 时序卷积网络（TCN） 的 EEG 解码模型。
# 总参数量约 186,404。
# ATCNet 的核心创新在于 
   # 用 Attention 做特征选择、
   # 用 TCN 做时序建模、
   # 用多窗口捕捉不同时间尺度的信息 ，三者组合实现了强大的 EEG 解码能力。

from braindecode.models import ATCNet

# 创建模型
atcnet = ATCNet(
    n_chans=22,            # 输入通道数（电极数），这里使用 22 通道 EEG
    n_outputs=4,           # 输出类别数，对应 4 种运动想象任务
    n_times=1000,          # 每个通道的时间采样点数，即时间序列长度
    n_windows=2,           # 时间窗口数量，用于多窗口处理时序特征
    num_heads=2,           # 多头注意力机制的头数，决定注意力的并行粒度
    conv_block_n_filters=30,  # 卷积块中滤波器的数量，控制特征提取的丰富度
)

print(f"ATCNet:")
print(f"  参数量: {sum(p.numel() for p in atcnet.parameters()):,}")

# 测试
output = atcnet(dummy_input)
print(f"  输入: {dummy_input.shape}")
print(f"  输出: {output.shape}")

print("""
ATCNet 架构：
- 结合注意力机制和 TCN
- 多窗口处理
- 自注意力提取关键时序特征

特点：
- 中等参数量
- 时序建模能力强
- 适合需要长程依赖的任务
""")

# ============================================================
# 5. TCN (Temporal Convolutional Network)
# ============================================================
print("\n5. TCN")
print("-" * 40)

# braindecode 中的 TCN 是基于 因果膨胀卷积 的时序建模模型，总参数量约 56,354。
# TCN 的两大核心特点：
# 1. 因果卷积（Causal Convolution） ：只看过去，不看未来
# 2. 膨胀卷积（Dilated Convolution） ：指数级扩大感受野
from braindecode.models import TCN

# 创建模型
tcn = TCN(
    n_chans=22,
    n_outputs=4,
    n_filters=50,
    kernel_size=4,
    drop_prob=0.2,
)

print(f"TCN:")
print(f"  参数量: {sum(p.numel() for p in tcn.parameters()):,}")

# 测试
output = tcn(dummy_input)
print(f"  输入: {dummy_input.shape}")
print(f"  输出: {output.shape}")

print("""
TCN 架构：
- 因果卷积（Causal Convolution）
- 膨胀卷积（Dilated Convolution）
- 残差连接

特点：
- 保持因果性，适合时序预测
- 感受野大，能捕捉长程依赖
- 训练稳定
""")

# ============================================================
# 6. 模型比较
# ============================================================
print("\n6. 模型比较")
print("-" * 40)

from braindecode.models import EEGNet

models = {
    "EEGNet": EEGNet(n_chans=22, n_times=1000, n_outputs=4),
    "ShallowConvNet": ShallowFBCSPNet(n_chans=22, n_times=1000, n_outputs=4),
    "DeepConvNet": Deep4Net(n_chans=22, n_times=1000, n_outputs=4),
    "ATCNet": ATCNet(n_chans=22, n_times=1000, n_outputs=4),
    "TCN": TCN(n_chans=22, n_outputs=4),
}

print(f"{'模型':<20} {'参数量':>12} {'推理时间 (ms)':>15}")
print("-" * 50)

for name, model in models.items():
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    # 测量推理时间
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(dummy_input)

        # 计时
        import time

        start = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        elapsed = (time.time() - start) / 100 * 1000  # 毫秒

    print(f"{name:<20} {n_params:>12,} {elapsed:>15.2f}")

# ============================================================
# 7. 模型选择指南
# ============================================================
print("\n7. 模型选择指南")
print("-" * 40)

print("""
模型选择建议：

1. 数据量小（< 1000 trials）
   - 推荐: EEGNet, ShallowConvNet
   - 原因: 参数量小，不易过拟合

2. 数据量大（> 5000 trials）
   - 推荐: DeepConvNet, ATCNet
   - 原因: 特征提取能力强

3. 实时应用
   - 推荐: EEGNet, ShallowConvNet
   - 原因: 推理速度快

4. 长序列解码
   - 推荐: TCN, ATCNet
   - 原因: 时序建模能力强

5. 基线模型
   - 推荐: EEGNet
   - 原因: 广泛使用，便于比较

6. 跨被试迁移
   - 推荐: EEGNet, ATCNet
   - 原因: 泛化能力较好
""")

# ============================================================
# 8. 模型集成
# ============================================================
print("\n8. 模型集成")
print("-" * 40)


class EnsembleModel(nn.Module):
    """简单的模型集成"""

    def __init__(self, models):
      super().__init__()
      # 将传入的模型列表包装成 PyTorch 的 ModuleList，
      # 这样所有子模型的参数都会自动注册到父模块中，
      # 便于统一管理、保存和加载模型权重
      self.models = nn.ModuleList(models)

    def forward(self, x):
      # 对每个模型的输出求平均
      outputs = [model(x) for model in self.models]
      return torch.stack(outputs).mean(dim=0)


# 创建集成模型
ensemble = EnsembleModel(
    [
      EEGNet(n_chans=22, n_times=1000, n_outputs=4),
      ShallowFBCSPNet(n_chans=22, n_times=1000, n_outputs=4),
    ]
)

print(f"集成模型:")
print(f"  包含 {len(ensemble.models)} 个子模型")
print(f"  总参数量: {sum(p.numel() for p in ensemble.parameters()):,}")

# 测试
output = ensemble(dummy_input)
print(f"  输入: {dummy_input.shape}")
print(f"  输出: {output.shape}")

print("""
模型集成方法：
1. 平均集成: 对多个模型的输出求平均
2. 投票集成: 选择多数模型的预测
3. 堆叠集成: 使用元模型学习如何组合

优点：
- 提高泛化能力
- 降低方差
- 更稳定的预测

缺点：
- 推理时间增加
- 模型复杂度提高
""")

# ============================================================
# 9. 迁移学习
# ============================================================
print("\n9. 迁移学习")
print("-" * 40)

print("""
迁移学习在 EEG 解码中的应用：

1. 跨被试迁移
   - 在源被试上预训练
   - 在目标被试上微调

2. 跨任务迁移
   - 在源任务上预训练
   - 在目标任务上微调

3. 跨数据集迁移
   - 在源数据集上预训练
   - 在目标数据集上微调

示例代码：
""")

# 示例：跨被试迁移学习
print("# 1. 在源被试上预训练")
print("source_model = EEGNet(n_chans=22, n_times=1000, n_outputs=4)")
print("# ... 训练源模型 ...")
print("")
print("# 2. 冻结部分层")
print("for name, param in source_model.named_parameters():")
print("    if 'classifier' not in name:  # 冻结除分类头外的所有层")
print("        param.requires_grad = False")
print("")
print("# 3. 修改分类头（如果类别数不同）")
print("source_model.classifier = nn.Linear(in_features, new_n_classes)")
print("")
print("# 4. 在目标被试上微调")
print("optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, source_model.parameters()), lr=1e-4)")

print("\n" + "=" * 60)
print("学习要点：")
print("1. braindecode 提供了 65+ 种模型")
print("2. EEGNet 适合实时应用，DeepConvNet 适合大数据集")
print("3. 模型选择取决于数据量、任务需求和计算资源")
print("4. 模型集成可以提高性能")
print("5. 迁移学习可以有效利用有限数据")
print("=" * 60)
