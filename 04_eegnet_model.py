"""
04_eegnet_model.py - EEGNet 模型详解

学习目标：
- 深入理解 EEGNet 的网络架构
- 掌握分离卷积（Separable Convolution）的原理
- 使用 braindecode 构建 EEGNet 模型
- 理解模型各层的作用和参数含义

EEGNet 是 BCI 领域最经典的轻量级模型之一，由 Lawhern et al. (2018) 提出。
它使用分离卷积来学习时间和空间特征，参数量小，适合实时应用。
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("EEGNet 模型详解")
print("=" * 60)

# ============================================================
# 1. EEGNet 架构概述
# ============================================================
print("\n1. EEGNet 架构概述")
print("-" * 40)

print("""
EEGNet 架构流程：

输入: (batch, 1, C, T)  [C=通道数, T=时间点]
  ↓
[Block1: 时间卷积]
  Conv2d(1, F1, (1, kernel_length)) + BatchNorm
  ↓
[Block1: 深度卷积（空间滤波）]
  DepthwiseConv2d(F1, F1*D, (C, 1)) + BatchNorm + ELU + AvgPool + Dropout
  ↓
[Block2: 分离卷积]
  SeparableConv2d(F1*D, F1*D, (1, kernel_length)) + BatchNorm + ELU + AvgPool + Dropout
  ↓
[分类头]
  Flatten + Linear(F1*D*final_time, n_classes) + LogSoftmax

关键参数：
- F1: 时间卷积的滤波器数量（通常 8）
- D: 深度乘数（通常 2）
- F2 = F1 * D: 分离卷积的输出通道数（通常 16）
- kernel_length: 时间卷积核大小（通常 125 @ 250Hz）
""")

# ============================================================
# 2. 手动实现 EEGNet（理解每一层）
# ============================================================
print("\n2. 手动实现 EEGNet（理解每一层）")
print("-" * 40)


class EEGNetManual(nn.Module):
    """
    手动实现的 EEGNet，用于理解每一层的作用
    """

    def __init__(
        self,
        n_channels=22,
        n_times=1000,
        n_classes=4,
        F1=8,
        D=2,
        F2=16,
        kernel_length=125,
        dropout=0.5,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: 时间卷积
        # 输入: (batch, 1, C, T)
        # 输出: (batch, F1, C, T)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        )
        # EEG 数据的幅度在不同被试、不同试次间差异很大（可能从微伏到毫伏级波动）。
        # BatchNorm 可以有效抵消这种 输入分布的变化 ，让模型专注于学习 时间和空间模式 ，而不是绝对幅度的数值。
        self.bn1 = nn.BatchNorm2d(F1) # shape不变

        # Block 1: 深度卷积（空间滤波）
        # 使用 depthwise conv，每个输入通道独立卷积
        # 输入: (batch, F1, C, T)
        # 输出: (batch, F1*D, 1, T)
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,  # depthwise convolution
            bias=False,
        )
        """
        为什么是 D=2 个核/每输入通道
        每个输入通道（时间特征）产生 2 个输出，相当于学习 2 种不同的空间滤波方式 。例如：

        - 核 A 可能学到"额区 + 中央区"的加权模式
        - 核 B 可能学到"顶区 + 枕区"的加权模式
        这样每个时间特征都能被映射到 2 种不同的空间投影上，提升表达能力。
        """
        self.bn2 = nn.BatchNorm2d(F1 * D) # shape不变
        self.activation1 = nn.ELU()  # 激活函数：引入非线性，帮助模型学习复杂模式
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4)) # (batch, F1*D, 1, T//4)
        self.dropout1 = nn.Dropout(dropout)

        # Block 2: 分离卷积
        # 结合时间卷积和空间卷积
        # 输入: (batch, F1*D, 1, T/4)
        # 输出: (batch, F2, 1, T/4)
        self.separable_conv = nn.Sequential(
            # 时间卷积（depthwise）
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, kernel_length // 4),
                padding=(0, (kernel_length // 4) // 2),
                groups=F1 * D,
                bias=False,
            ), # (batch, F1*D, 1, T/4)
            # 1x1 卷积（pointwise）：通道融合，降维/升维
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ), # (batch, F2, 1, T/4)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8)) # (batch, F2, 1, T/32)
        self.dropout2 = nn.Dropout(dropout) # （batch, F2, 1, T/32）

        # 计算分类头的输入维度
        self.final_time = self._get_final_time_dim()
        # n_classes 是模型需要区分的 类别数量 ，完全由你的 分类任务决定
        self.classifier = nn.Linear(F2 * self.final_time, n_classes)

    # 试运行获取最终时间维度，用于分类头
    def _get_final_time_dim(self):
        """计算经过池化后的时间维度"""
        x = torch.zeros(1, 1, self.n_channels, self.n_times)
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.pool1(x)
        x = self.separable_conv(x)
        x = self.pool2(x)
        return x.shape[3]

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量 (batch, 1, C, T) 或 (batch, C, T)

        返回:
            输出: (batch, n_classes)
        """
        # 确保输入是 4D
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, C, T) -> (batch, 1, C, T)

        # Block 1: 时间卷积
        x = self.conv1(x)
        x = self.bn1(x)

        # Block 1: 深度卷积
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2: 分离卷积
        x = self.separable_conv(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # 分类头
        # flatten(1) 将除 batch 以外的所有维度展平
        x = x.flatten(1)  # (batch, F2, 1, T_final) -> (batch, F2*T_final)
        x = self.classifier(x)

        return x


# 创建模型实例
model_manual = EEGNetManual(
    n_channels=22,
    n_times=1000,
    n_classes=4,
)

# 测试模型
dummy_input = torch.randn(8, 22, 1000)  # (batch, channels, time)
output = model_manual(dummy_input)

print(f"手动实现的 EEGNet:")
print(f"  输入形状: {dummy_input.shape}")
print(f"  输出形状: {output.shape}")
print(f"  参数量: {sum(p.numel() for p in model_manual.parameters()):,}")

# ============================================================
# 3. 使用 braindecode 的 EEGNet
# ============================================================
print("\n3. 使用 braindecode 的 EEGNet")
print("-" * 40)

from braindecode.models import EEGNet

# braindecode 提供了 EEGNet 的标准实现
model_braindecode = EEGNet(
    n_chans=22,
    n_times=1000,
    n_outputs=4,
    F1=8,
    F2=16,
    D=2,
    kernel_length=125,
    drop_prob=0.5,
)

print(f"braindecode 的 EEGNet:")
print(f"  模型类型: {type(model_braindecode).__name__}")
print(f"  参数量: {sum(p.numel() for p in model_braindecode.parameters()):,}")

# 测试
output_bd = model_braindecode(dummy_input)
print(f"  输入形状: {dummy_input.shape}")
print(f"  输出形状: {output_bd.shape}")

# ============================================================
# 4. 模型结构可视化
# ============================================================
print("\n4. 模型结构可视化")
print("-" * 40)

print("\n手动实现的 EEGNet 结构:")
print(model_manual)

# ============================================================
# 5. 关键组件详解
# ============================================================
print("\n5. 关键组件详解")
print("-" * 40)

print("""
(1) 时间卷积 (Temporal Convolution)
   - 作用: 学习时间特征（如 ERP 成分、节律）
   - 卷积核: (1, kernel_length)，只在时间维度卷积
   - kernel_length 通常设置为采样率的一半（如 250Hz -> 125 样本）

(2) 深度卷积 (Depthwise Convolution)
   - 作用: 空间滤波，学习通道间的空间模式
   - 卷积核: (C, 1)，只在空间维度卷积
   - groups=F1: 每个时间滤波器独立进行空间滤波
   - 类似于 CSP (Common Spatial Pattern) 的深度学习版本

(3) 分离卷积 (Separable Convolution)
   - 作用: 结合时间和空间特征的二次学习
   - 由 depthwise + pointwise 卷积组成
   - 参数量比标准卷积少很多

(4) 批归一化 (Batch Normalization)
   - 作用: 加速训练，稳定梯度
   - 位置: 每个卷积层之后

(5) 平均池化 (Average Pooling)
   - 作用: 降采样，减少计算量
   - 位置: 每个 block 的末尾

(6) Dropout
   - 作用: 防止过拟合
   - 通常设置为 0.5
""")

# ============================================================
# 6. 参数量分析
# ============================================================
print("\n6. 参数量分析")
print("-" * 40)

for name, param in model_manual.named_parameters():
    if param.requires_grad:
        print(f"{name:40s} {str(list(param.shape)):20s} {param.numel():>8,}")

total_params = sum(p.numel() for p in model_manual.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")

# ============================================================
# 7. 特征图可视化
# ============================================================
print("\n7. 特征图可视化")
print("-" * 40)


def visualize_feature_maps(model, input_tensor):
    """可视化中间层的特征图"""
    model.eval()
    with torch.no_grad():
        # 逐层获取特征图
        x = input_tensor.unsqueeze(1) if input_tensor.dim() == 3 else input_tensor

        # Block 1: 时间卷积
        x1 = model.conv1(x)
        x1 = model.bn1(x1)
        print(f"时间卷积后: {x1.shape} -> {x1.shape[1]} 个时间滤波器")

        # Block 1: 深度卷积
        x2 = model.depthwise_conv(x1)
        x2 = model.bn2(x2)
        x2 = model.activation1(x2)
        x2 = model.pool1(x2)
        print(f"深度卷积后: {x2.shape} -> {x2.shape[1]} 个空间滤波器")

        # Block 2: 分离卷积
        x3 = model.separable_conv(x2)
        x3 = model.bn3(x3)
        x3 = model.activation2(x3)
        x3 = model.pool2(x3)
        print(f"分离卷积后: {x3.shape} -> {x3.shape[1]} 个特征图")

        return x1, x2, x3


# 可视化
input_sample = torch.randn(1, 22, 1000)
feat1, feat2, feat3 = visualize_feature_maps(model_manual, input_sample)

# ============================================================
# 8. 模型保存与加载
# ============================================================
print("\n8. 模型保存与加载")
print("-" * 40)

# 保存模型
torch.save(model_manual.state_dict(), "eegnet_manual.pth")
print("模型已保存到: eegnet_manual.pth")

# 加载模型
model_loaded = EEGNetManual(n_channels=22, n_times=1000, n_classes=4)
model_loaded.load_state_dict(torch.load("eegnet_manual.pth", weights_only=True))
print("模型已加载")

# 验证
model_manual.eval()
model_loaded.eval()
with torch.no_grad():
    out1 = model_manual(dummy_input)
    out2 = model_loaded(dummy_input)
    diff = (out1 - out2).abs().max().item()
    print(f"模型输出差异: {diff:.10f} (应接近 0)")

# 清理临时文件
import os

if os.path.exists("eegnet_manual.pth"):
    os.remove("eegnet_manual.pth")

print("\n" + "=" * 60)
print("学习要点：")
print("1. EEGNet 使用分离卷积：时间卷积 -> 深度卷积 -> 分离卷积")
print("2. 深度卷积学习空间模式，类似 CSP")
print("3. 参数量小（约 2000-3000），适合实时应用")
print("4. braindecode 提供了优化的 EEGNet 实现")
print("5. 理解每一层的作用有助于调试和改进模型")
print("=" * 60)
