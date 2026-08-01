"""
01_tensor_basics.py - PyTorch 张量基础

学习目标：
- 理解 PyTorch 张量的创建和操作
- 掌握张量的形状变换（reshape, permute）
- 理解 GPU 加速的基本概念

EEG 数据在 PyTorch 中表示为张量，形状通常为：
(batch_size, channels, time_points) 或 (batch_size, channels, height, width)
"""

import torch
import numpy as np

print("=" * 60)
print("PyTorch 张量基础 - EEG 数据视角")
print("=" * 60)

# 1. 创建张量
print("\n1. 创建张量")
print("-" * 40)

# 从 numpy 数组创建（常见于加载 EEG 数据）
eeg_data = np.random.randn(64, 1000)  # 64 通道，1000 时间点
tensor_from_numpy = torch.from_numpy(eeg_data)
print(f"从 numpy 创建: shape={tensor_from_numpy.shape}, dtype={tensor_from_numpy.dtype}")

# 直接创建张量
random_tensor = torch.randn(32, 64, 1000)  # batch=32, channels=64, time=1000
print(f"随机张量: shape={random_tensor.shape}")

zeros_tensor = torch.zeros(32, 64, 1000)
ones_tensor = torch.ones(32, 64, 1000)
print(f"零张量: shape={zeros_tensor.shape}")
print(f"一Tensor: shape={ones_tensor.shape}")

# 2. 张量形状操作
print("\n2. 张量形状操作")
print("-" * 40)

# 模拟 EEG 数据: (batch, channels, time)
eeg_batch = torch.randn(16, 64, 500)
print(f"原始形状: {eeg_batch.shape}")

# reshape - 改变形状但不改变数据
flattened = eeg_batch.reshape(16, -1)  # -1 自动计算
print(f"展平后: {flattened.shape}")

# view - 类似 reshape，但要求内存连续
reshaped = eeg_batch.view(16, 64, 10, 50)  # 将时间维度拆分
print(f"重新塑形: {reshaped.shape}")

# permute - 改变维度顺序（常用于通道重排）
permuted = eeg_batch.permute(1, 0, 2)  # (channels, batch, time)
print(f"维度置换: {permuted.shape}")

# unsqueeze/squeeze - 添加/移除维度
expanded = eeg_batch.unsqueeze(1)  # 添加通道维度
print(f"添加维度: {expanded.shape}")
squeezed = expanded.squeeze(1)  # 移除维度
print(f"移除维度: {squeezed.shape}")

# 3. 张量运算
print("\n3. 张量运算")
print("-" * 40)

a = torch.randn(32, 64, 100)
b = torch.randn(32, 64, 100)

# 逐元素运算
c = a + b
d = a * 2
e = torch.relu(a)  # 激活函数
print(f"逐元素运算: c.shape={c.shape}, d.shape={d.shape}, e.shape={e.shape}")

# 矩阵运算
matrix_a = torch.randn(32, 64, 128)
matrix_b = torch.randn(32, 128, 10)
matrix_c = torch.bmm(matrix_a, matrix_b)  # 批量矩阵乘法
print(f"批量矩阵乘法: {matrix_a.shape} @ {matrix_b.shape} = {matrix_c.shape}")

# 归约操作
mean_val = a.mean()
sum_val = a.sum(dim=2)  # 沿时间维度求和
max_val, max_idx = a.max(dim=1)  # 沿通道维度求最大值
print(f"均值: {mean_val.item():.4f}")
print(f"沿时间维度求和: {sum_val.shape}")
print(f"沿通道维度最大值: {max_val.shape}, 索引: {max_idx.shape}")

# 4. GPU 加速
print("\n4. GPU 加速")
print("-" * 40)

print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    # 将张量移动到 GPU
    gpu_tensor = a.cuda()
    print(f"GPU 张量设备: {gpu_tensor.device}")
    
    # 在 GPU 上执行运算
    gpu_result = gpu_tensor + gpu_tensor
    print(f"GPU 运算结果: {gpu_result.shape}")
    
    # 移回 CPU
    cpu_result = gpu_result.cpu()
    print(f"移回 CPU: {cpu_result.device}")
else:
    print("未检测到 GPU，使用 CPU 进行训练")

# 5. 自动求导
print("\n5. 自动求导 (Autograd)")
print("-" * 40)

# 创建需要梯度的张量
x = torch.randn(3, requires_grad=True)
print(f"x 需要梯度: {x.requires_grad}")

# 计算
y = x * 2
z = y.sum()
print(f"y = x * 2, z = y.sum()")

# 反向传播
z.backward()
print(f"x 的梯度: {x.grad}")

# 6. EEG 数据实战示例
print("\n6. EEG 数据实战示例")
print("-" * 40)

# 模拟一个批次的 EEG 数据
batch_size = 8
n_channels = 22
n_times = 1000

eeg_input = torch.randn(batch_size, n_channels, n_times)
print(f"EEG 输入形状: {eeg_input.shape}")
print(f"  - 批次大小: {batch_size}")
print(f"  - 通道数: {n_channels}")
print(f"  - 时间点: {n_times}")

# 模拟卷积操作（EEGNet 的第一层）
# 输入: (batch, channels, time) -> (batch, 1, channels, time)
eeg_4d = eeg_input.unsqueeze(1)
print(f"添加通道维度后: {eeg_4d.shape}")

# 时间卷积
conv1 = torch.nn.Conv2d(
    in_channels=1,
    out_channels=16,
    kernel_size=(1, 125),  # 只在时间维度卷积
    padding=(0, 62)
)
output1 = conv1(eeg_4d)
print(f"时间卷积后: {output1.shape}")

# 深度卷积（空间滤波）
conv2 = torch.nn.Conv2d(
    in_channels=16,
    out_channels=32,
    kernel_size=(n_channels, 1),  # 只在空间维度卷积
    groups=16  # 深度可分离卷积
)
output2 = conv2(output1)
print(f"深度卷积后: {output2.shape}")

print("\n" + "=" * 60)
print("学习要点：")
print("1. EEG 数据形状: (batch, channels, time)")
print("2. 卷积前通常需要 unsqueeze 添加通道维度")
print("3. EEGNet 使用分离卷积：先时间卷积，再空间卷积")
print("4. 理解 reshape/permute 对于模型构建至关重要")
print("=" * 60)
