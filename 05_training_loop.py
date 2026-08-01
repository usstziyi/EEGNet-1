"""
05_training_loop.py - 训练循环与监控

学习目标：
- 掌握 PyTorch 训练循环的构建
- 理解损失函数、优化器、学习率调度器
- 实现训练监控和早停机制
- 使用 braindecode 的 EZGridClassifier 进行快速训练

braindecode 提供了两种训练方式：
1. 手动训练循环（灵活，适合学习）
2. EZGridClassifier（简单，基于 scikit-learn API）
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

print("=" * 60)
print("训练循环与监控")
print("=" * 60)

# ============================================================
# 1. 创建模拟数据集
# ============================================================
print("\n1. 创建模拟数据集")
print("-" * 40)

# 模拟 EEG 数据
n_samples = 1000
n_channels = 22
n_times = 1000
n_classes = 4

# 生成随机数据
X = torch.randn(n_samples, n_channels, n_times)
y = torch.randint(0, n_classes, (n_samples,))

# 划分训练集和验证集
n_train = int(0.8 * n_samples)
X_train, X_val = X[:n_train], X[n_train:]
y_train, y_val = y[:n_train], y[n_train:]

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(f"训练集: {len(train_dataset)} 样本")
print(f"验证集: {len(val_dataset)} 样本")
print(f"类别分布: {torch.bincount(y_train)}")

# ============================================================
# 2. 定义模型
# ============================================================
print("\n2. 定义模型")
print("-" * 40)

from braindecode.models import EEGNetv4

model = EEGNetv4(
    in_chans=n_channels,
    n_times=n_times,
    n_classes=n_classes,
    F1=8,
    F2=16,
    depth=2,
    kernel_length=125,
)

print(f"模型: {type(model).__name__}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 3. 损失函数和优化器
# ============================================================
print("\n3. 损失函数和优化器")
print("-" * 40)

# 损失函数
criterion = nn.CrossEntropyLoss()
print(f"损失函数: {criterion}")

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print(f"优化器: {optimizer}")

# 学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
print(f"学习率调度器: CosineAnnealingLR")

# ============================================================
# 4. 训练循环（手动实现）
# ============================================================
print("\n4. 训练循环（手动实现）")
print("-" * 40)


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


# ============================================================
# 5. 完整训练流程
# ============================================================
print("\n5. 完整训练流程")
print("-" * 40)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"使用设备: {device}")

# 训练参数
n_epochs = 10
best_val_acc = 0.0
patience = 5
patience_counter = 0

# 记录训练历史
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "lr": [],
}

print("\n开始训练...")
for epoch in range(n_epochs):
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

    # 验证
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    # 更新学习率
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()

    # 记录历史
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    # 打印进度
    print(
        f"Epoch {epoch+1:3d}/{n_epochs} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
        f"LR: {current_lr:.6f}"
    )

    # 早停检查
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> 保存最佳模型 (Val Acc: {best_val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  -> 早停触发，停止训练")
            break

# 加载最佳模型
model.load_state_dict(torch.load("best_model.pth", weights_only=True))
print(f"\n训练完成！最佳验证准确率: {best_val_acc:.2f}%")

# ============================================================
# 6. 可视化训练历史
# ============================================================
print("\n6. 可视化训练历史")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 损失曲线
axes[0, 0].plot(history["train_loss"], label="Train Loss", marker="o")
axes[0, 0].plot(history["val_loss"], label="Val Loss", marker="s")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Loss Curve")
axes[0, 0].legend()
axes[0, 0].grid(True)

# 准确率曲线
axes[0, 1].plot(history["train_acc"], label="Train Acc", marker="o")
axes[0, 1].plot(history["val_acc"], label="Val Acc", marker="s")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy (%)")
axes[0, 1].set_title("Accuracy Curve")
axes[0, 1].legend()
axes[0, 1].grid(True)

# 学习率曲线
axes[1, 0].plot(history["lr"], label="Learning Rate", marker="o", color="green")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Learning Rate")
axes[1, 0].set_title("Learning Rate Schedule")
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].set_yscale("log")

# 损失对比
axes[1, 1].plot(
    range(1, len(history["train_loss"]) + 1),
    history["train_loss"],
    label="Train",
    marker="o",
)
axes[1, 1].plot(
    range(1, len(history["val_loss"]) + 1),
    history["val_loss"],
    label="Validation",
    marker="s",
)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Loss")
axes[1, 1].set_title("Train vs Validation Loss")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=150)
print("训练历史图已保存到: training_history.png")
plt.close()

# ============================================================
# 7. 使用 braindecode 的 EZGridClassifier
# ============================================================
print("\n7. 使用 braindecode 的 EZGridClassifier")
print("-" * 40)

from braindecode.classifier import EEGClassifier

# braindecode 提供了基于 skorch 的 EEGClassifier
# 它封装了训练循环，使用更简洁

# 创建 EEGClassifier
clf = EEGClassifier(
    module=EEGNetv4,
    module__in_chans=n_channels,
    module__n_times=n_times,
    module__n_classes=n_classes,
    max_epochs=5,
    batch_size=64,
    lr=0.001,
    device=device,
    optimizer=torch.optim.AdamW,
    optimizer__weight_decay=0.01,
    train_split=None,  # 不使用内置验证集划分
    verbose=1,
)

print(f"EEGClassifier 已创建")
print(f"  - 模型: EEGNetv4")
print(f"  - 最大 epoch: {clf.max_epochs}")
print(f"  - 批次大小: {clf.batch_size}")

# 训练（需要转换为 numpy 数组）
# 注意：skorch 期望输入为 numpy 数组
X_train_np = X_train.numpy()
y_train_np = y_train.numpy()

print("\n使用 EEGClassifier 训练（5 个 epoch）...")
clf.fit(X_train_np, y_train_np)

# 预测
y_pred = clf.predict(X_val.numpy())
accuracy = (y_pred == y_val.numpy()).mean() * 100
print(f"\n验证准确率: {accuracy:.2f}%")

# ============================================================
# 8. 模型评估指标
# ============================================================
print("\n8. 模型评估指标")
print("-" * 40)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 使用手动训练的模型进行预测
model.eval()
with torch.no_grad():
    X_val_device = X_val.to(device)
    outputs = model(X_val_device)
    _, y_pred_manual = outputs.max(1)
    y_pred_manual = y_pred_manual.cpu().numpy()

y_true = y_val.numpy()

# 计算指标
acc = accuracy_score(y_true, y_pred_manual)
precision = precision_score(y_true, y_pred_manual, average="macro")
recall = recall_score(y_true, y_pred_manual, average="macro")
f1 = f1_score(y_true, y_pred_manual, average="macro")

print(f"准确率 (Accuracy): {acc:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 分数: {f1:.4f}")

print("\n分类报告:")
print(classification_report(y_true, y_pred_manual, target_names=[f"Class {i}" for i in range(n_classes)]))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred_manual)
print(f"\n混淆矩阵:")
print(cm)

# 可视化混淆矩阵
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

classes = [f"Class {i}" for i in range(n_classes)]
ax.set(
    xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel="True label",
    xlabel="Predicted label",
    title="Confusion Matrix",
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 在每个单元格中显示数值
fmt = "d"
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

fig.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\n混淆矩阵图已保存到: confusion_matrix.png")
plt.close()

# 清理临时文件
import os

if os.path.exists("best_model.pth"):
    os.remove("best_model.pth")

print("\n" + "=" * 60)
print("学习要点：")
print("1. 训练循环包括：前向传播、计算损失、反向传播、更新参数")
print("2. 验证集用于监控过拟合，早停防止过拟合")
print("3. 学习率调度器可以动态调整学习率")
print("4. braindecode 的 EEGClassifier 简化了训练流程")
print("5. 评估指标包括准确率、精确率、召回率、F1 分数")
print("=" * 60)
