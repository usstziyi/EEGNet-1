"""
07_project.py - 实战项目：运动想象 EEG 解码

项目目标：
- 使用 MOABB 加载 BCI Competition IV 2a 数据集
- 构建完整的 EEG 解码管道
- 比较多个模型的性能
- 实现交叉验证和统计分析

BCI Competition IV 2a 数据集：
- 9 个被试
- 4 类运动想象：左手、右手、双脚、舌头
- 22 个 EEG 通道
- 采样率 250 Hz
- 每个被试 288 个 trials（训练 + 测试）

这个实战项目将展示：
1. 数据加载与预处理
2. 数据管道构建
3. 模型训练与评估
4. 模型比较
5. 结果可视化
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, cohen_kappa_score

print("=" * 60)
print("实战项目：运动想象 EEG 解码")
print("=" * 60)

# ============================================================
# 1. 配置参数
# ============================================================
print("\n1. 配置参数")
print("-" * 40)

# 实验配置
CONFIG = {
    "subject_id": 1,  # 被试 ID (1-9)
    "n_classes": 4,  # 4 类运动想象
    "n_channels": 22,  # EEG 通道数
    "sfreq": 250,  # 采样率
    "tmin": 0.0,  # trial 开始时间（秒）
    "tmax": 6.0,  # trial 结束时间（秒）
    "f1": 8,  # EEGNet 参数
    "f2": 16,
    "depth": 2,
    "kernel_length": 125,
    "dropout": 0.5,
    "batch_size": 64,
    "n_epochs": 50,
    "lr": 0.001,
    "weight_decay": 0.01,
    "patience": 10,  # 早停耐心值
    "n_folds": 5,  # 交叉验证折数
    "random_seed": 42,
}

print("实验配置:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# 设置随机种子
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

# 计算时间窗口长度（始终定义，避免后续引用错误）
n_times = int((CONFIG["tmax"] - CONFIG["tmin"]) * 128)

# ============================================================
# 2. 数据加载（使用 MOABB）
# ============================================================
print("\n2. 数据加载")
print("-" * 40)

try:
    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import (
        preprocess,
        Preprocessor,
        create_windows_from_events,
    )

    # 加载 BCI Competition IV 2a 数据集
    print(f"正在加载被试 {CONFIG['subject_id']} 的数据...")
    dataset = MOABBDataset(
        dataset_name="BNCI2014001",  # BCI Competition IV 2a
        subject_ids=[CONFIG["subject_id"]],
    )
    print(f"数据集加载成功: {len(dataset.datasets)} 个 session")

    # 数据预处理
    print("\n应用预处理...")
    # 注意：MOABB 返回的数据单位通常是微伏(μV)，需要乘以 1e-6 转换为伏特(V)
    # braindecode 期望的输入单位是伏特
    preprocessors = [
        # 选择 EEG 通道
        Preprocessor("pick_types", eeg=True, misc=False),
        # 转换为伏特（数据从微伏转为伏特）
        Preprocessor(lambda data: data * 1e-6),
        # 带通滤波
        Preprocessor("filter", l_freq=4.0, h_freq=38.0),
        # 降采样
        Preprocessor("resample", sfreq=128),
    ]
    preprocess(dataset, preprocessors)

    # 设置平均参考（set_eeg_reference 返回元组，不适合放在 Preprocessor 中）
    for ds in dataset.datasets:
        ds.raw.set_eeg_reference("average", projection=False)

    print("预处理完成")

    # 创建窗口
    print("\n创建滑动窗口...")
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=int((CONFIG["tmax"] - CONFIG["tmin"]) * 128),
        window_stride_samples=int((CONFIG["tmax"] - CONFIG["tmin"]) * 128),
        preload=True,
    )
    print(f"窗口数据集: {len(windows_dataset)} 个样本")

    # 检查数据
    if len(windows_dataset) > 0:
        X_sample, y_sample, _ = windows_dataset[0]
        print(f"单个样本: X.shape={X_sample.shape}, y={y_sample}")

    data_loaded = True

except Exception as e:
    print(f"数据加载失败: {e}")
    print("使用模拟数据进行演示...")
    data_loaded = False

# ============================================================
# 3. 创建模拟数据（如果 MOABB 不可用）
# ============================================================
if not data_loaded:
    print("\n3. 创建模拟数据")
    print("-" * 40)

    # 生成模拟 EEG 数据
    n_subjects = 1
    n_sessions = 2  # train + test
    n_trials_per_session = 144
    n_channels = CONFIG["n_channels"]
    n_times = int((CONFIG["tmax"] - CONFIG["tmin"]) * 128)
    n_classes = CONFIG["n_classes"]

    # 创建模拟数据
    X_all = []
    y_all = []

    for session in range(n_sessions):
        # 生成随机 EEG 数据
        X_session = np.random.randn(n_trials_per_session, n_channels, n_times)
        # 生成随机标签
        y_session = np.random.randint(0, n_classes, n_trials_per_session)

        X_all.append(X_session)
        y_all.append(y_session)

    # 合并
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    print(f"模拟数据:")
    print(f"  总样本数: {len(X_all)}")
    print(f"  样本形状: {X_all.shape}")
    print(f"  类别分布: {np.bincount(y_all)}")

    # 转换为 PyTorch 数据集
    from torch.utils.data import TensorDataset

    X_tensor = torch.from_numpy(X_all).float()
    y_tensor = torch.from_numpy(y_all).long()
    full_dataset = TensorDataset(X_tensor, y_tensor)

    data_loaded = True

# ============================================================
# 4. 定义模型
# ============================================================
print("\n4. 定义模型")
print("-" * 40)

from braindecode.models import EEGNetv4, ShallowFBCSPNet, DeepConvNet

# 定义要比较的模型
models_dict = {
    "EEGNet": EEGNetv4(
        in_chans=CONFIG["n_channels"],
        n_times=n_times if data_loaded else int((CONFIG["tmax"] - CONFIG["tmin"]) * 128),
        n_classes=CONFIG["n_classes"],
        F1=CONFIG["f1"],
        F2=CONFIG["f2"],
        depth=CONFIG["depth"],
        kernel_length=CONFIG["kernel_length"],
        drop_prob=CONFIG["dropout"],
    ),
    "ShallowConvNet": ShallowFBCSPNet(
        in_chans=CONFIG["n_channels"],
        n_times=n_times if data_loaded else int((CONFIG["tmax"] - CONFIG["tmin"]) * 128),
        n_classes=CONFIG["n_classes"],
    ),
    "DeepConvNet": DeepConvNet(
        in_chans=CONFIG["n_channels"],
        n_times=n_times if data_loaded else int((CONFIG["tmax"] - CONFIG["tmin"]) * 128),
        n_classes=CONFIG["n_classes"],
    ),
}

print("要比较的模型:")
for name, model in models_dict.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n_params:,} 参数")

# ============================================================
# 5. 训练函数
# ============================================================
print("\n5. 训练函数")
print("-" * 40)


def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs,
    lr,
    weight_decay,
    patience,
    device,
):
    """训练模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(n_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            train_total += y_batch.size(0)
            train_correct += predicted.eq(y_batch).sum().item()

        train_loss /= train_total
        train_acc = 100.0 * train_correct / train_total

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total

        # 更新学习率
        scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return model, history, best_val_acc


# ============================================================
# 6. 交叉验证
# ============================================================
print("\n6. 交叉验证")
print("-" * 40)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 存储结果
results = {name: [] for name in models_dict.keys()}

# 交叉验证
kf = KFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_seed"])

if data_loaded:
    # 获取数据
    if "full_dataset" in locals():
        X_all = full_dataset.tensors[0].numpy()
        y_all = full_dataset.tensors[1].numpy()
    else:
        # 从 windows_dataset 提取
        X_all = []
        y_all = []
        for i in range(len(windows_dataset)):
            X, y, _ = windows_dataset[i]
            X_all.append(X.numpy())
            y_all.append(y)
        X_all = np.array(X_all)
        y_all = np.array(y_all)

    print(f"数据形状: X={X_all.shape}, y={y_all.shape}")
    print(f"类别分布: {np.bincount(y_all)}")

    # 交叉验证循环
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
        print(f"\n折 {fold + 1}/{CONFIG['n_folds']}")
        print("-" * 40)

        # 划分数据
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")

        # 创建 DataLoader
        from torch.utils.data import TensorDataset

        train_dataset = TensorDataset(
            torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long()
        )

        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG["batch_size"], shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

        # 训练每个模型
        for model_name, model_template in models_dict.items():
            print(f"\n训练 {model_name}...")

            # 重新初始化模型
            import copy

            model = copy.deepcopy(model_template)

            # 训练
            model, history, best_val_acc = train_model(
                model,
                train_loader,
                val_loader,
                n_epochs=CONFIG["n_epochs"],
                lr=CONFIG["lr"],
                weight_decay=CONFIG["weight_decay"],
                patience=CONFIG["patience"],
                device=device,
            )

            # 在验证集上评估
            model.eval()
            y_pred = []
            with torch.no_grad():
                for X_batch, _ in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    _, predicted = outputs.max(1)
                    y_pred.extend(predicted.cpu().numpy())

            y_pred = np.array(y_pred)
            acc = accuracy_score(y_val, y_pred)
            kappa = cohen_kappa_score(y_val, y_pred)

            results[model_name].append({"accuracy": acc, "kappa": kappa})

            print(f"  {model_name}: Acc={acc:.4f}, Kappa={kappa:.4f}")

# ============================================================
# 7. 结果汇总
# ============================================================
print("\n7. 结果汇总")
print("-" * 40)

# 计算平均性能
summary = []
for model_name, fold_results in results.items():
    if len(fold_results) > 0:
        accs = [r["accuracy"] for r in fold_results]
        kappas = [r["kappa"] for r in fold_results]

        summary.append(
            {
                "Model": model_name,
                "Accuracy (mean)": np.mean(accs),
                "Accuracy (std)": np.std(accs),
                "Kappa (mean)": np.mean(kappas),
                "Kappa (std)": np.std(kappas),
            }
        )

summary_df = pd.DataFrame(summary)
print("\n模型性能比较:")
print(summary_df.to_string(index=False))

# ============================================================
# 8. 可视化结果
# ============================================================
print("\n8. 可视化结果")
print("-" * 40)

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 准确率对比
ax = axes[0, 0]
models = [s["Model"] for s in summary]
acc_means = [s["Accuracy (mean)"] for s in summary]
acc_stds = [s["Accuracy (std)"] for s in summary]

bars = ax.bar(models, acc_means, yerr=acc_stds, capsize=5, alpha=0.7)
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison")
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

# 在柱子上显示数值
for bar, mean, std in zip(bars, acc_means, acc_stds):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{mean:.3f}\n±{std:.3f}",
        ha="center",
        va="bottom",
    )

# 2. Kappa 对比
ax = axes[0, 1]
kappa_means = [s["Kappa (mean)"] for s in summary]
kappa_stds = [s["Kappa (std)"] for s in summary]

bars = ax.bar(models, kappa_means, yerr=kappa_stds, capsize=5, alpha=0.7, color="orange")
ax.set_ylabel("Cohen's Kappa")
ax.set_title("Model Kappa Comparison")
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

for bar, mean, std in zip(bars, kappa_means, kappa_stds):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{mean:.3f}\n±{std:.3f}",
        ha="center",
        va="bottom",
    )

# 3. 各折准确率
ax = axes[1, 0]
for model_name, fold_results in results.items():
    if len(fold_results) > 0:
        accs = [r["accuracy"] for r in fold_results]
        ax.plot(range(1, len(accs) + 1), accs, marker="o", label=model_name)

ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy per Fold")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, CONFIG["n_folds"] + 1))

# 4. 箱线图
ax = axes[1, 1]
data_to_plot = []
labels = []
for model_name, fold_results in results.items():
    if len(fold_results) > 0:
        accs = [r["accuracy"] for r in fold_results]
        data_to_plot.append(accs)
        labels.append(model_name)

if len(data_to_plot) > 0:
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Distribution")
    ax.grid(True, alpha=0.3)

    # 设置颜色
    colors = ["lightblue", "lightgreen", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors[:len(labels)]):
        patch.set_facecolor(color)

plt.tight_layout()
plt.savefig("project_results.png", dpi=150)
print("结果图已保存到: project_results.png")
plt.close()

# ============================================================
# 9. 保存结果
# ============================================================
print("\n9. 保存结果")
print("-" * 40)

# 保存为 CSV
summary_df.to_csv("project_results.csv", index=False)
print("结果已保存到: project_results.csv")

# 打印最终结果
print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("\n生成的文件:")
print("  - project_results.png: 结果可视化")
print("  - project_results.csv: 详细结果数据")
print("\n学习要点：")
print("1. 使用 MOABB 加载公开 BCI 数据集")
print("2. 交叉验证评估模型泛化能力")
print("3. 多模型比较选择最佳方案")
print("4. 完整的端到端管道：数据 -> 预处理 -> 训练 -> 评估")
print("=" * 60)
