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
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
import mne
import copy
from sklearn.metrics import accuracy_score, cohen_kappa_score
# 在 BCI 运动想象领域， Kappa 系数是比准确率更常用的指标 ，因为它能更客观地反映模型对 EEG 信号的真实解码能力。
# kappa = (观察到的一致率 - 随机一致率) / (1 - 随机一致率)

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
    "sfreq": 128,  # 采样率（降采样后）
    "tmin": 0.0,  # trial 开始时间（秒）
    "tmax": 4.0,  # trial 结束时间（秒），BCI 运动想象持续 4 秒
    "n_times": 512,  # 时间窗口长度（样本数）
    "f1": 8,  # EEGNet 参数
    "f2": 16,
    "depth": 2,
    "kernel_length": 64,
    "pool1_kernel_size": 4,
    "pool2_kernel_size": 8,
    "dropout": 0.3,
    "batch_size": 32,
    "n_epochs": 50,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 1,  # 早停耐心值
    "n_folds": 5,  # 交叉验证折数
    "random_seed": 42,
}

print("实验配置:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# 设置随机种子
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])


# ============================================================
# 2. 数据加载（使用 MOABB）
# ============================================================
print("\n2. 数据加载")
print("-" * 40)

try:
    from braindecode.datasets import MOABBDataset
    from braindecode.preprocessing import (
        preprocess,
        PickTypes, # 选择 EEG 通道
        Filter,    # 带通滤波
        Resample,  # 降采样到 128 Hz
        Rescale,   # 转换为伏特（数据从微伏转为伏特）
        SetEEGReference, # 设置平均参考
        create_windows_from_events,
    )
    """
    Subject（被试 / 人）
    └── Session（会话 / 天）
            └── Run（轮次 / 单次采集文件）
                └── Trial（试次 / 单个刺激事件）
    """
    # 加载 BCI Competition IV 2a 数据集
    print(f"正在加载被试 {CONFIG['subject_id']} 的数据...")
    dataset = MOABBDataset(
        dataset_name="BNCI2014_001",  # BCI Competition IV 2a
        subject_ids=[1],
    )

    print(dataset.description)

    # 数据预处理
    print("\n应用预处理...")
    # 注意：MOABB 返回的数据单位通常是微伏(μV)，需要乘以 1e-6 转换为伏特(V)
    # braindecode 期望的输入单位是伏特
    mne.set_log_level("ERROR")  # 关闭 mne 日志，避免打印警告
    preprocessors = [
        PickTypes(eeg=True, misc=False),         # 选择 EEG 通道
        Filter(l_freq=4.0, h_freq=38.0),         # 带通滤波（在 μV 量级上进行，精度更高）
        Resample(sfreq=CONFIG["sfreq"]),         # 降采样到 128 Hz
        SetEEGReference(ref_channels="average"), # 设置平均参考
        Rescale(scalings=1e-6),                  # 最后转换为伏特（数据从微伏转为伏特）
    ]
    preprocess(dataset, preprocessors)

    print("预处理完成")
    """
    原始数据:
    BaseConcatDataset
    └── RawDataset (subject=1, session=0train, run=0)
    └── RawDataset (subject=1, session=0train, run=1)
    └── ...
    └── RawDataset (subject=1, session=1test, run=5)

    ↓ create_windows_from_events

    窗口数据:
    BaseConcatDataset
    └── WindowsDataset (subject=1, session=0train, run=0, 48 windows)
    └── WindowsDataset (subject=1, session=0train, run=1, 48 windows)
    └── ...
    └── WindowsDataset (subject=1, session=1test, run=5, 48 windows)
    """
    # 创建窗口
    print("\n创建滑动窗口...")
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,  # trial 起始偏移（采样点数），0 表示窗口从 trial 开始事件处开始，若设为正数则延迟开始，负数则提前开始
        trial_stop_offset_samples=0,   # trial 结束偏移（采样点数），0 表示窗口在 trial 结束事件处结束，若设为正数则延长结束，负数则提前结束
        window_size_samples=CONFIG["n_times"],
        window_stride_samples=CONFIG["n_times"],
        preload=True,
    ) # -> BaseConcatDataset[WindowsDataset | EEGWindowsDataset]
    print(f"窗口数据集: {len(windows_dataset)} 个样本")
    print(f"窗口数据集类型: {type(windows_dataset)}")
    print(f"窗口数据集样本类型: {type(windows_dataset[0])}")


    # 划分训练集和测试集（按 session）
    print("\n划分训练集和测试集...")
    # dict[str, WindowsDataset]
    splits = windows_dataset.split('session')
    print(f"数据划分 keys: {list(splits.keys())}")


    """
    划分训练集和测试集...
    数据划分 keys: ['0train', '1test']
    训练集: 288 个样本 (session=0train)
    测试集: 288 个样本 (session=1test)
    """

    train_key = None
    test_key = None
    for key in splits.keys():
        key_str = str(key).lower()
        if 'train' in key_str: # "0train"
            train_key = key
        elif 'test' in key_str: # "1test"
            test_key = key

    if train_key is None or test_key is None:
        keys = list(splits.keys())
        train_key = keys[0]
        test_key = keys[1]
        print(f"警告：无法自动识别 train/test，使用 {train_key} 作为 train，{test_key} 作为 test")

    train_dataset = splits[train_key] # WindowsDataset
    test_dataset = splits[test_key]   # WindowsDataset
    print(f"训练集: {len(train_dataset)} 个样本 (session={train_key})")
    print(f"测试集: {len(test_dataset)} 个样本 (session={test_key})")

except Exception as e:
    print(f"数据加载失败: {e}")
    print("使用模拟数据进行演示...")



# WindowsDataset 转为 numpy 数组格式
# KFold 需要 numpy 做索引划分
# 获取训练集数据（仅用于交叉验证）
X_train_cv = []
y_train_cv = []
for i in range(len(train_dataset)):
    X, y, _ = train_dataset[i]
    X_train_cv.append(X)
    y_train_cv.append(y)
X_train_cv = np.array(X_train_cv)
y_train_cv = np.array(y_train_cv)
print(f"训练数据形状: X={X_train_cv.shape}, y={y_train_cv.shape}")
print(f"训练数据类别分布: {np.bincount(y_train_cv)}")

# 获取测试集数据（用于最终评估）
X_test = []
y_test = []
for i in range(len(test_dataset)):
    X, y, _ = test_dataset[i]
    X_test.append(X)
    y_test.append(y)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(f"测试数据形状: X={X_test.shape}, y={y_test.shape}")
print(f"测试数据类别分布: {np.bincount(y_test)}")


# ============================================================
# 3. 定义模型
# ============================================================
print("\n3. 定义模型")
print("-" * 40)

from braindecode.models import EEGNet, ShallowFBCSPNet, Deep4Net

# 定义要比较的模型
models_dict = {
    "EEGNet": EEGNet(
        n_chans=CONFIG["n_channels"],
        n_times=CONFIG["n_times"],
        n_outputs=CONFIG["n_classes"],
        F1=CONFIG["f1"],
        F2=CONFIG["f2"],
        D=CONFIG["depth"],
        kernel_length=CONFIG["kernel_length"],
        pool1_kernel_size=CONFIG["pool1_kernel_size"],
        pool2_kernel_size=CONFIG["pool2_kernel_size"],
        drop_prob=CONFIG["dropout"],
    ),
    "ShallowConvNet": ShallowFBCSPNet(
        n_chans=CONFIG["n_channels"],
        n_times=CONFIG["n_times"],
        n_outputs=CONFIG["n_classes"],
    ),
    "DeepConvNet": Deep4Net(
        n_chans=CONFIG["n_channels"],
        n_times=CONFIG["n_times"],
        n_outputs=CONFIG["n_classes"],
    ),
}

print("要比较的模型:")
for name, model in models_dict.items():
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: {n_params:,} 参数")

# ============================================================
# 4. 训练函数
# ============================================================
print("\n4. 训练函数")
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
    best_val_kappa = 0.0
    best_model_state = model.state_dict()
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_kappa": []}

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
        val_preds = []
        val_labels = []

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
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())

        val_loss /= val_total
        val_acc = 100.0 * val_correct / val_total
        val_kappa = cohen_kappa_score(val_labels, val_preds)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_kappa"].append(val_kappa)

        # 早停检查：kappa 或 accuracy 任一变好都更新（避免 kappa 恒为 0 导致的死锁）
        if val_kappa > best_val_kappa or val_acc > best_val_acc:
            if val_kappa > best_val_kappa:
                best_val_kappa = val_kappa
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Kappa: {val_kappa:.4f}")
    # 加载最佳模型
    model.load_state_dict(best_model_state)

    return model, history, best_val_acc, best_val_kappa

# ============================================================
# 5. 交叉验证
# ============================================================
print("\n5. 交叉验证")
print("-" * 40)

# 设备选择
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

# 存储结果
results = {name: [] for name in models_dict.keys()}
print(results)


# 交叉验证:5折
skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_seed"])

"""
    X_train_cv  ← train_dataset (MOABB train session, 288 个 trials)
     │
     └── skf.split(X_train_cv, y_train_cv)  ← 5 折交叉验证
             │
             ├── X_train (80%) → fold_train_dataset → train_loader
             └── X_val   (20%) → fold_val_dataset   → val_loader  ← L419 用到的就是这个
"""

# 交叉验证循环
# StratifiedKFold 需要 X 和 y 来保持类别比例
for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cv, y_train_cv)):
    print(f"\n折 {fold + 1}/{CONFIG['n_folds']}")
    print("-" * 40)
    # 划分数据
    X_train, X_val = X_train_cv[train_idx], X_train_cv[val_idx]
    y_train, y_val = y_train_cv[train_idx], y_train_cv[val_idx]
    print(f"训练集: {len(X_train)},类别：{np.bincount(y_train)}")
    print(f"验证集: {len(X_val)},类别：{np.bincount(y_val)}")
    # 创建 DataLoader
    fold_train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(), 
        torch.from_numpy(y_train).long()
    )
    fold_val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(), 
        torch.from_numpy(y_val).long()
    )
    train_loader = DataLoader(fold_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(fold_val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 训练每个模型
    for model_name, model_template in models_dict.items():
        print(f"\n训练 {model_name}...")
        model = copy.deepcopy(model_template)
        model, history, best_val_acc, best_val_kappa = train_model(
            model,
            train_loader,
            val_loader,
            n_epochs=CONFIG["n_epochs"],
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
            patience=CONFIG["patience"],
            device=device,
        )
        results[model_name].append({"accuracy": best_val_acc / 100.0, "kappa": best_val_kappa})
        print(f"  {model_name}: Acc={best_val_acc / 100.0:.4f}, Kappa={best_val_kappa:.6f}")


for model_name, fold_results in results.items():
    print(f"\n{model_name} 结果:")
    for result in fold_results:
        print(result)

# ============================================================
# 6. 测试集最终评估
# ============================================================
print("\n6. 测试集最终评估")
print("-" * 40)

# 设备选择
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

# 存储结果
print(results)

# 对每个模型，在完整训练集上重新训练，然后在测试集上评估
# 使用 train_test_split 从训练集中划出一部分作为内部验证集（用于早停），避免数据泄漏
test_results = {}
for model_name, model_template in models_dict.items():
    print(f"\n在完整训练集上训练 {model_name}...")


    """
    X_train_cv (288)
    │
    ├── train_test_split(test_size=0.1)
    │       │
    │       ├── X_tr_final (259, 90%) ──→ final_train_loader
    │       └── X_val_final (29,  10%) ──→ final_val_loader ──→ 传入 train_model()
    │                                                              │
    │                                                      同样的早停逻辑
    │                                                      监控 final_val_loader
    """
    # 划分训练集为最终训练集和内部验证集（用于早停）
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_train_cv, y_train_cv, test_size=0.1,
        random_state=CONFIG["random_seed"],  # 控制随机种子，确保每次划分结果可复现
        stratify=y_train_cv  # 分层划分，保证划分后的训练集和验证集 类别比例 与原始数据一致
    )
    print(f"最终训练集: {len(X_tr_final)},类别：{np.bincount(y_tr_final)}")
    print(f"最终验证集: {len(X_val_final)},类别：{np.bincount(y_val_final)}")

    # 创建 DataLoader
    final_train_dataset = TensorDataset(
        torch.from_numpy(X_tr_final).float(),
        torch.from_numpy(y_tr_final).long()
    )
    final_val_dataset = TensorDataset(
        torch.from_numpy(X_val_final).float(),
        torch.from_numpy(y_val_final).long()
    )
    final_train_loader = DataLoader(final_train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    final_val_loader = DataLoader(final_val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = copy.deepcopy(model_template)
    model, history, best_val_acc, best_val_kappa = train_model(
        model,
        final_train_loader,
        final_val_loader,
        n_epochs=CONFIG["n_epochs"],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        patience=CONFIG["patience"],
        device=device,
    )

    # 在测试集上评估
    test_tensor_dataset = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(y_test).long()
    )
    test_loader = DataLoader(test_tensor_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model.eval()
    y_pred_test = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = outputs.max(1)
            y_pred_test.extend(predicted.cpu().numpy())
    y_pred_test = np.array(y_pred_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_kappa = cohen_kappa_score(y_test, y_pred_test)
    test_results[model_name] = {"accuracy": test_acc, "kappa": test_kappa}
    print(f"  {model_name} 测试集: Acc={test_acc:.4f}, Kappa={test_kappa:.4f}")

# ============================================================
# 7. 结果汇总
# ============================================================
print("\n7. 结果汇总")
print("-" * 40)

# 计算交叉验证性能
summary = []
for model_name, fold_results in results.items():
    if len(fold_results) > 0:
        accs = [r["accuracy"] for r in fold_results]
        kappas = [r["kappa"] for r in fold_results]

        summary.append(
            {
                "Model": model_name,
                "CV Accuracy (mean)": np.mean(accs),
                "CV Accuracy (std)": np.std(accs),
                "CV Kappa (mean)": np.mean(kappas),
                "CV Kappa (std)": np.std(kappas),
                "Test Accuracy": test_results[model_name]["accuracy"],
                "Test Kappa": test_results[model_name]["kappa"],
            }
        )

summary_df = pd.DataFrame(summary)
print("\n交叉验证 + 测试集性能比较:")
print(summary_df.to_string(index=False))

# ============================================================
# 8. 可视化结果
# ============================================================
print("\n8. 可视化结果")
print("-" * 40)

plt.rcParams['font.sans-serif'] = ['Helvetica']

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 交叉验证准确率对比
ax = axes[0, 0]
models = [s["Model"] for s in summary]
cv_acc_means = [s["CV Accuracy (mean)"] for s in summary]
cv_acc_stds = [s["CV Accuracy (std)"] for s in summary]

bars = ax.bar(models, cv_acc_means, yerr=cv_acc_stds, capsize=5, alpha=0.7)
ax.set_ylabel("Accuracy")
ax.set_title("CV Accuracy (Train Only)")
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

for bar, mean, std in zip(bars, cv_acc_means, cv_acc_stds):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{mean:.3f}\n±{std:.3f}",
        ha="center",
        va="bottom",
    )

# 2. 交叉验证 Kappa 对比
ax = axes[0, 1]
cv_kappa_means = [s["CV Kappa (mean)"] for s in summary]
cv_kappa_stds = [s["CV Kappa (std)"] for s in summary]

bars = ax.bar(models, cv_kappa_means, yerr=cv_kappa_stds, capsize=5, alpha=0.7, color="orange")
ax.set_ylabel("Cohen's Kappa")
ax.set_title("CV Kappa (Train Only)")
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

for bar, mean, std in zip(bars, cv_kappa_means, cv_kappa_stds):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{mean:.3f}\n±{std:.3f}",
        ha="center",
        va="bottom",
    )

# 3. 测试集准确率对比
ax = axes[1, 0]
test_accs = [s["Test Accuracy"] for s in summary]
test_kappas = [s["Test Kappa"] for s in summary]

x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, test_accs, width, label="Accuracy", alpha=0.8)
bars2 = ax.bar(x + width/2, test_kappas, width, label="Kappa", alpha=0.8, color="green")
ax.set_ylabel("Score")
ax.set_title("Test Set Performance")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim([0, 1])
ax.legend()
ax.grid(True, alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

# 4. 各折准确率
ax = axes[1, 1]
for model_name, fold_results in results.items():
    if len(fold_results) > 0:
        accs = [r["accuracy"] for r in fold_results]
        ax.plot(range(1, len(accs) + 1), accs, marker="o", label=model_name)

ax.set_xlabel("Fold")
ax.set_ylabel("Accuracy")
ax.set_title("CV Accuracy per Fold (Train Only)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(range(1, CONFIG["n_folds"] + 1))

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
print("2. 划分 train/test：仅用 train 做交叉验证，test 做最终评估")
print("3. 交叉验证评估模型泛化能力")
print("4. 多模型比较选择最佳方案")
print("5. 完整的端到端管道：数据 -> 预处理 -> 训练 -> 评估")
print("=" * 60)
