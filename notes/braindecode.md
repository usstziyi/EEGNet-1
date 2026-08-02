toolName: web_search
            
status: success
          
query: braindecode classifier module EEGRegressor EZGridClassifier full API
          

        
            
toolName: web_search
            
status: success
          
query: https://braindecode.org/stable/api.html
          
## braindecode 封装的任务类型

### 核心封装器

braindecode 提供了 **两个** scikit-learn 风格的封装器：

| 类名 | 任务类型 | 模块 |
|------|---------|------|
| `EEGClassifier` | **分类任务** | `braindecode.classifier` |
| `EEGRegressor` | **回归任务** | `braindecode.regressor` |

### EEGRegressor 用法

```python
from braindecode.regressor import EEGRegressor
from braindecode.models import EEGNet

# 回归任务示例
reg = EEGRegressor(
    module=EEGNet,
    module__n_chans=22,
    module__n_times=1000,
    module__n_outputs=1,  # 回归输出 1 维
    criterion=nn.MSELoss,  # 回归用 MSE
    max_epochs=10,
    lr=0.001,
)

reg.fit(X_train, y_train)  # y_train 是连续值
y_pred = reg.predict(X_test)
```

### 其他封装模块

除了 Classifier/Regressor，braindecode 还有以下高层封装：

```
braindecode 模块结构：
├── classifier/          # 分类器 (EEGClassifier)
├── regressor/           # 回归器 (EEGRegressor)
├── models/              # 模型库 (65+ 模型)
├── training/            # 训练工具
├── datasets/            # 数据集加载 (MOABB, BIDS, TUH...)
├── preprocessing/       # 预处理 (滤波、重采样、窗口切分)
├── augmentation/       # 数据增强
├── visualization/       # 可视化 (归因分析)
└── utils/              # 工具函数
```

### 对比：两个封装器的区别

| 特性 | EEGClassifier | EEGRegressor |
|------|--------------|--------------|
| 损失函数 | CrossEntropyLoss (默认) | MSELoss (默认) |
| 输出要求 | logits（无 softmax） | 连续值 |
| 评估指标 | accuracy, f1 等 | mse, mae, r2 等 |
| 适用场景 | 类别分类 | 连续值预测 |

### 还有一个 EZGridClassifier

```python
from braindecode.classifier import EEGClassifier  # 基础版
# 旧版本可能有 EZGridClassifier（已整合到新版本）
```

新版本 braindecode 将 `EZGridClassifier` 的功能整合到了 `EEGClassifier` 中，通过 skorch 实现交叉验证和网格搜索兼容。

### 总结

```
braindecode 的高层封装：
  ✓ EEGClassifier  — 分类任务（最常用）
  ✓ EEGRegressor   — 回归任务
  ✗ 没有更多封装器了
  
其他功能通过独立模块提供：
  • 数据加载 → braindecode.datasets
  • 预处理 → braindecode.preprocessing  
  • 增强   → braindecode.augmentation
  • 可视化 → braindecode.visualization
```