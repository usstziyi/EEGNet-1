toolName: web_search
            
status: success
          
query: braindecode training module PyTorch Lightning training utilities
          

        
            
toolName: web_search
            
status: success
          
query: https://braindecode.org/stable/api.html
          
## braindecode `training` 模块

### 核心定位

`braindecode.training` 是 braindecode 提供的**训练辅助工具**，主要有两个功能：

1. **PyTorch Lightning 集成** — 让 braindecode 模型可以直接用 Lightning 的训练框架
2. **通用训练组件** — 损失函数、训练循环辅助等

### 模块结构

```
braindecode.training/
├── losses.py          # 损失函数
├── lightning.py       # PyTorch Lightning 集成
└── utils.py           # 训练工具函数
```

### 用法示例

#### 方式一：配合 PyTorch Lightning（推荐）

```python
import torch
import lightning as L  # 或 import pytorch_lightning as pl
from braindecode.models import EEGNet
from braindecode.training import EEGClassifierLightning

# 创建 Lightning 模块（braindecode 提供的封装）
model = EEGClassifierLightning(
    module=EEGNet,
    n_chans=22,
    n_times=1000,
    n_classes=4,
    lr=0.001,
)

# 用 Lightning Trainer 训练
trainer = L.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_dataloaders=train_loader)
```

#### 方式二：手动 + training 工具

```python
from braindecode.training.losses import CroppedLoss

# braindecode 特有的损失函数（支持裁剪输入）
criterion = CroppedLoss()

# 普通训练循环
for epoch in range(n_epochs):
    ...
```

### 主要组件

| 组件 | 说明 |
|------|------|
| `EEGClassifierLightning` | Lightning 版本的 EEG 分类器 |
| `CroppedLoss` | 支持裁剪输入的损失函数 |
| 其他辅助函数 | 数据处理、指标计算等 |

### 什么时候用 training 模块

| 场景 | 推荐方案 |
|------|---------|
| 快速训练 | `EEGClassifier` (skorch) |
| 需要 Lightning 功能 | `training` 模块 |
| 多 GPU 分布式训练 | `training` + Lightning |
| 复杂自定义训练 | 手动循环 + `training.losses` |

### 对比三种训练方式

```python
# 1. EEGClassifier（最简单）
from braindecode.classifier import EEGClassifier
clf = EEGClassifier(module=EEGNet, ...)
clf.fit(X, y)  # 一行搞定

# 2. training + Lightning（中等复杂度）
from braindecode.training import EEGClassifierLightning
import lightning as L
model = EEGClassifierLightning(module=EEGNet, ...)
L.Trainer(max_epochs=10).fit(model, train_loader)

# 3. 纯手动（最灵活）
from braindecode.models import EEGNet
model = EEGNet(...)
# 手动写训练循环
for epoch in range(10):
    for X_batch, y_batch in loader:
        ...
```

### 总结

```
braindecode.training = Lightning 集成 + 训练工具

适用场景：
  ✓ 想用 PyTorch Lightning 的高级功能（分布式训练、回调等）
  ✓ 需要配合 Lightning 生态（W&B、TensorBoard 等）
  ✗ 简单任务用 EEGClassifier 更方便
```

如果你对 Lightning 不熟悉，建议先学习 `EEGClassifier`，再考虑用 `training` 模块获得更强大的训练能力。