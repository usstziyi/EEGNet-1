import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from braindecode import EEGClassifier
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    preprocess,
    PickTypes,
    Filter,
    Resample,
    Rescale,
    SetEEGReference,
    create_windows_from_events,
)
from braindecode.models import EEGNet, ShallowFBCSPNet, Deep4Net
from skorch.callbacks import EarlyStopping
from skorch.helper import predefined_split
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import TensorDataset
import mne

print("=" * 60)
print("EEG Motor Imagery Decoding with braindecode API")
print("=" * 60)

CONFIG = {
    "subject_id": 1,
    "n_classes": 4,
    "n_channels": 22,
    "sfreq": 128,
    "n_times": 512,
    "dropout": 0.3,
    "batch_size": 32,
    "n_epochs": 50,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 15,
    "n_folds": 5,
    "random_seed": 42,
}

torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

print("\n[1/5] Loading & preprocessing data...")
mne.set_log_level("ERROR")
dataset = MOABBDataset("BNCI2014_001", subject_ids=[CONFIG["subject_id"]])

preprocess(dataset, [
    PickTypes(eeg=True, misc=False),
    Filter(l_freq=4.0, h_freq=38.0),
    Resample(sfreq=CONFIG["sfreq"]),
    SetEEGReference(ref_channels="average"),
    Rescale(scalings=1e-6),
])

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=CONFIG["n_times"],
    window_stride_samples=CONFIG["n_times"],
    preload=True,
)

splits = windows_dataset.split("session")
train_key = test_key = None
for key in splits.keys():
    ks = str(key).lower()
    if "train" in ks:
        train_key = key
    elif "test" in ks:
        test_key = key
if not train_key or not test_key:
    keys = list(splits.keys())
    train_key, test_key = keys[0], keys[1]

train_ds = splits[train_key]
test_ds = splits[test_key]
print(f"Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")

X_train_cv = np.array([train_ds[i][0] for i in range(len(train_ds))])
y_train_cv = np.array([train_ds[i][1] for i in range(len(train_ds))])
X_test = np.array([test_ds[i][0] for i in range(len(test_ds))])
y_test = np.array([test_ds[i][1] for i in range(len(test_ds))])

print("\n[2/5] Setting up models...")
model_configs = {
    "EEGNet": dict(
        n_chans=CONFIG["n_channels"], n_times=CONFIG["n_times"],
        n_outputs=CONFIG["n_classes"], F1=8, F2=16, D=2,
        kernel_length=64, pool1_kernel_size=4, pool2_kernel_size=8,
        drop_prob=CONFIG["dropout"],
    ),
    "ShallowConvNet": dict(
        n_chans=CONFIG["n_channels"], n_times=CONFIG["n_times"],
        n_outputs=CONFIG["n_classes"],
    ),
    "DeepConvNet": dict(
        n_chans=CONFIG["n_channels"], n_times=CONFIG["n_times"],
        n_outputs=CONFIG["n_classes"],
    ),
}

model_classes = {
    "EEGNet": EEGNet,
    "ShallowConvNet": ShallowFBCSPNet,
    "DeepConvNet": Deep4Net,
}

for name, cls in model_classes.items():
    m = cls(**model_configs[name])
    print(f"  {name}: {sum(p.numel() for p in m.parameters()):,} params")

def make_classifier(model_class, model_cfg, valid_ds=None):
    model = model_class(**model_cfg)
    callbacks = [EarlyStopping(patience=CONFIG["patience"], monitor="valid_loss")] if valid_ds else []
    return EEGClassifier(
        model,
        max_epochs=CONFIG["n_epochs"],
        optimizer__lr=CONFIG["lr"],
        optimizer__weight_decay=CONFIG["weight_decay"],
        batch_size=CONFIG["batch_size"],
        device=device,
        callbacks=callbacks,
        train_split=predefined_split(valid_ds) if valid_ds else None,
    )

print("\n[3/5] Cross-validation...")
skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_seed"])
cv_results = {name: [] for name in model_classes}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_cv, y_train_cv)):
    X_tr, X_val = X_train_cv[train_idx], X_train_cv[val_idx]
    y_tr, y_val = y_train_cv[train_idx], y_train_cv[val_idx]
    print(f"\nFold {fold+1}/{CONFIG['n_folds']} | Train:{len(X_tr)} Val:{len(X_val)}")

    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    valid_ds = TensorDataset(X_val_t, y_val_t)

    for model_name in model_classes:
        print(f"  {model_name}...", end=" ")
        clf = make_classifier(model_classes[model_name], model_configs[model_name], valid_ds)
        clf.fit(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        y_pred = clf.predict(X_val_t)
        acc = accuracy_score(y_val, y_pred)
        kappa = cohen_kappa_score(y_val, y_pred)
        cv_results[model_name].append({"accuracy": acc, "kappa": kappa})
        print(f"Acc={acc:.4f}, Kappa={kappa:.6f}")

print("\n[4/5] Final test-set evaluation...")
test_results = {}

for model_name in model_classes:
    print(f"\n  {model_name}...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_cv, y_train_cv, test_size=0.1,
        random_state=CONFIG["random_seed"], stratify=y_train_cv
    )
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).long()
    valid_ds = TensorDataset(X_val_t, y_val_t)
    clf = make_classifier(model_classes[model_name], model_configs[model_name], valid_ds)
    clf.fit(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    y_pred = clf.predict(torch.from_numpy(X_test).float())
    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    test_results[model_name] = {"accuracy": acc, "kappa": kappa}
    print(f"    Test Acc={acc:.4f}, Kappa={kappa:.6f}")

print("\n[5/5] Summary & visualization...")
summary = []
for model_name in model_classes:
    fr = cv_results[model_name]
    if fr:
        accs = [r["accuracy"] for r in fr]
        kappas = [r["kappa"] for r in fr]
        summary.append({
            "Model": model_name,
            "CV Acc (mean)": np.mean(accs),
            "CV Acc (std)": np.std(accs),
            "CV Kappa (mean)": np.mean(kappas),
            "CV Kappa (std)": np.std(kappas),
            "Test Acc": test_results[model_name]["accuracy"],
            "Test Kappa": test_results[model_name]["kappa"],
        })

df = pd.DataFrame(summary)
print("\n" + df.to_string(index=False))

plt.rcParams["font.sans-serif"] = ["Helvetica"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
models = [s["Model"] for s in summary]

ax = axes[0, 0]
acc_m, acc_s = [s["CV Acc (mean)"] for s in summary], [s["CV Acc (std)"] for s in summary]
bars = ax.bar(models, acc_m, yerr=acc_s, capsize=5, alpha=0.7)
ax.set_ylabel("Accuracy"); ax.set_title("CV Accuracy"); ax.set_ylim([0,1])
ax.grid(True, alpha=0.3)
for b, m, s in zip(bars, acc_m, acc_s):
    ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom")

ax = axes[0, 1]
kap_m, kap_s = [s["CV Kappa (mean)"] for s in summary], [s["CV Kappa (std)"] for s in summary]
bars = ax.bar(models, kap_m, yerr=kap_s, capsize=5, alpha=0.7, color="orange")
ax.set_ylabel("Cohen's Kappa"); ax.set_title("CV Kappa"); ax.set_ylim([0,1])
ax.grid(True, alpha=0.3)
for b, m, s in zip(bars, kap_m, kap_s):
    ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom")

ax = axes[1, 0]
tacc = [s["Test Acc"] for s in summary]
tkap = [s["Test Kappa"] for s in summary]
x = np.arange(len(models)); w = 0.35
ax.bar(x-w/2, tacc, w, label="Accuracy", alpha=0.8)
ax.bar(x+w/2, tkap, w, label="Kappa", alpha=0.8, color="green")
ax.set_ylabel("Score"); ax.set_title("Test Set"); ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylim([0,1]); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for mn, fr in cv_results.items():
    if fr:
        ax.plot(range(1,len(fr)+1), [r["accuracy"] for r in fr], marker="o", label=mn)
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy"); ax.set_title("CV per Fold")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(range(1, CONFIG["n_folds"]+1))

plt.tight_layout()
plt.savefig("project_results.png", dpi=150)
plt.close()
df.to_csv("project_results.csv", index=False)
print("\nSaved: project_results.png, project_results.csv")
print("Done!")
