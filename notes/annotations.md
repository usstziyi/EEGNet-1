## `annotations` 和 `stim` 的关系

这是 MNE 中表示「事件/刺激标记」的**两种形式**，可以互相转换。

---

### 1. 先分别是什么

| 名称 | 是什么 | 存在位置 |
|------|--------|---------|
| **stim 通道** | 一根**物理/虚拟的数据通道**（和 C3、Cz 这些 EEG 通道平级），取值几乎全是 0，只在事件发生的那个采样点上跳出一个非零整数（事件编码） | 存储在 `.gdf` / `.edf` 等原始数据文件里，读入后是 `raw.ch_names` 中的 `STI 014`、`stim` 等通道 |
| **annotations** | MNE 定义的一个**软件层数据结构**（`mne.Annotations` 类），存三份列表：`onset`（秒）、`duration`（秒）、`description`（标签） | 不对应通道，是 `raw.annotations` 这个属性，也可以手动构造（就像你第227-230行做的） |

---

### 2. 两者的核心转换关系

```
stim 通道（采样点维度，整数编码）
       │
       │  mne.find_events(raw)    ← 从 stim 通道「抠出」非零点 → 事件数组 (N, 3)
       ▼
  事件数组 events shape=(N, 3)
  列含义：[起始采样点, 上一个事件值, 当前事件值]
       │
       │  mne.annotations_from_events(events, sfreq, ...)
       ▼
  annotations 对象
  (onset:秒, duration:秒, description:str/int)
       │
       │  raw.set_annotations(annotations)    ← 挂到 Raw 上
       ▼
  raw.annotations （就能给 create_windows_from_events 用了）
```

反过来也行：

```
annotations
       │
       │  mne.events_from_annotations(raw)
       ▼
  事件数组 events (N, 3)
       │
       │  把事件写回一根 stim 通道
       ▼
  stim 通道
```

---

### 3. stim 通道长啥样？

以一个采样率 250 Hz、时长 16 秒的 Raw 为例：

```
通道名:  STI 014（stim 通道）
采样点:  0  250  500  750 1000 1250 1500 1750 2000 2250 2500 ... 4000
取值:    0    0    0    0    1    0    0    0    2    0    0 ...    0
                       ↑                    ↑
                  第4秒事件码=1         第8秒事件码=2
```

**stim 通道是离散的、基于采样点的**：除了事件那一瞬间是事件码（1、2、769、770…），其他时刻全是 0。

对应转换成 annotations 就变成：

```python
onsets       = [4.0, 8.0]            # 秒
durations    = [0.0, 0.0]            # 0 表示瞬间事件
descriptions = [1,   2  ]            # 标签
annotations  = mne.Annotations(onsets, durations, descriptions)
```

---

### 4. 为什么有两种形式？

**历史原因 + 灵活性：**

| 形式 | 适合场景 | 优缺点 |
|------|---------|--------|
| **stim 通道** | 原始文件存储（GDF/EDF/BCI2000 .dat） | ✅ 随 EEG 一起存盘，不会丢<br>❌ 只能在采样点上打标记，精度受限；无法存非整数标签； duration 不好表达 |
| **annotations** | MNE 软件内部处理（打坏段、写注释、事件扩展） | ✅ 单位是秒（浮点精度）；description 可以是字符串（如 `"bad_segment"`、`"left_hand"`）；duration 可以 >0 表达一段区间<br>❌ 是 MNE 自己的结构，写回原文件要额外操作 |

**现代推荐：** 在 MNE 里主要用 **annotations**，stim 通道只是读文件时的信息来源之一。

---

### 5. 你的代码里用的是哪一种？

**代码引用：** [第227-231行手动构造 annotations](file:///Users/usst_ziyi/Programs/trae/DeepL/EEGNet/02_eeg_data.py#L227-L231)

因为 `raw_reref` 是你用 `mne.io.RawArray` **模拟生成**的，数据中根本没有 stim 通道（只有 EEG 通道），所以你走的是**「直接构造 annotations」**这条路径：

```
没有 stim 通道的模拟 Raw
    │
    │  第227~230行：手动算 onsets、造标签 → annotations
    │  第231行：    raw.set_annotations(annotations)
    ▼
raw_annotated（带上了 annotations，可以被 create_windows_from_events 识别）
```

如果加载的是**真实 GDF 文件**（比如 Schirrmeister2017 下载下来的），流程是反过来：

```
真实 .gdf 文件（自带 STI 014 stim 通道）
    │
    │  mne.io.read_raw_gdf() 读取
    ▼
raw 对象 → raw 里有 stim 通道
    │
    │  mne.find_events(raw)  从 stim 通道提取事件数组
    │  或者  moabb/braindecode 内部自动转成 annotations
    ▼
带 annotations / events 的 Raw → 继续处理
```

---

### 6. 一句话总结

> **stim 通道是「存储层」的事件编码**（文件里一根通道、采样点跳非零）；
> **annotations 是「应用层」的事件表达**（MNE 里的结构化对象，秒为单位，有标签和时长）。
> 两者通过 MNE 的 `find_events` / `events_from_annotations` / `annotations_from_events` 互相转换。