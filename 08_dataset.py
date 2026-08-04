from braindecode.datasets import MOABBDataset
"""
    Subject（被试 / 人）
    └── Session（会话 / 天）
            └── Run（轮次 / 单次采集文件）
                └── Trial（试次 / 单个刺激事件）
"""

dataset = MOABBDataset(
    dataset_name="BNCI2014_001",  # BCI Competition IV 2a
    subject_ids=[1,2],
)


print(type(dataset)) # MOABBDataset/BaseConcatDataset
print(type(dataset.datasets)) # list
print(type(dataset.datasets[0])) # RawDataset,第一个 RawDataset（对应一个 subject+session+run）
print(type(dataset.datasets[0].raw)) # RawArray

# print(dataset)
"""
<BaseConcatDataset | 24 RawDataset(s) | 2321640 total samples>
  Sfreq*: 250.0 Hz
  Channels*: 26 (22 EEG, 3 EOG, 1 STIM)
  Ch. names*: Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, ... (+16 more)
  Montage*: head
  Duration*: 386.9 s
  (* from first recording)
  Description: 24 recordings × 3 columns [subject, session, run]
"""

# print(dataset.datasets)
"""
[<RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=0train, run=0, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=0train, run=1, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=0train, run=2, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=0train, run=3, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=0train, run=4, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=0train, run=5, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=1test, run=0, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=1test, run=1, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=1test, run=2, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=1test, run=3, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=1test, run=4, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=1, session=1test, run=5, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=0train, run=0, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=0train, run=1, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=0train, run=2, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=0train, run=3, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=0train, run=4, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=0train, run=5, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=1test, run=0, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=1test, run=1, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=1test, run=2, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=1test, run=3, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=1test, run=4, <RawDataset | 26 ch (22 EEG, 3 EOG, 1 STIM) | 250.0 Hz | 96735 samples (386.9 s)>
  description: subject=2, session=1test, run=5]
"""




# # pandas.DataFrame
# print(dataset.description)             # BaseConcatDataset
# print(dataset.datasets[0].description) # RawDataset
# print(dataset.datasets[0].raw.info)    # RawArray


splits = dataset.split(by="subject")

print(type(splits)) # dict

subject_1 = splits["1"]
subject_2 = splits["2"]

print(type(subject_1)) # BaseConcatDataset
print(type(subject_2)) # BaseConcatDataset
# print(subject_1.description)
# print(subject_2.description)

# print(dataset.description["subject"]) # 方式二


