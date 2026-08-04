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


