import pandas as pd
from Config import className, data_label_link, full_labels, mapClassName,data_link, data_label_mapping
from random import randint
import json


dataset = pd.read_csv(data_link)
nRow = len(dataset)
dataset = dataset.iloc[:, 1:]
k = round(len(dataset) / 20)


colDataset = full_labels
colDataset.append(className)

# for lab in labels:
#     mapLab = {}
#     for item in data[lab]:
#         mapLab[item["key"]] = item["value"]
#     dataset[lab] = dataset[lab].replace(mapLab)

# dataset[className] = dataset[className].replace(mapClassName)

dataset = dataset[colDataset]
# dataset.to_csv("./Dataset/DatasetFile.csv")

X = dataset.drop(columns=[className])
y = dataset[className]
iLoc = []

for i in range(25):
    iLoc.append(randint(0, nRow))

X.iloc[iLoc].to_csv("FileTestDuDoan.csv")





