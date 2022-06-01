import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from xgboost import XGBClassifier
from Config import className, data_label_link, fullLabel2, mapClassName,data_link, data_label_mapping,tileName
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
data = pd.read_csv(data_link)
data = data.iloc[:, 1:]
y_train = data[className]
nRow = len(data)
data = data[fullLabel2]
# print(data)
labels = []
f = open(data_label_link)
dataDict = json.load(f)
f.close()
# print(dataDict)
dtMapping = {}
# dataDict={"ketqua": "0", \
#             "phantram": "0"}
for key in dataDict:
    # print(key)
    ds = {int(x["key"]): x["value"] for x in dataDict[key]}
    dtMapping[key] = ds

def duDoanFile(link_file):
    inputData = pd.read_csv(link_file)
    dataTest = inputData.iloc[:, 1:]
    # print(dataTest)
    dataset = pd.concat([data, dataTest])
    dataset = dataset.apply(LabelEncoder().fit_transform)
    X_train = dataset.iloc[:nRow]
    # print(X_train)
    X_test = dataset.iloc[nRow:]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prod = model.predict_proba(X_test)
    # for y in y_prod:
    #     print(y)
    phan_tram = [round(np.max(y)*100,2) for y in y_prod]
    # print(phan_tram)
    y_pred = [mapClassName[x] for x in y_pred]
    dataTest[className] = y_pred
    dataTest[tileName] = phan_tram
    # print(dataTest[className])
    # print(accuracy_score(dataTest[className], y_pred))
    for lab in labels:
        mapLab = {}
        for item in dataDict[lab]:
            mapLab[item["key"]] = item["value"]
            # print(item)
        dataTest[lab] = dataTest[lab].replace(mapLab)
        # print(dataTest[lab])
    for lab in dtMapping:
        # print(lab)
        dataTest[lab] = dataTest[lab].replace(dtMapping[lab])
        # print(dataTest[lab])
    dataTest = dataTest.to_json(orient="records")
    print(dataTest)
    return dataTest
# duDoanFile(("dataD18.csv"))