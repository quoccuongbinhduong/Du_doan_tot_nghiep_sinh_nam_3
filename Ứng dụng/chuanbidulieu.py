import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import joblib
from xgboost import XGBClassifier
from Config import data_link, data_label_link, model_link, labels, className, className_t


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


data = pd.read_csv(data_link)
data = data.iloc[:,1:13]
# print(data)

# Tiền xử lý dữ liệu

data = data.fillna(value=0)
data.to_csv(data_link)

le = LabelEncoder()
dataset = data[labels]
# print(data)
dataset = pd.DataFrame(dataset, columns=labels)

data_label = {}

for label in labels:

    dataset[label] = le.fit_transform(data[label])
    data_label[label] = [{"key": i, "value": l} for i, l in enumerate(le.classes_)]

with open(data_label_link, "w") as outfile:
    json.dump(data_label, outfile, cls=NpEncoder)

X = data.drop(columns=className_t)
# print(X)
# dataset = pd.concat([dataset, X], axis=1)
# print(dataset)
# X = dataset.drop(columns=className)

print(X)
y = data[className]
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

model = LogisticRegression()
# print(y_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prod = model.predict_proba(X_test)[:, 1]
print(type(model).__name__, accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prod))
joblib.dump(model, model_link)
print("Thanh cong")
