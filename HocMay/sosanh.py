

from pickle import dump
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
# Đọc file dữ liệu
import json
data = pd.read_csv("dataD17.csv")
data=data.iloc[:,1:13]
from sklearn.tree import ExtraTreeClassifier
labels = [	'DRL_1','DRL_2','DRL_3','DRL_TB','DTBN_1','DTBN_2','DTBN_3','So_gio_lam_them','So_mon_chua_hoc','So_mon_chua_chua_tra_no','DTBTL']
# Tiền xử lý dữ liệu
data = data.fillna(value=0) # điền đầy dử liệu ( thay thế giá trị rỗng bằng giá trị 0)
le = LabelEncoder()
dataset = data[labels]
# print(dataset)
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



dataset = dataset.apply(le.fit_transform)
dataset = pd.DataFrame(dataset,columns=labels)
X = data.drop(columns=labels)
dataset = pd.concat([dataset, X], axis=1)# axis=1 theo cột = 0 theo hàng
import xgboost
from sklearn.tree import DecisionTreeClassifier
import shap


# train XGBoost model
X=dataset.iloc[:,0:11]
y=dataset.iloc[:,11]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35, random_state=0)
model = DecisionTreeClassifier().fit(X_train, y_train)
# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
# import matplotlib.pyplot as plt

from sklearn import tree
# import graphviz
# target = list(y.unique())
# feature_names = list(X.columns)
# dot_data = tree.export_graphviz(model,
#                                 out_file=None,
#                       feature_names=feature_names,
#                       class_names=target,
#                       filled=True, rounded=True,
#                       special_characters=True)
# graph = graphviz.Source(dot_data)
#
# graph

shap.plots.beeswarm(shap_values[:,:,0])
