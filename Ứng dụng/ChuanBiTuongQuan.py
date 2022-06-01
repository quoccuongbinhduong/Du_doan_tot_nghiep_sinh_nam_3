import json
import pandas as pd
from Config import className, labels_name, full_labels, data_link, tuongquan_link
from sklearn.preprocessing import LabelEncoder

filter_label = full_labels
filter_label.append(className)

data = pd.read_csv(data_link)
data = data.iloc[:, 1:]
data = data.fillna(value=0)
data = data[filter_label]
data = data.apply(LabelEncoder().fit_transform)

filter_label = list(filter(lambda x: x != className, filter_label))

pearson = data.corr(method="pearson")[className].values
pearson = list(filter(lambda x: x < 1, pearson))

spearman = data.corr(method="spearman")[className].values
spearman = list(filter(lambda x: x < 1, spearman))

tuongQuan = []


for index, label in enumerate(filter_label):
    tq = {"name": labels_name[label], "pearson": pearson[index], "spearman": spearman[index]}
    tuongQuan.append(tq)

with open(tuongquan_link, "w") as outfile:
    json.dump(tuongQuan, outfile)
