
from pickle import dump
import numpy as np 
import pandas as pd 
import seaborn as sns
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
import csv
from scipy import stats
# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Đọc file dữ liệu
data = pd.read_csv("dataD17.csv")
data=data.iloc[:,1:13]
#print(data[data.isnull().any(axis=1)].head())
# print(data.info())
columns = list(data)

# Kết quả dữ liệu thu thập được
# print(data)

pd.crosstab(data.DRL_1,data.Ket_qua).plot(kind="bar",figsize=(20,6))
plt.title('Tần suất điểm rèn luyện năm 1 ảnh hưởng đúng hay không đúng tiến độ của sinh viên')
plt.xlabel('Điểm rèn luyện năm 1')
plt.ylabel('Tần số')
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.savefig('DRL_1.png')
# plt.show()
pd.crosstab(data.DRL_2,data.Ket_qua).plot(kind="bar",figsize=(20,6))
plt.title('Tần suất điểm rèn luyện năm 2 ảnh hưởng đúng hay không đúng tiến độ của sinh viên')
plt.xlabel('Điểm rè luyện năm 2')
plt.ylabel('Tần số')
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.savefig('DRL_2.png')

pd.crosstab(data.DRL_3,data.Ket_qua).plot(kind="bar",figsize=(20,6))
plt.title('Tần suất điểm rèn luyện năm 1 ảnh hưởng đúng hay không đúng tiến độ của sinh viên')
plt.xlabel('Điểm rèn luyện năm 3')
plt.ylabel('Tần số')
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.savefig('DRL_13.png')

pd.crosstab(data.DRL_TB,data.Ket_qua).plot(kind="bar",figsize=(20,6))
plt.title('Tần suất điểm rèn luyện trung bình ảnh hưởng đúng hay không đúng tiến độ của sinh viên')
plt.xlabel('Điểm rèn luyện trung bình')
plt.ylabel('Tần số')
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.savefig('DRL_TB.png')

pd.crosstab(data.So_gio_lam_them,data.Ket_qua).plot(kind="bar",figsize=(20,6))
plt.title('Tần suất số giờ làm thêm/tuần ảnh hưởng đúng hay không đúng tiến độ của sinh viên')
plt.xlabel('Số giờ làm thêm/tuần')
plt.ylabel('Tần số')
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.savefig('So_gio_lam_them.png')

pd.crosstab(data.So_mon_chua_hoc,data.Ket_qua).plot(kind="bar",figsize=(20,6))
plt.title('Tần suất môn chưa học ảnh hưởng đúng hay không đúng tiến độ của sinh viên')
plt.xlabel('Số môn chưa học')
plt.ylabel('Tần số')
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.savefig('So_mon_chua_hoc.png')

pd.crosstab(data.So_mon_chua_chua_tra_no,data.Ket_qua).plot(kind="bar",figsize=(20,6))
plt.title('Tần suất số môn chưa trả nợ ảnh hưởng đúng hay không đúng tiến độ của sinh viên')
plt.xlabel('Số môn chưa trả nợ')
plt.ylabel('Tần số')
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.savefig('So_mon_chua_chua_tra_no.png')

labels = [	'DRL_1','DRL_2','DRL_3','DRL_TB','DTBN_1','DTBN_2','DTBN_3','So_gio_lam_them','So_mon_chua_hoc','So_mon_chua_chua_tra_no','DTBTL']
# Tiền xử lý dữ liệu
data = data.fillna(value=0) # điền đầy dử liệu ( thay thế giá trị rỗng bằng giá trị 0)
# print(" Bảng dữ liệu sau khi thay thế giá trị rỗng bằng giá trị 0")
# print(data.info())
# mã hóa dữ liệu
le = LabelEncoder()
dataset = data[labels]
# print(dataset)
dataset = dataset.apply(le.fit_transform)
dataset = pd.DataFrame(dataset,columns=labels)
X = data.drop(columns=labels)
# print(X)
dataset = pd.concat([dataset, X], axis=1)# axis=1 theo cột = 0 theo hàng
# print(dataset)
# print("Bảng dữ liệu sau mã hóa LabelEncoder ")
# print(dataset.info())

# Trích xuất đặt=c trưng
array = dataset.values
P = array[:,0:11]
# print(P)
Q = array[:,11]
# feature extraction
test = SelectKBest(score_func=f_classif, k=11)#k=11 là lựu chọn  đặc trưng
fit = test.fit(P,Q)
# summarize scores
set_printoptions(precision=3)
# print(fit.scores_)
features = fit.transform(P)
# summarize selected features
print("features:")
print(features[:8,:])
print("Tuong quan du lieu")
print("############# AVEARGE, MAX, MIN, MEAN, MEDIAN, MODE, RANGE ##############")
print(dataset.describe())
info = dataset.describe()

AX = dataset.iloc[:,0]
AY =dataset.iloc[:,11]
SP= stats.spearmanr(AX,AY)

print("Bảng tương quan các yếu tố Pearson")
# Độ tương quan, các yếu tố ảnh hưởng
print(dataset.corr(method="pearson")["Ket_qua"])
plt.figure(figsize=(20,10))
sns.heatmap(dataset.corr(method="pearson"), annot=True)
plt.title("Correlation pearson")
plt.xticks(rotation=0)
plt.savefig('Pearson.png')


print("Bảng tương quan các yếu tố spearman")
print(dataset.corr(method="spearman")["Ket_qua"])
plt.figure(figsize=(20,10))
sns.heatmap(dataset.corr(method="spearman"), annot=True)
plt.title("Correlation spearman")
plt.xticks(rotation=30)# tên các trường hiển thị trên truc x nghiêng 30 độ
plt.yticks(rotation=0)# tên các trường hiển thị trên truc y nghiêng 0 độ
plt.savefig('Spearman.png')

plt.figure(figsize=(15,8))
sns.countplot(y="DTBN_1", hue="Ket_qua", data=data)
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.title("Biểu đồ ảnh hưởng của điểm trung bình năm 1 đến đúng hay không đúng tiến độ tốt nghiệp")
plt.savefig('AnhHuongDTBN_1.png')

plt.figure(figsize=(15,8))
sns.countplot(y="DTBN_2", hue="Ket_qua", data=data)
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.title("Biểu đồ ảnh hưởng của điểm trung bình năm 2 đến đúng hay không đúng tiến độ tốt nghiệp")
plt.savefig('AnhHuongDTBN_2.png')

plt.figure(figsize=(15,8))
sns.countplot(y="DTBN_3", hue="Ket_qua", data=data)
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.title("Biểu đồ ảnh hưởng của điểm trung bình năm 3 đến đúng hay không đúng tiến độ tốt nghiệp")
plt.savefig('AnhHuongDTBN_3.png')

plt.figure(figsize=(15,8))
sns.countplot(y="DTBTL", hue="Ket_qua", data=data)
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'])
plt.title("Biểu đồ ảnh hưởng của điểm trung bình Tích Lũy đến đúng hay không đúng tiến độ tốt nghiệp")
plt.savefig('AnhHuongDTBTL.png')

plt.figure(figsize=(15,8))
data["Ket_qua"].value_counts().plot(kind='bar', color=['#ffa180', '#ffb865'])
plt.legend(["Không đúng tiến độ",'Đúng tiến độ'],loc ="lower right")
plt.xlabel("Số lượng")
plt.ylabel("Kết quả")
plt.savefig('Ket_qua.png')




# dự đoán
print("**************** X ****************")
X = dataset.drop(columns=["Ket_qua"])
X=dataset.iloc[:,0:11]
y=dataset.iloc[:,11]
print("**************** y ****************")
# Mô hình không có  kfold
# print("Mô hình không kfold")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.35, random_state=0)
models = [
          # GaussianNB(),
          DecisionTreeClassifier(),
          LogisticRegression(),
          # RandomForestClassifier(),
          KNeighborsClassifier(),
          SVC(),
          ExtraTreeClassifier()
]

temp = []
temp1 = []
for  model in models:
  print (model)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print("######################################################################")
  print(type(model).__name__, accuracy_score(y_test,y_pred),roc_auc_score(y_test,y_pred))
  temp.append(accuracy_score(y_test, y_pred))
  temp1.append(roc_auc_score(y_test, y_pred))


# Vẽ biểu đồ so sánh độ chính xác Accuracy và sai số MSE giữa các mô hình
labels = [ 'DecisionTree', 'Logistic',"KNN","SVM","Extra"]
x = np.arange(len(labels))  # the label locations
fig, ax = plt.subplots()
rects1 = ax.bar(x, temp, label='Accuracy')
rects2 = ax.bar(x, temp1, label='AUC')
ax.set_xlabel('Mô hình')
ax.set_ylabel('Kết quả')
ax.set_title('So sánh Accuracy, và AUC các mô hình test/train: 35/65')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.savefig('SoSanhCacModell.png')
# plt.show()



print("################ Lưu model cây quyết định ##################")
dump(DecisionTreeClassifier(), open('CAY_QUYETDINH.sav', 'wb'))
# Mô hình có  kfold
print("********************* Mô hình có  kfold *************************")
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, random_state=0)
models = []

models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
models.append(('LogisticRegression', LogisticRegression()))
models.append(('KNeighborsClassifier', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('ExtraTreeClassifier', ExtraTreeClassifier()))

# evaluate each model in turn
results = []
names = []
tam=[]
tam1=[]
print("Accuracy")
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=None,shuffle=False)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("######################################################################")
    print(name, cv_results.mean(), cv_results.std())
    tam.append(cv_results.mean())
    tam1.append(cv_results.std())
# Vẽ biểu đồ so sánh độ chính xác Accuracy và sai số STD giữa các mô hình
labels = ['DecisionTree', 'Logistic',"KNN","SVM","Extra"]
x = np.arange(len(labels))  # the label locations
fig, ax = plt.subplots()
rects1 = ax.bar(x, tam, label='Accuracy')
rects2 = ax.bar(x, tam1, label='AUC')
ax.set_xlabel('Mô hình')
ax.set_ylabel('Kết quả')
ax.set_title('So sánh Accuracy, và AUC các mô hình có kfold=5')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.savefig('SoSanhCacModellKfold_5.png')
# plt.show()

results = []
names = []
tam=[]
tam1=[]
print("Accuracy và AUC")
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=None,shuffle=False)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,cv=kfold, scoring='roc_auc')
    cv_results1 = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("######################################################################")
    print(name, cv_results.mean(), cv_results1.mean())
    tam.append(cv_results.mean())
    tam1.append(cv_results1.mean())
# Vẽ biểu đồ so sánh độ chính xác AUC và sai số STD giữa các mô hình
labels = ['DecisionTree', 'Logistic',"KNN","SVM","Extra"]
x = np.arange(len(labels))  # the label locations
fig, ax = plt.subplots()
rects1 = ax.bar(x, tam, label='Accuracy')
rects2 = ax.bar(x, tam1, label='AUC')
ax.set_xlabel('Mô hình')
ax.set_ylabel('Kết quả')
ax.set_title('So sánh Accuracy, và AUC các mô hình có kfold=10')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
# plt.figure(figsize=(15,8))
fig.tight_layout()
plt.savefig('SoSanhCacModellKfold_10.png')
# plt.show()
## Lưu mô hình cây quyết định làm dự đoán
CART=DecisionTreeClassifier()
CART.fit(X_train,Y_train)
predictions=CART.predict(X_validation)
print(CART.predict([[83,85,86,48.7,7.06,7.48,7.99,20,0,0,7.75]]))
dump(CART,open('CAY_QUYETDINH11.sav','wb'))
LR=LogisticRegression()
LR.fit(X_train,Y_train)
predictions=LR.predict(X_validation)
print(LR.predict([[83,85,86,48.7,7.06,7.48,7.99,20,0,0,7.75]]))
dump(LR,open('LR_HOIQUY11.sav','wb'))