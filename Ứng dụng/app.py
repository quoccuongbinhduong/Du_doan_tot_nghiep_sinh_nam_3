import uuid
from DuDoanFile import duDoanFile
from flask import Flask, render_template, jsonify, request
import json
from Config import data_label_link, full_labels, model_link, tuongquan_link, fullLabel2
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import json
app = Flask(__name__)

@app.route("/")
def main():
    return render_template("Wellcome.html")

@app.route("/ChucNangPhanTich")
def ChucNangPhanTich():
    return render_template("PhanTichDuLieu.html")

@app.route("/ChucNangTuongQuan")
def ChucNangTuongQuan():
    return render_template("KhaoSatTuongQuan.html")

@app.route("/ChucNangMoHinh")
def ChucNangMoHinh():
    return render_template("XayDungMoHinh.html")

@app.route("/ChucNangDuDoan")
def ChucNangDuDoan():
    return render_template("index.html")

@app.route('/dudoan', methods=['POST'])
def duDoan():
    dtMapping = {}
    dataDict={"ketqua": "0", \
                "phantram": "0"}
    x = []
    # print(dtMapping)
    for label in fullLabel2:
        x.append(request.get_json()[label])
    x = np.asarray([x])
    model = joblib.load(model_link)
    y_pred = model.predict(x)
    y_prod = model.predict_proba(x)
    phan_tram = np.max(y_prod)
    dataDict['ketqua'] = str(y_pred[0])
    dataDict['phantram'] = round(phan_tram*100,2)
        # print(dataTest)
    # dataDict = dataDict.to_json(orient="records")
    print(dataDict)
    return dataDict

@app.route('/getdata')
def getData():
    f = open(data_label_link)
    data = json.load(f)
    f.close()
    return jsonify(data)

@app.route('/gettuongquan')
def getTuongQuan():
    f = open(tuongquan_link)
    data = json.load(f)
    f.close()
    print(data)
    return jsonify(data)

@app.route('/getTyLe')
def getTyLe():
    f = open("./Dataset/Tyle.json")
    data = json.load(f)
    f.close()
    return jsonify(data)

@app.route('/getKFold')
def getKFold():
    f = open("./Dataset/KFold.json")
    data = json.load(f)
    f.close()
    return jsonify(data)

@app.route('/DuDoanFile', methods=['POST'])
def DuDoanFile():
    f = request.files['file']
    filename = uuid.uuid1()
    filename = f"{filename}.csv"
    link_file = f"KetQuaFile/{filename}"
    f.save(link_file)
    data = duDoanFile(link_file)
    return jsonify(data)

if __name__ == "__main__":
    app.run()
    # getTuongQuan()
