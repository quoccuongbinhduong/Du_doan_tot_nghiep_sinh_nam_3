label = ['DTBN_1','DTBN_2','DTBN_3','So_gio_lam_them','So_mon_chua_hoc','So_mon_chua_chua_tra_no','DTBTL'],
fullLabel = ['DRL_1','DRL_2','DRL_3','DRL_TB','DTBN_1','DTBN_2','DTBN_3','So_gio_lam_them','So_mon_chua_hoc','So_mon_chua_chua_tra_no','DTBTL'],
fullLabel2 = ['DTBN_1','DTBN_2','DTBN_3','So_gio_lam_them','So_mon_chua_hoc','So_mon_chua_chua_tra_no','DTBTL','Ket_qua','Ti_le'],
phanLop = {0: "Không đúng tiến độ", 1: "Đúng tiến độ"},
ketqua = ['ketqua','phantram'],
mauSac = {1: "#4CAF50", 0: "#FF5722"};


$(document).ready(function () {
    loadDuLieu()
    loadTuongQuan()
    loadKFoldMau()
    loadTyLeMau()
});


function loadDuLieu() {
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function (evt) {
        let response = JSON.parse(xhttp.responseText)
        // document.getElementById("demo").innerHTML = this.responseText;
        labels.forEach(label => {
            let labelName = "#" + label,
                select = $(labelName)
            response[label].forEach(item => {
                select.append(new Option(item["value"], item["key"]))
            })
            // select.formSelect();
        })
    }
    xhttp.open("GET", "/getdata", true);
    xhttp.send();
}

function loadTuongQuan() {
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function (evt) {
        let response = JSON.parse(xhttp.responseText)
        let table = $("#tqTable")
        response.forEach((item, index) => {
            let row = $(`<tr>`)
            row.append(`<td>` + item["name"] + `</td>`)
            row.append(`<td>` + item["pearson"] + `</td>`)
            row.append(`<td>` + item["spearman"] + `</td>`)
            table.append(row)
            console.log(table)
            // select.formSelect();
        })
    }
    xhttp.open("GET", "/gettuongquan", true);
    xhttp.send();
}

function loadTyLeMau() {
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function (evt) {
        let response = JSON.parse(xhttp.responseText)
        let table = $("#tyleTable")
        response.forEach((item, index) => {
            let row = $(`<tr>`)
            row.append(`<td>` + item["Tyle"] + `</td>`)
            row.append(`<td>` + item["PhuongPhap"] + `</td>`)
            row.append(`<td>` + item["Accuracy"] + `</td>`)
            row.append(`<td>` + item["AUC"] + `</td>`)
            table.append(row)
            // select.formSelect();
        })
    }
    xhttp.open("GET", "/getTyLe", true);
    xhttp.send();
}

function loadKFoldMau() {
    const xhttp = new XMLHttpRequest();
    xhttp.onload = function (evt) {
        let response = JSON.parse(xhttp.responseText)
        let table = $("#kFoldTable")
        response.forEach((item, index) => {
            let row = $(`<tr>`)
            row.append(`<td>` + item["K"] + `</td>`)
            row.append(`<td>` + item["PhuongPhap"] + `</td>`)
            row.append(`<td>` + item["Accuracy"] + `</td>`)
            row.append(`<td>` + item["AUC"] + `</td>`)
            table.append(row)
            // select.formSelect();
        })
    }
    xhttp.open("GET", "/getKFold", true);
    xhttp.send();
}
