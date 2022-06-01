let fileDinhKem = ""

async function duDoan() {
    $("#panelKetQua").show()
    let data = {},
        flag = true
    fullLabel.forEach(label => {
        let idName = "#" + label,
            val = $(idName).val()
        data[label] = val ? parseInt(val) : 0
        if (val === null || val === "")
            flag = false
    })

    if (flag) {
        let xhttp = new XMLHttpRequest();
        xhttp.onload = function () {
            let response = xhttp.responseText,
                kq = JSON.parse(response)
            $("#panelKetQua").show()
            let ketqua = $("#ketqua")
            ketqua.text(phanLop[kq["ketqua"]])
            ketqua.css('color', mauSac[kq["ketqua"]]);
            let phantram = $("#phantram")
            phantram.text(kq["phantram"]);
            phantram.css('color', mauSac[kq["ketqua"]]);
        }
        xhttp.open('POST', '/dudoan', true);
        xhttp.setRequestHeader('Content-Type', 'application/json');
        xhttp.send(JSON.stringify(data));
    } else
        alert("Vui lòng nhập đầy đủ thông tin")

}

async function loadLai() {
    $("#panelKetQua").hide()
    label.forEach(label => {
        let labelName = "#" + label,
            select = $(labelName)
        select.val(null)
    })

}

function onChangeDuongDan(input) {
    fileDinhKem = input.files[0]
}

function onDuDoanFile() {
    if (!fileDinhKem)
        return

    let data = new FormData();
    data.append("file", fileDinhKem, fileDinhKem["name"]);

    let xhttp = new XMLHttpRequest();
    xhttp.onload = function (evt) {
        let response = JSON.parse(xhttp.responseText)
        response = JSON.parse(response)
        let table = $("#kqFile")
        response.forEach((item,index) => {
        console.log(item["Ket_qua"])
            if(item["Ket_qua"]=="Đúng tiến độ"){
                let row = $(`<tr>`)
                fullLabel2.forEach(e=>{
                row.append(`<td >` + item[e] + `</td>`);
                table.append(row)
            })
            }
            else
             {
                console.log(item["Ket_qua"])
                let row = $(`<tr >`)
                fullLabel2.forEach(e=>{
                row.append(`<td ><font color="#FF0000">` + item[e] + `</font></td>`);
                table.append(row)
            })
             }


            // select.formSelect();
        })
    }
    xhttp.open("POST", "/DuDoanFile", true);
    xhttp.send(data);
}
