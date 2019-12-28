var canvas = document.getElementById("paintArea");
var ctx = canvas.getContext("2d");
//var BB = canvas.getBoundingClientRect();
resize();

// resize canvas when window is resized
function resize() {

//  ctx.canvas.width = window.innerWidth;
//  ctx.canvas.height = window.innerHeight;
    ctx.canvas.width
}

// add event listeners to specify when functions should be triggered
window.addEventListener("resize", resize);
document.addEventListener("mousemove", draw);
document.addEventListener("mousedown", setPosition);
document.addEventListener("mouseenter", setPosition);
document.getElementById("clearCanvas").addEventListener("click", clearCanvas);
document.getElementById("submitDrawing").addEventListener("click", submitDrawing);
document.getElementById("displayImg").addEventListener("click", displayImg);

// last known position
var pos = { x: 0, y: 0 };

var classes = ['airplane', 'alarm clock', 'ant', 'axe', 'bicycle', 'The Mona Lisa'];

// new position from mouse events
function setPosition(e) {
    var BB=canvas.getBoundingClientRect();
    pos.x = e.clientX - BB.left;
    pos.y = e.clientY - BB.top;
}

function draw(e) {
  if (e.buttons !== 1) return; // if mouse is pressed.....

  ctx.beginPath(); // begin the drawing path

  ctx.lineWidth = 10; // width of line
  ctx.lineCap = "round"; // rounded end cap
  ctx.strokeStyle = '#000000'; // hex color of line

  ctx.moveTo(pos.x, pos.y); // from position
  setPosition(e);
  ctx.lineTo(pos.x, pos.y); // to position

  ctx.stroke(); // draw it!
}

function resetTable() {
    $('#probTable tbody').empty();
//    for (var cl of classes) {
//
//        document.getElementById(cl).innerHTML = 0;
//        document.getElementById(cl).style.backgroundColor = 'white';
//    }
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    resetTable();
}

function displayVanillaCNNPredictions(tbody, prediction, probs, model_name) {
    var newRow = tbody.insertRow(tbody.rows.length);

    var to_append = "<tr><td>"+model_name+"</td>";
    for (var img_class of classes) {
        if (prediction == img_class) {
            to_append += "<td style='background-color: yellow;'>"+probs[img_class]+"</td>";
        } else {
            to_append += "<td>"+probs[img_class]+"</td>";
        }

    }
    to_append += "</tr>";
    newRow.innerHTML = to_append;
}

function displaySVMPredictions(tbody, prediction, model_name) {
    var newRow = tbody.insertRow(tbody.rows.length);
    var to_append = "<tr><td>"+model_name+"</td>";
    for (var img_class of classes) {
        if (prediction == img_class) {
            to_append += "<td style='background-color: yellow;'>1</td>";
        } else {
            to_append += "<td>0</td>";
        }

    }
    to_append += "</tr>";
    newRow.innerHTML = to_append;
}

function displayPredictions(tbody, prediction, probs, model_name) {
    var newRow = tbody.insertRow(tbody.rows.length);
    var to_append = "<tr><td>"+model_name+"</td>";

    // if probs isn't {}
    if (!(Object.keys(probs).length === 0 && probs.constructor === Object)) {
        for (var img_class of classes) {
            if (prediction == img_class) {
                to_append += "<td style='background-color: yellow;'>"+probs[img_class]+"</td>";
            } else {
                to_append += "<td>"+probs[img_class]+"</td>";
            }
        }
    }
    else {
        for (var img_class of classes) {
            if (prediction == img_class) {
                to_append += "<td style='background-color: yellow;'>1</td>";
            } else {
                to_append += "<td>0</td>";
            }
        }
    }
    to_append += "</tr>";
    newRow.innerHTML = to_append;
}

function submitDrawing() {
    resetTable();
    // here is the most important part because if you dont replace you will get a DOM 18 exception.
    var image_base64 = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
    $.ajax({
        type: 'POST',
        url: '/saveimage',
        data: JSON.stringify(image_base64),
        dataType: 'json',
        contentType: 'application/json;charset=UTF-8',
        success: function(obj) {
            max_class = "";
            max_prob = 0;
            var tbody = document.getElementById('probTable').getElementsByTagName('tbody')[0];

            displayPredictions(tbody, obj.prediction, obj.probabilities, 'Vanilla CNN 10k');
            displayPredictions(tbody, obj.vanilla_cnn_100k_prediction, obj.vanilla_cnn_100k_probabilities, 'Vanilla CNN 100k');
            displayPredictions(tbody, obj.SVM2k_prediction, {}, 'SVM 2k');
            displayPredictions(tbody, obj.SVM10k_prediction, {}, 'SVM 10k');
            displayPredictions(tbody, obj.VGG19_10k_prediction, obj.VGG19_10k_probabilities, 'VGG19_10k');
        }
    });
}

function displayImg() {
    $.ajax({
        type: 'GET',
        url: '/getimage',
        contentType: 'application/json;charset=UTF-8'
    });
}