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

// last known position
var pos = { x: 0, y: 0 };

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

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
}

function submitDrawing() {
    // here is the most important part because if you dont replace you will get a DOM 18 exception.
    var image_base64 = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
//    $.post( "/saveimage", {
//        javascript_data: image_base64,
//        contentType: 'application/json',
//        success: function(prediction) {
//            console.log(prediction);
//        }, error: function(err) {
//            console.log(err);
//        }
//    });
    $.ajax({
        type: 'POST',
        url: '/saveimage',
        data: JSON.stringify(image_base64),
        dataType: 'json',
        contentType: 'application/json;charset=UTF-8',
        success: function(prediction) {
            alert(prediction);
        }
    });
}