from flask import Flask, escape, request, send_file, render_template
import image_operations
import numpy as np



app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route("/saveimage", methods=['POST'])
def save_image():
    try:
        base64Img = request.form['javascript_data']
        imgdata = image_operations.decode_base64(base64Img)
        img_array = image_operations.convert_base64_to_numpy_array(imgdata)

        # image_operations.save_as_image('data/img.npy', img_array)
        # image_operations.display_image(img_array)

        print('ok')
        return 'ok'
    except:
        print('fail')
        return 'fail'

if __name__ == "__main__":
    app.run(debug=True)