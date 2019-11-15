from flask import Flask, escape, request, send_file, render_template
import numpy as np
import image_operations
import cv2

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
        # 3D -> 2D
        img_gray = img_array[:, :, 3]
        '''
        1. resize to 28x28
        2. flatten to 1, 784, so that image_operations.load_images() can do the reshape
        '''
        img_gray = np.ravel(cv2.resize(img_gray, dsize=(28, 28), interpolation=cv2.INTER_CUBIC))[np.newaxis]
        image_operations.save_as_image('data/img.npy', img_gray)

        loaded_image = image_operations.load_images('data/img.npy')[0]
        image_operations.display_image(loaded_image)
        print('ok')
        return 'ok'
    except Exception as e:
        print(e)
        print('fail')
        return 'fail'

if __name__ == "__main__":
    app.run(debug=True)