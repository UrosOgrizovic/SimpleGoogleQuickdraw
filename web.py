from flask import Flask, escape, request, send_file, render_template
import image_operations
import numpy as np
from skimage import io, color


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
        # (400, 400, 4) -> (400, 400)
        img_gray = img_array[:, :, 3]
        # (400, 400) -> 160000 TODO convert to 1D array
        img_final = np.ravel(img_gray)
        image_operations.save_as_image('data/img.npy', img_gray)

        # loaded_image = image_operations.load_images('data/img.npy')[0]
        loaded_image = image_operations.load_image('data/img.npy')
        print(loaded_image.shape)
        # not working
        image_operations.display_image(loaded_image.reshape(28, 28))
        # image_operations.display_image(loaded_image)
        print('ok')
        return 'ok'
    except Exception as e:
        image_operations.load_images('data/full_numpy_bitmap_axe.npy')
        print(e)
        print('fail')
        return 'fail'

if __name__ == "__main__":
    app.run(debug=True)