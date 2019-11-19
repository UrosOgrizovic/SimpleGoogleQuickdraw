from flask import Flask, escape, request, send_file, render_template
import numpy as np
import image_operations
import cv2
from models.vanilla_cnn import vanilla_cnn
from keras.models import load_model
from flask import jsonify, json, make_response, Response
import os

dirname = os.path.dirname(__file__)
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route("/saveimage", methods=['POST'])
def save_image():
    try:
        # print(request.get_json())
        base64Img = request.get_json()
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

        loaded_image = image_operations.load_images('data/img.npy')
        # image_operations.display_image(loaded_image)
        prediction = make_prediction_for_image(loaded_image)
        return app.response_class(response=json.dumps(prediction),
                                  status=200,
                                  mimetype='application/json')
        # print('ok')
        # return 'ok'
    except Exception as e:
        print(e)
        print('fail')
        return app.response_class(response=json.dumps('fail'),
                                  status=400,
                                  mimetype='application/json')

def make_prediction_for_image(image):

    model = load_model(os.path.join(dirname, 'models/vanilla_cnn/vanilla_cnn_model.h5'))

    test_image = np.expand_dims(image, axis=-1)
    max_idx = np.argmax(model.predict(test_image))

    return vanilla_cnn.reverse_labels[max_idx]
if __name__ == "__main__":
    app.run(debug=False, threaded=False)