from flask import Flask, escape, request, send_file, render_template
import numpy as np
import image_operations
import cv2
from models.vanilla_cnn import vanilla_cnn
from tensorflow.keras.models import load_model
from tensorflow import cast, float32
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
        vanilla_cnn_prediction, probs = make_prediction_for_image(loaded_image,
                                                                  'models/vanilla_cnn/vanilla_cnn_model_10k.h5')
        to_return = {'prediction': vanilla_cnn_prediction, 'probabilities': probs}
        return app.response_class(response=json.dumps(to_return),
                                  status=200,
                                  mimetype='application/json')
        # print('ok')
        # return 'ok'
    except Exception as e:

        print(e)
        print('fail')
        return app.response_class(response=json.dumps('fail'),
                                  status=500,
                                  mimetype='application/json')

@app.route("/getimage", methods=['GET'])
def get_image():
    try:
        loaded_image = image_operations.load_images('data/img.npy')
        loaded_image = np.squeeze(loaded_image)
        image_operations.display_image(loaded_image)
        return 'ok'
    except Exception as e:
        print(e)
        print('fail')
        return app.response_class(response=json.dumps('fail'),
                                  status=500,
                                  mimetype='application/json')

def make_prediction_for_image(image, path_to_model):
    model = load_model(os.path.join(dirname, path_to_model), compile=False)
    test_image = np.expand_dims(image, axis=-1)
    max_idx = np.argmax(model.predict(test_image))
    to_return_probs = {'airplane': 0, 'alarm clock': 0, 'axe': 0, 'The Mona Lisa': 0,
          'bicycle': 0, 'ant': 0}
    predicted_probs = model.predict(test_image).tolist()[0]
    to_return_probs['airplane'] = predicted_probs[0]
    to_return_probs['alarm clock'] = predicted_probs[1]
    to_return_probs['axe'] = predicted_probs[2]
    to_return_probs['The Mona Lisa'] = predicted_probs[3]
    to_return_probs['bicycle'] = predicted_probs[4]
    to_return_probs['ant'] = predicted_probs[5]
    return vanilla_cnn.reverse_labels[max_idx], to_return_probs


if __name__ == "__main__":
    app.run(debug=False, threaded=False)