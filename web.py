from flask import Flask, escape, request, send_file, render_template
import numpy as np
import image_operations
import cv2
from models.vanilla_cnn import vanilla_cnn
from tensorflow.keras.models import load_model
from tensorflow import cast, float32
from flask import jsonify, json, make_response, Response
import os
from models.SVM import SVM
from models.transfer_learning import transfer_learning

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
        vanilla_cnn_10k_prediction, vanilla_cnn_10k_probs = vanilla_cnn.make_prediction_for_image(loaded_image,
                                                                  'vanilla_cnn_model_10k.h5')
        vanilla_cnn_100k_prediction, vanilla_cnn_100k_probs = vanilla_cnn.make_prediction_for_image(loaded_image,
                                                                                      'vanilla_cnn_model_100k.h5')
        svm2k_prediction = SVM.make_prediction_for_image(loaded_image, 'SVM_2k.joblib')
        svm10k_prediction = SVM.make_prediction_for_image(loaded_image, 'SVM_10k.joblib')
        vgg19_10k_prediction, vgg19_10k_probs = transfer_learning.make_prediction_for_image(loaded_image,
                                                                                            'VGG19_10k.h5')

        to_return = {'prediction': vanilla_cnn_10k_prediction, 'probabilities': vanilla_cnn_10k_probs,
                     'vanilla_cnn_100k_prediction': vanilla_cnn_100k_prediction,
                     'vanilla_cnn_100k_probabilities': vanilla_cnn_100k_probs,
                     'SVM2k_prediction': svm2k_prediction, 'SVM10k_prediction': svm10k_prediction,
                     'VGG19_10k_prediction': vgg19_10k_prediction,
                     'VGG19_10k_probabilities': vgg19_10k_probs}
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




if __name__ == "__main__":
    app.run(debug=False, threaded=False)