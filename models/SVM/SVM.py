from sklearn import svm
from models import data_operations
import os
import numpy as np
# using joblib instead of pickle because it's more efficient on objects that carry large numpy arrays
from joblib import dump, load
from constants import labels, reverse_labels
dirname = os.path.dirname(__file__)


def make_prediction_for_image(image, model_name):
    image = np.array(image)
    image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
    svm = load(os.path.join(dirname, model_name))
    predicted = reverse_labels[svm.predict(image)[0]]

    return predicted


def standardize_x(x):
    arr_x = np.asarray(x)
    s_arr_x = np.array(arr_x, float)
    s_arr_x /= 255.0
    return s_arr_x


if __name__ == "__main__":
    x, y = data_operations.load_data(2000)
    x = np.array(x)
    standardized_x = standardize_x(x)

    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
    y = np.array(y)
    clf = svm.SVC()
    clf.fit(x, y)
    dump(clf, 'SVM_2k.joblib', compress=3)
    # clf = load('SVM_2k.joblib')
    # test_image = np.array(image_operations.load_images(os.path.join(dirname, '../../data/img.npy')))
    # # image_operations.display_image(np.squeeze(test_image))
    # test_image = test_image.reshape(test_image.shape[0], test_image.shape[1] * test_image.shape[2])
    # standardized_test_image = np.array(test_image/255.0, float)
    # print(reverse_labels[clf.predict(test_image)[0]])
