from sklearn import svm
from models import data_operations
import os
import numpy as np
# using joblib instead of pickle because it's more efficient on objects that carry large numpy arrays
from joblib import dump, load
from constants import labels, reverse_labels
from sklearn.model_selection import train_test_split
import cv2
dirname = os.path.dirname(__file__)


def make_prediction_for_image(image, model_name):
    image = np.array(image)
    image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
    svm = load(os.path.join(dirname, model_name))
    predicted = reverse_labels[svm.predict(image)[0]]

    return predicted


if __name__ == "__main__":
    x, y = data_operations.load_data(2000, False)

    x = np.array(x)
    y = np.array(y)


    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=2)

    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1] * x_val.shape[2])
    # y_train = np.array(y_train)
    # y_val = np.array(y_val)

    # clf = svm.SVC()
    # clf.fit(x_train, y_train)
    # print(clf.score(x_val, y_val))
    # dump(clf, 'SVM_10k.joblib', compress=3)
    # clf = load('SVM_10k.joblib')
    # clf.fit(x_train, y_train)
    # print(clf.score(x_train, y_train))

    # test_image = np.array(image_operations.load_images(os.path.join(dirname, '../../data/img.npy')))
    # # image_operations.display_image(np.squeeze(test_image))
    # test_image = test_image.reshape(test_image.shape[0], test_image.shape[1] * test_image.shape[2])
    # standardized_test_image = np.array(test_image/255.0, float)
    # print(reverse_labels[clf.predict(test_image)[0]])
