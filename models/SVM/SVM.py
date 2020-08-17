from sklearn import svm
from models import data_operations
import os
import numpy as np
# using joblib instead of pickle because it's more efficient on objects that carry large numpy arrays
from joblib import dump, load
from constants import labels, reverse_labels
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import cv2
dirname = os.path.dirname(__file__)


def get_model(model_name):
    return load(os.path.join(dirname, model_name))


def make_prediction_for_image(image, model):
    image = np.array(image)
    image = image.reshape(image.shape[0], image.shape[1] * image.shape[2])
    predicted = reverse_labels[model.predict(image)[0]]

    return predicted


if __name__ == "__main__":
    x, y = data_operations.load_data(10000, False)

    x = np.array(x)
    y = np.array(y)


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=2)

    x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # svc = svm.SVC(gamma='scale')

    # clf = svm.SVC()

    # parameters = {'C': [2**2, 2**3, 2**4, 2**5, 2**6]}

    # clf = GridSearchCV(svc, parameters, cv=10)
    # clf.fit(x_train, y_train)
    # print(clf.best_params_)

    # dump(clf, 'SVM_2k.joblib', compress=3)
    clf = load('SVM_10k.joblib')


    print(clf.score(x_train, y_train))
    print(clf.score(x_test, y_test))

    # test_image = np.array(image_operations.load_images(os.path.join(dirname, '../../data/img.npy')))
    # # image_operations.display_image(np.squeeze(test_image))
    # test_image = test_image.reshape(test_image.shape[0], test_image.shape[1] * test_image.shape[2])
    # standardized_test_image = np.array(test_image/255.0, float)
    # print(reverse_labels[clf.predict(test_image)[0]])
