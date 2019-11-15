import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import image_operations
import load_data


labels = ['airplane', 'alarm clock', 'axe', 'The Mona Lisa']

data = load_data.load_data(100000)
print(data)


def create_model(x_train, y_train, y_test):
    y_train = keras.utils.to_categorical(y_train, num_classes=len(labels), dtype='float32')


if __name__ == "__main__":
    y_tst = image_operations.load_images('../data/img.npy')[0]
    # create_model(x_tr, y_tr, y_tst)