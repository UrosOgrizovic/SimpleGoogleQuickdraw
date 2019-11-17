import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
# import image_operations
import data_operations


labels = {'airplane': np.uint8(0), 'alarm clock': np.uint8(1), 'axe': np.uint8(2), 'The Mona Lisa': np.uint8(3)}
img_rows, img_cols = 28, 28


def create_model(x_train, y_train, x_validation, y_validation, x_test, y_test, num_classes=len(labels),
                 input_shape=(28, 28, 1), batch_size=2, epochs=2):
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes, dtype='uint8')
    y_validation = keras.utils.to_categorical(y_validation, num_classes=num_classes, dtype='uint8')
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes, dtype='uint8')
    # print(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape, x_test.shape, y_test.shape)

    # print(x_train[0])
    # quit()
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # model.add(Flatten(input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_validation, y_validation))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def split_data_into_x_y(_data):
    # TODO shuffle data when loading into x and y, otherwise the model only gets to train on labels 0, 1, 2
    # here it's important to use numpy.resize, not ndarray.resize (https://stackoverflow.com/a/23253578/7305715)
    x = np.resize(_data["image"].values, (400, 28, 28, 1))
    y = np.resize(_data["label"].values, (400, 1))
    # print(x)
    # quit()
    _x_tr = np.zeros(shape=(240, 28, 28, 1), dtype=np.uint8)
    _x_val = np.zeros(shape=(80, 28, 28, 1), dtype=np.uint8)
    _x_tst = np.zeros(shape=(80, 28, 28, 1), dtype=np.uint8)

    _y_tr = np.zeros(shape=(240, 1), dtype=np.uint8)
    _y_val = np.zeros(shape=(240, 1), dtype=np.uint8)
    _y_tst = np.zeros(shape=(240, 1), dtype=np.uint8)

    # splitting image data into train:validation:test sets into ratio 60:20:20
    len_x = len(x)
    sixty_percent_x = int(0.6 * len_x)
    eighty_percent_x = int(0.8 * len_x)
    _x_tr = x[:sixty_percent_x]
    _x_val = x[sixty_percent_x:eighty_percent_x]
    _x_tst = x[eighty_percent_x:]

    # z = []
    # for i in range(sixty_percent_x):
    #
    #     z = np.expand_dims(x[i], axis=0)
    #     print(x[i])
    #     quit()
    #     # np.append(_x_tr, x[i])
    # for i in range(sixty_percent_x, eighty_percent_x):
    #     z = np.expand_dims(x[i], axis=0)
    #     np.append(_x_val, z)
    # for i in range(eighty_percent_x, len_x):
    #     z = np.expand_dims(x[i], axis=0)
    #     np.append(_x_tst, z)

    # print(_x_tr.shape)
    # quit()

    # splitting label data into train:validation:test sets into ratio 60:20:20
    len_y = len(y)
    sixty_percent_y = int(0.6*len_y)
    eighty_percent_y = int(0.8*len_y)
    _y_tr = y[0:sixty_percent_y]
    _y_val = y[sixty_percent_y:eighty_percent_y]
    _y_tst = y[eighty_percent_y:]

    # for i in range(sixty_percent_y):
    #     z = np.expand_dims(y[i], axis=0)
    #     np.append(_y_tr, z)
    # for i in range(sixty_percent_y, eighty_percent_y):
    #     z = np.expand_dims(y[i], axis=0)
    #     np.append(_y_val, z)
    # for i in range(eighty_percent_y, len_y):
    #     z = np.expand_dims(y[i], axis=0)
    #     np.append(_y_tst, z)

    # print(_y_tr)
    # quit()


    return _x_tr, _y_tr, _x_val, _y_val, _x_tst, _y_tst


if __name__ == "__main__":
    data = data_operations.load_data(100)
    x_tr, y_tr, x_val, y_val, x_tst, y_tst = split_data_into_x_y(data)

    # y_tst = image_operations.load_images('../data/img.npy')[0]
    create_model(x_tr, y_tr, x_val, y_val, x_tst, y_tst)