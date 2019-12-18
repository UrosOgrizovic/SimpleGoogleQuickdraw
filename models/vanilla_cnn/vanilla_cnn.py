import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import image_operations
import os
from keras.models import load_model
from models import data_operations
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from keras.constraints import max_norm # trying weight contraints

dirname = os.path.dirname(__file__)

labels = {'airplane': np.uint8(0), 'alarm clock': np.uint8(1), 'axe': np.uint8(2), 'The Mona Lisa': np.uint8(3),
          'bicycle': np.uint8(4), 'ant': np.uint8(5)}
reverse_labels = {0: 'airplane', 1: 'alarm clock', 2: 'axe', 3: 'The Mona Lisa', 4: 'bicycle', 5: 'ant'}
img_rows, img_cols = 28, 28
batch_size = 32
number_of_images_per_label = 10000

def get_data(number_of_images_to_load_per_label):
    x, y = data_operations.load_data(number_of_images_to_load_per_label)
    x = np.array(x)
    # (?, 28, 28) -> (?, 28, 28, 1)
    x = np.expand_dims(x, axis=-1)

    y = np.array(y)

    y = keras.utils.to_categorical(y, num_classes=len(labels.keys()), dtype='uint8')

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=2)
    return x_train, x_val, y_train, y_val


def plot_training_and_validation_data(train_acc, val_acc, train_loss, val_loss):
    """
    plots training and validation curves
    :param train_acc:
    :param val_acc:
    :param train_loss:
    :param val_loss:
    :return:
    """
    epochs = range(1, len(train_acc) + 1)

    # Train and validation accuracy
    plt.plot(epochs, train_acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()


def create_train_save_model(x_train, x_val, y_train, y_val):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dropout(0.3))  # Dropout for regularization
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(l=0.001)))
    model.add(Dense(len(labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
    val_data_gen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_data_gen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_data_gen.flow(x_val, y_val, batch_size=batch_size)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    file_to_save_to = ''
    if number_of_images_per_label == 100000:
        file_to_save_to = 'vanilla_cnn_model_100k.h5'
    elif number_of_images_per_label == 10000:
        file_to_save_to = 'vanilla_cnn_model_10k.h5'
    else:
        file_to_save_to = 'vanilla_cnn_model.h5'
    mcp_save = ModelCheckpoint(file_to_save_to, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    # augmentation helps avoid overfitting
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(x_train) // batch_size,
                                  epochs=64,
                                  validation_data=val_generator,
                                  validation_steps=len(x_val) // batch_size,
                                  callbacks=[early_stopping, mcp_save, reduce_lr_loss],
                                  verbose=2)

    # model.save('vanilla_cnn_model.h5') # not using this because of mcp_save
    return history

if __name__ == "__main__":
    x_train, x_val, y_train, y_val = get_data(number_of_images_per_label)
    # print(x_train)


    # model = load_model(os.path.join(dirname, 'vanilla_cnn_model.h5'))
    # print(model.summary())

    history = create_train_save_model(x_train, x_val, y_train, y_val)
    # get the details form the history object
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plot_training_and_validation_data(train_acc, val_acc, train_loss, val_loss)

    test_image = image_operations.load_images(os.path.join(dirname, '../../data/img.npy'))

    # image_operations.display_image(np.squeeze(test_image))
    # (28, 28) -> (1, 28, 28, 1)
    # test_image = np.expand_dims(test_image, axis=-1)
    # print(test_image.shape)
    # max_idx = np.argmax(model.predict(test_image))
    # print(reverse_labels[max_idx])
    # print(model.predict(test_image))
