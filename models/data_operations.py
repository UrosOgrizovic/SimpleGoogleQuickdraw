import pandas as pd
import numpy as np
import image_operations
import cv2
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from constants import labels
import os
dirname = os.path.dirname(__file__)

# file_path_prefix = '../../data/full_numpy_bitmap_'
file_path_prefix = os.path.join(dirname, '../data/full_numpy_bitmap_')
# os.path.join(dirname, '../../data/img.npy')

def load_data(number_of_images_to_load_per_label):
    X = []
    Y = []
    for lab in labels:
        for img in image_operations.load_images(file_path_prefix + lab + '.npy', number_of_images_to_load_per_label):
            X.append(img)
            Y.append(labels[lab])

    return X, Y


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


def create_train_and_validation_sets(x, y, is_transfer_learning=False):
    x = np.array(x)
    if not is_transfer_learning:
        # (?, 28, 28) -> (?, 28, 28, 1)
        x = np.expand_dims(x, axis=-1)

    y = np.array(y)

    y = keras.utils.to_categorical(y, num_classes=len(labels.keys()), dtype='uint8')

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=2)
    return x_train, x_val, y_train, y_val


def transfer_learning_load_data(number_of_images_to_load_per_label):
    X = []
    Y = []
    for lab in labels:
        for img in image_operations.load_images(file_path_prefix + lab + '.npy', number_of_images_to_load_per_label):
            img = np.reshape(img, (28, 28))
            img = np.pad(img, 2)
            img = np.repeat(img[..., np.newaxis], 3, -1)

            X.append(img)
            Y.append(labels[lab])

    return X, Y


if __name__ == "__main__":
    load_data(100)