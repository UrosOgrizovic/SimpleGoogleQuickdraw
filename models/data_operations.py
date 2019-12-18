import pandas as pd
import numpy as np
import image_operations
import cv2
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

file_path_prefix = '../../data/full_numpy_bitmap_'
labels = {'airplane': np.uint8(0), 'alarm clock': np.uint8(1), 'axe': np.uint8(2), 'The Mona Lisa': np.uint8(3),
          'bicycle': np.uint8(4), 'ant': np.uint8(5)}




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

def create_train_and_validation_sets(x, y):
    x = np.array(x)
    # (?, 28, 28) -> (?, 28, 28, 1)
    x = np.expand_dims(x, axis=-1)

    y = np.array(y)

    y = keras.utils.to_categorical(y, num_classes=len(labels.keys()), dtype='uint8')

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=2)
    return x_train, x_val, y_train, y_val


if __name__ == "__main__":
    load_data(100)