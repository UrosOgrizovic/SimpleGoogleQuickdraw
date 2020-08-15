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


def load_data(number_of_images_to_load_per_label, is_transfer_learning=False):
    x = []
    y = []
    for lab in labels:
        for img in image_operations.load_images(file_path_prefix + lab + '.npy', number_of_images_to_load_per_label):
            if is_transfer_learning:
                img = image_operations.pad_image(img)
                img = np.repeat(img[..., np.newaxis], 3, -1)
            # normalization
            normalized_img = img/255.0
            x.append(normalized_img)
            y.append(labels[lab])

    return x, y


def plot_training_and_validation_data(train_acc, val_acc, train_loss, val_loss, num_images_per_label):
    """
    plots training and validation curves
    :param train_acc:
    :param val_acc:
    :param train_loss:
    :param val_loss:
    :param num_images_per_label:
    :return:
    """
    epochs = range(1, len(train_acc) + 1)

    # Train and validation accuracy
    plt.plot(epochs, train_acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.xlabel('Epochs')
    plt.title('Training and Validation accurarcy')
    plt.legend()
    plt.savefig('vanilla_cnn_'+num_images_per_label+'_train_val_acc', bbox_inches='tight')

    plt.figure()
    # Train and validation loss
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.xlabel('Epochs')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig('vanilla_cnn_'+num_images_per_label+'_train_val_loss', bbox_inches='tight')

    plt.show()


def create_train_and_test_sets(x, y, is_transfer_learning=False):
    x = np.array(x)
    if not is_transfer_learning:
        # (?, 28, 28) -> (?, 28, 28, 1)
        x = np.expand_dims(x, axis=-1)

    y = np.array(y)

    y = keras.utils.to_categorical(y, num_classes=len(labels.keys()), dtype='uint8')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=2)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x, y = load_data(100000)
    x_tr, x_tst, y_tr, y_tst = create_train_and_test_sets(x, y)

    airplane_tr = len([lbl for lbl in y_tr if np.argmax(lbl) == 0])
    alarm_clock_tr = len([lbl for lbl in y_tr if np.argmax(lbl) == 1])
    axe_tr = len([lbl for lbl in y_tr if np.argmax(lbl) == 2])
    the_mona_lisa_tr = len([lbl for lbl in y_tr if np.argmax(lbl) == 3])
    bicycle_tr = len([lbl for lbl in y_tr if np.argmax(lbl) == 4])
    ant_tr = len([lbl for lbl in y_tr if np.argmax(lbl) == 5])

    airplane_tst = len([lbl for lbl in y_tst if np.argmax(lbl) == 0])
    alarm_clock_tst = len([lbl for lbl in y_tst if np.argmax(lbl) == 1])
    axe_tst = len([lbl for lbl in y_tst if np.argmax(lbl) == 2])
    the_mona_lisa_tst = len([lbl for lbl in y_tst if np.argmax(lbl) == 3])
    bicycle_tst = len([lbl for lbl in y_tst if np.argmax(lbl) == 4])
    ant_tst = len([lbl for lbl in y_tst if np.argmax(lbl) == 5])

    print(airplane_tr, alarm_clock_tr, axe_tr, the_mona_lisa_tr, bicycle_tr, ant_tr)
    print(airplane_tst, alarm_clock_tst, axe_tst, the_mona_lisa_tst, bicycle_tst, ant_tst)