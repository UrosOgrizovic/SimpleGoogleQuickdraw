import numpy as np
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import image_operations
import os
from keras.models import load_model
from models import data_operations
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.vgg19 import VGG19
from constants import labels, reverse_labels
dirname = os.path.dirname(__file__)

number_of_images_per_label = 10000

def make_prediction_for_image(image, model_name):
    image = np.reshape(image, (28, 28))

    model = load_model(os.path.join(dirname, model_name))
    image = np.pad(image, 2)
    image = np.repeat(image[..., np.newaxis], 3, -1)
    # (32, 32, 3) -> (1, 32, 32, 3)
    image = np.expand_dims(image, axis=0)

    label_predictions = model.predict(image)[0]
    to_return_probs = {'airplane': str(round(label_predictions[0], 2)), 'alarm clock': str(round(label_predictions[1], 2)),
                       'axe': str(round(label_predictions[2], 2)), 'The Mona Lisa': str(round(label_predictions[3], 2)),
                       'bicycle': str(round(label_predictions[4], 2)), 'ant': str(round(label_predictions[5], 2))}
    max_idx = np.argmax(label_predictions)
    return reverse_labels[max_idx], to_return_probs

if __name__ == "__main__":
    # x, y = data_operations.transfer_learning_load_data(number_of_images_per_label)
    # x_train, x_val, y_train, y_val = data_operations.create_train_and_validation_sets(x, y, True)
    # base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    #
    # # model has to be tweaked because the number of classes isn't the same
    # outputs = base_model.output
    # outputs = Dense(1024, activation='relu')(outputs)
    # outputs = Dense(1024, activation='relu')(outputs)
    # outputs = Dense(1024, activation='relu')(outputs)
    # outputs = Flatten()(outputs)
    # predictions = Dense(len(labels.keys()), activation='softmax')(outputs)
    # model = Model(inputs=base_model.input, outputs=predictions)
    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    # model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val))
    # model.save('VGG19_10k.h5')

    # model = load_model(os.path.join(dirname, 'VGG19_10k.h5'))
    # test_image = np.reshape(np.load(os.path.join(dirname, '../../data/img.npy')), (28, 28))
    # image_operations.display_image(test_image)
    # test_image = np.pad(test_image, 2)
    # test_image = np.repeat(test_image[..., np.newaxis], 3, -1)
    # # (32, 32, 3) -> (1, 32, 32, 3)
    # test_image = np.expand_dims(test_image, axis=0)
    #
    # label_predictions = model.predict(test_image)
    # display_predictions = "--\n"
    # i = 0
    # for label in labels.keys():
    #     display_predictions += label + ": " + str(label_predictions[0][i]) + "\n"
    #     i += 1
    # display_predictions += "--\n"
    # reverse_labels_key = np.argmax(label_predictions)
    # predicted_class = reverse_labels[reverse_labels_key]
    # print(display_predictions)
    # print(predicted_class)
    pass