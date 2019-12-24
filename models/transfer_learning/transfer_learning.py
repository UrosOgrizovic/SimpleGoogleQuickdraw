import numpy as np
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import image_operations
import os
from keras.models import load_model
import data_operations
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.vgg19 import VGG19
from constants import labels, reverse_labels
dirname = os.path.dirname(__file__)

number_of_images_per_label = 10000

if __name__ == "__main__":
    x, y = data_operations.transfer_learning_load_data(number_of_images_per_label)
    x_train, x_val, y_train, y_val = data_operations.create_train_and_validation_sets(x, y, True)
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # model has to be tweaked because the number of classes isn't the same
    outputs = base_model.output
    outputs = Dense(1024, activation='relu')(outputs)
    outputs = Dense(1024, activation='relu')(outputs)
    outputs = Dense(1024, activation='relu')(outputs)
    # print(outputs.shape)
    outputs = Flatten()(outputs)
    print(outputs.shape)
    predictions = Dense(len(labels.keys()), activation='softmax')(outputs)
    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    model.fit(x_train, x_val, y_train, y_val)

    test_image = np.reshape(np.load(os.path.join(dirname, '../../data/img.npy')), (28, 28))
    # tl_image_operations.display_image(test_image)
    test_image = np.pad(test_image, 2)
    test_image = np.repeat(test_image[..., np.newaxis], 3, -1)
    # (32, 32, 3) -> (1, 32, 32, 3)
    test_image = np.expand_dims(test_image, axis=0)

    label_predictions = model.predict(test_image)
    print(label_predictions)
    display_predictions = "--\n"
    i = 0
    for label in labels.keys():
        display_predictions += label + ": " + str(label_predictions[0][i]) + "\n"
        i += 1
    display_predictions += "--\n"
    reverse_labels_key = np.argmax(label_predictions)
    predicted_class = reverse_labels[reverse_labels_key]
    print(display_predictions)
    print(predicted_class)