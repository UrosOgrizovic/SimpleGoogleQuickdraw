import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import image_operations
import os
from keras.models import load_model
from models import data_operations
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
from keras.constraints import max_norm  # trying weight constraints
from constants import labels, reverse_labels
import math
from sklearn.metrics import confusion_matrix, classification_report
import time
from kerastuner.tuners import RandomSearch
from models import HyperModels
from pathlib import Path


dirname = os.path.dirname(__file__)


img_rows, img_cols = 28, 28
batch_size = 32
number_of_images_per_label = 10000


def define_random_tuner(num_classes, directory=Path("./"), project_name="vanilla_cnn_tuning"):
    random_tuner = RandomSearch(
        HyperModels.CNNHyperModel(input_shape=(28, 28, 1), num_classes=num_classes),
        objective="val_loss",
        max_trials=40,
        executions_per_trial=2,
        directory=f"{directory}_random_search",
        project_name=project_name,
    )

    return random_tuner

def create_train_save_model(x_train, y_train):

    # Hyperparameter values were calculated by keras tuners
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_constraint=max_norm(3),
                     bias_constraint=max_norm(3), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.1))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.15))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dropout(0.2))

    # odavde
    # model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), padding='same'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.2))


    # model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), padding='same'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.2))
    #
    #
    # model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), padding='same'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.2))
    #
    #
    # model.add(Conv2D(256, (3, 3), activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), padding='same'))
    # model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.2))

    # dovde

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_constraint=max_norm(3),
                     bias_constraint=max_norm(3)))
    model.add(MaxPooling2D((2, 2), padding='same'))
    # model.add(Dropout(0.05))

    model.add(Flatten())
    # model.add(Dropout(0.25))  # Dropout for regularization
    model.add(Dense(768, activation='relu', kernel_regularizer=l2(l=0.001)))
    model.add(Dense(len(labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])

    x_train = x_train[:math.ceil(0.8*len(x_train))]
    x_val = x_train[math.ceil(0.8*len(x_train)):]
    y_train = y_train[:math.ceil(0.8 * len(y_train))]
    y_val = y_train[math.ceil(0.8 * len(y_train)):]

    # tuner = define_random_tuner(num_classes=len(labels))
    # tuner.search(x_train, y_train, epochs=20, validation_data=(x_val, y_val))
    # tuner.results_summary()
    # print('-------------------------------------')
    # best_hp = tuner.get_best_hyperparameters()[0]
    # model = tuner.hypermodel.build(best_hp)
    # print(model.get_config())
    #
    # quit()


    ''' train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True)
                                        '''
    train_data_gen = ImageDataGenerator(rescale=1. / 255)
    val_data_gen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_data_gen.flow(x_train, y_train, batch_size=batch_size)
    val_generator = val_data_gen.flow(x_val, y_val, batch_size=batch_size)

    early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min')
    file_to_save_to = ''
    if number_of_images_per_label == 100000:
        file_to_save_to = 'vanilla_cnn_model_100k.h5'
    elif number_of_images_per_label == 10000:
        file_to_save_to = 'vanilla_cnn_model_10k.h5'
    else:
        file_to_save_to = 'vanilla_cnn_model.h5'
    mcp_save = ModelCheckpoint(file_to_save_to, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min')

    # augmentation helps avoid overfitting
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=len(x_train) // batch_size,
                                  epochs=64,
                                  validation_data=val_generator,
                                  validation_steps=len(x_val) // batch_size,
                                  callbacks=[early_stopping, mcp_save, reduce_lr_loss],
                                  verbose=2)

    # model.save(file_to_save_to) # not using this because of mcp_save
    return history


def get_model(model_name):
    return load_model(os.path.join(dirname, model_name), compile=False)


def make_prediction_for_image(image, model):
    test_image = np.expand_dims(image, axis=-1)
    max_idx = np.argmax(model.predict(test_image))
    to_return_probs = {'airplane': 0, 'alarm clock': 0, 'axe': 0, 'The Mona Lisa': 0,
          'bicycle': 0, 'ant': 0}
    predicted_probs = model.predict(test_image).tolist()[0]
    to_return_probs['airplane'] = predicted_probs[0]
    to_return_probs['alarm clock'] = predicted_probs[1]
    to_return_probs['axe'] = predicted_probs[2]
    to_return_probs['The Mona Lisa'] = predicted_probs[3]
    to_return_probs['bicycle'] = predicted_probs[4]
    to_return_probs['ant'] = predicted_probs[5]
    return reverse_labels[max_idx], to_return_probs


if __name__ == "__main__":
    x, y = data_operations.load_data(number_of_images_per_label)
    x_train, x_test, y_train, y_test = data_operations.create_train_and_test_sets(x, y)

    model = load_model(os.path.join(dirname, 'vanilla_cnn_model_100k.h5'))
    # y_train_pred = model.predict(x_train)
    # y_train_pred = np.argmax(y_train_pred, axis=1)
    # y_test_pred = np.argmax(model.predict(x_test), axis=1)



    # from one-hot back to digits, because that's what sklearn.metrics.f1_score requires
    # y_train = np.argmax(y_train, axis=1)
    # y_test = np.argmax(y_test, axis=1)
    #
    # print('Confusion matrix:')
    # print(confusion_matrix(y_train, y_train_pred))
    # print('Classification report tr:')
    # print(classification_report(y_train, y_train_pred))
    # print('Classification report tst:')
    # print(classification_report(y_test, y_test_pred))

    # print(model.metrics_names)
    print(model.evaluate(x_test, y_test))




    # history = create_train_save_model(x_train, y_train)
    # # get the details form the history object
    # train_acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # train_loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # data_operations.plot_training_and_validation_data(train_acc, val_acc, train_loss, val_loss, '10k')

    # test_image = image_operations.load_images(os.path.join(dirname, '../../data/img.npy'))

    # image_operations.display_image(np.squeeze(test_image))
    # (28, 28) -> (1, 28, 28, 1)
    # test_image = np.expand_dims(test_image, axis=-1)
    # print(test_image.shape)
    # max_idx = np.argmax(model.predict(test_image))
    # print(reverse_labels[max_idx])
    # print(model.predict(test_image))
