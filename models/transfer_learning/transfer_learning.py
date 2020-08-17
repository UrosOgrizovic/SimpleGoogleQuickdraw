import numpy as np
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import image_operations
import os
from keras.models import load_model
from models import data_operations
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.vgg19 import VGG19
from constants import labels, reverse_labels
from sklearn.metrics import confusion_matrix, classification_report
import math
dirname = os.path.dirname(__file__)

batch_size = 32
number_of_images_per_label = 10000

def get_model(model_name):
    return load_model(os.path.join(dirname, model_name))


def make_prediction_for_image(image, model):
    image = np.reshape(image, (28, 28))
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
    x, y = data_operations.load_data(number_of_images_per_label, True)
    x_train, x_test, y_train, y_test = data_operations.create_train_and_test_sets(x, y, True)

    base_model = VGG19(weights=None, include_top=False, input_shape=(32, 32, 3))
    # make only last few layers trainable
    # for layer in base_model.layers:
    #     layer.trainable = False

    # model has to be tweaked because the number of classes isn't the same
    outputs = base_model.output
    # outputs = Dense(1024, activation='relu')(outputs)
    # outputs = Dropout(0.2)(outputs)
    # outputs = BatchNormalization()(outputs)
    # outputs = Dense(1024, activation='relu')(outputs)
    # outputs = Dropout(0.5)(outputs)
    # outputs = BatchNormalization()(outputs)
    # outputs = Dense(1024, activation='relu')(outputs)
    # outputs = Dropout(0.2)(outputs)
    # outputs = BatchNormalization()(outputs)
    outputs = Flatten()(outputs)
    predictions = Dense(len(labels.keys()), activation='softmax')(outputs)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(lr=1e-6), metrics=['acc'])
    # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-6), metrics=['acc'])

    # train_data_gen = ImageDataGenerator(rescale=1. / 1)
    # val_data_gen = ImageDataGenerator(rescale=1. / 1)
    # train_generator = train_data_gen.flow(x_train, y_train, batch_size=batch_size)
    # val_generator = val_data_gen.flow(x_val, y_val, batch_size=batch_size)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min', min_delta=0.05)
    # mcp_save = ModelCheckpoint('VGG19_100k.h5', save_best_only=True, monitor='val_loss', mode='min')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, min_delta=1e-4, mode='min')
    # history = model.fit_generator(train_generator,
    #                     steps_per_epoch=len(x_train) // batch_size,
    #                     epochs=32,
    #                     validation_data=val_generator,
    #                     validation_steps=len(x_val) // batch_size,
    #                     callbacks=[early_stopping, mcp_save, reduce_lr_loss],
    #                     verbose=2)
    # train_acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # train_loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # data_operations.plot_training_and_validation_data(train_acc, val_acc, train_loss, val_loss, 'VGG_100k')

    # model.fit(x=x_train, y=y_train, validation_split=0.2)
    # model.save('VGG19_10k.h5')

    model = load_model(os.path.join(dirname, 'VGG19_10k.h5'))
    # print(model.metrics_names)
    # print(model.evaluate(x_test, y_test))
    y_train_pred = model.predict(x_train)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    y_test_pred = np.argmax(model.predict(x_test), axis=1)
    # from one-hot back to digits, because that's what sklearn.metrics.f1_score requires
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    #
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
    #
    print(classification_report(y_test, y_test_pred))
    print(confusion_matrix(y_test, y_test_pred))
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
