from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.callbacks.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.constraints import max_norm
from keras.regularizers import l2

class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        HyperModel.__init__(self) # must call __init__() of superclass
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        max_pooling_layer = MaxPooling2D((2, 2))

        model = keras.Sequential()
        model.add(Conv2D(filters=hp.Choice("num_filters", values=[16, 32], default=32,), kernel_size=(3, 3),
                         activation='relu', input_shape=self.input_shape, kernel_constraint=max_norm(3),
                         bias_constraint=max_norm(3), padding='same'))
        model.add(max_pooling_layer)
        model.add(Dropout(rate=hp.Float("dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Conv2D(filters=hp.Choice("num_filters_1", values=[64, 128, 256], default=64,), kernel_size=(3, 3),
                         activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
        model.add(max_pooling_layer)
        model.add(Dropout(rate=hp.Float("dropout_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Conv2D(filters=hp.Choice("num_filters_2", values=[64, 128, 256], default=64,), kernel_size=(3, 3),
                         activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
        model.add(max_pooling_layer)
        model.add(Dropout(rate=hp.Float("dropout_3", min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Conv2D(filters=hp.Choice("num_filters_3", values=[64, 128, 256], default=64,), kernel_size=(3, 3),
                         activation='relu', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), padding='same'))
        model.add(max_pooling_layer)
        model.add(Dropout(rate=hp.Float("dropout_4", min_value=0.0, max_value=0.5, default=0.25, step=0.05)))

        model.add(Flatten())
        model.add(Dropout(rate=hp.Float("dropout_5", min_value=0.0, max_value=0.5, default=0.25, step=0.05)))
        model.add(Dense(units=hp.Int(
                    "units", min_value=128, max_value=1024, step=32, default=512
                ), activation='relu', kernel_regularizer=l2(l=0.001)))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.RMSprop(
            hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log", default=1e-3)), metrics=['acc'])

        return model