from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation, Dropout, Input, \
    LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD, Adam
import json


class BaseModel:
    model = None

    def save_model(self, json_file, weights_file):
        self.model.save_weights(weights_file)
        with open(json_file, "w+") as file:
            file.write(self.model.to_json())

    @staticmethod
    def predict(model_json, weights, X):
        with open(model_json, 'r') as json_file:
            model = model_from_json(json.dumps(json.load(json_file)))
            model.load_weights(weights)
            return model.predict(np.array(np.array(X).reshape((X.shape[0], np.array(X).shape[1], np.array(X).shape[2], 1))))


class CustomModel(BaseModel):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, 3, activation='relu', padding='valid', input_shape=(32, 32, 1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='valid'))
        self.model.add(Conv2D(64, 3, activation='relu', padding='same'))
        self.model.add(Conv2D(128, 5, activation='relu', padding='same'))
        self.model.add(MaxPool2D(pool_size=(3, 3), strides=2, padding='valid'))
        self.model.add(Conv2D(256, 5, activation='relu', padding='valid'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(512, 2, activation='relu', padding='valid'))
        self.model.add(MaxPool2D(pool_size=(3, 3), strides=2, padding='same'))
        self.model.add(Flatten())
        self.model.add(Dense(256, kernel_initializer='uniform'))
        self.model.add(Dense(128))
        self.model.add(Dense(11, activation='softmax'))

    def fit(self, X_train, Y_train):
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        X_train = np.array(X_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
        self.model.fit(X_train,
                       Y_train=to_categorical(Y_train),
                       shuffle=True,
                       callbacks=[callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                          verbose=0, mode='min')],
                       epochs=100,
                       validation_split=0.2)

    def evaluate(self, X_test, Y_test):
        Y_test = to_categorical(Y_test)
        X_test = np.array(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        score, accuracy = self.model.evaluate(X_test, Y_test, verbose=0)
        return {
            "score": score,
            "accuracy": accuracy
        }


class VGG16Model(BaseModel):
    def __init__(self, pretrained=True):
        if pretrained:
            weights = 'imagenet'
        else:
            weights = None
        output = Dense(11, activation='softmax', name='predictions')(
                    Dense(4096, activation='relu', name='conn2')(
                        Dense(4096, activation='relu', name='conn1')(
                            Flatten(name='flatten')(
                                VGG16(include_top=False, weights=weights, input_shape=(32, 32, 3))(
                                    Input(shape=(32, 32, 3), name='image_input'))))))
        self.model = Model(Input(shape=(32, 32, 3), name='image_input'), output)

    def fit(self, X_train, Y_train):
        self.model.compile(optimizer=Adam(lr=1e-4, amsgrad=True), loss='categorical_crossentropy', metrics=['accuracy'])
        X_train = np.array(X_train)
        self.model.fit(X_train=X_train,
                       Y_train=to_categorical(Y_train),
                       shuffle=True,
                       validation_split=0.2,
                       callbacks=[callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                                         verbose=0, mode='min')],
                       epochs=100)

    def evaluate(self, X_test, Y_test):
        Y_test = to_categorical(Y_test)
        X_test = np.array(X_test)
        score, accuracy = self.model.evaluate(X_test, Y_test, verbose=0)
        return {
            "score": score,
            "accuracy": accuracy
        }
