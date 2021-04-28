from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation, Dropout, Input, LeakyReLU, ZeroPadding2D, UpSampling2D
from tensorflow.keras import callbacks
from tensorflow.keras.utils import to_categorical, plot_model
import numpy as np
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD, Adam
import json


class BaseModel:
    history = None
    model = None
    loss_img = None
    acc_img = None

    def save_model(self, model_json_file, weights_file):
        self.model.save_weights(weights_file)
        model_json = self.model.to_json()
        with open(model_json_file, "w+") as file:
            file.write(model_json)

    @staticmethod
    def predict(model_json, weights, X):
        with open(model_json, 'r') as json_file:
            model_json = json.load(json_file)
            loaded_model = model_from_json(json.dumps(model_json))
            loaded_model.load_weights(weights)
            X = np.array(X)
            Y = loaded_model.predict(np.array(X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))))
            return Y


class CustomModel(BaseModel):
    def __init__(self):
        self.model = Sequential()
        self.acc_img = 'acc_custom.png'
        self.loss_img = 'loss_custom.png'
        self.model.add(Conv2D(32, 3, activation = 'relu', padding = 'valid', input_shape = (32, 32, 1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D(pool_size = (2, 2), strides = 2, padding = 'valid'))
        self.model.add(Conv2D(64, 3, activation = 'relu', padding = 'same'))
        self.model.add(Conv2D(128, 5, activation = 'relu', padding = 'same'))
        self.model.add(MaxPool2D(pool_size = (3, 3), strides = 2, padding = 'valid'))
        self.model.add(Conv2D(256, 5, activation = 'relu', padding = 'valid'))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(512, 2, activation = 'relu', padding = 'valid'))
        self.model.add(MaxPool2D(pool_size = (3, 3), strides = 2, padding = 'same'))
        self.model.add(Flatten())
        self.model.add(Dense(256, kernel_initializer = 'uniform'))
        self.model.add(Dense(128))
        self.model.add(Dense(11, activation = 'softmax'))

    def fit(self, X_train, Y_train):
        Y_train = to_categorical(Y_train)
        self.model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        X_train = np.array(X_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
        self.history = self.model.fit(X_train, Y_train, shuffle=True,
        callbacks=[callbacks. EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')],
                                      epochs=100, validation_split=0.2)

    def evaluate(self, X_test, Y_test):
        Y_test = to_categorical(Y_test)
        X_test = np.array(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
        score, accuracy = self.model.evaluate(X_test, Y_test, verbose = 0)
        return {
            "score": score,
            "accuracy": accuracy
        }


class VGG16Model(BaseModel):
    def __init__(self, pretrained=True):
        if pretrained:
            weights = 'imagenet'
            self.acc_img = 'acc_VGG_Pretrained.png'
            self.loss_img = 'loss_VGG_Pretrained.png'
        else:
            weights = None
            self.acc_img = 'acc_VGG.png'
            self.loss_img = 'loss_VGG.png'
        vgg16 = VGG16(include_top=False, weights=weights, input_shape=(32, 32, 3))
        input_layer = Input(shape = (32, 32, 3), name = 'image_input')
        vgg16_output = vgg16(input_layer)
        flatten_layer = Flatten(name = 'flatten')(vgg16_output)
        fc1 = Dense(4096, activation = 'relu', name = 'conn1')(flatten_layer)
        fc2 = Dense(4096, activation = 'relu', name = 'conn2')(fc1)
        output = Dense(11, activation = 'softmax', name = 'predictions')(fc2)
        self.model = Model(input_layer, output)

    def fit(self, X_train, Y_train):
        Y_train = to_categorical(Y_train)
        adam = Adam(lr=1e-4, amsgrad=True)
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        X_train = np.array(X_train)
        self.history = self.model.fit(X_train, Y_train, shuffle=True, validation_split=0.2,
        callbacks=[callbacks. EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min')],
                                      epochs=100)

    def evaluate(self, X_test, Y_test):
        Y_test = to_categorical(Y_test)
        X_test = np.array(X_test)
        score, accuracy = self.model.evaluate(X_test, Y_test, verbose = 0)
        return {
            "score": score,
            "accuracy": accuracy
        }