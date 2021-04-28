import numpy as np
import cv2
from scipy.io import loadmat
import os
import h5py

FORMAT1_PATH = "./data/format1/"
FORMAT2_PATH = "./data/format2/"
TEST_PATH = "./data/format1/test-images/"


def train_model(model, json_path, weights_path, grayscale=True):
    channels = 1 if grayscale else 3
    X_train, Y_train, X_test, Y_test = preprocess(grayscale)
    X_train_neg, Y_train_neg = load_neg_data('train', grayscale)
    X_train = np.array(X_train)
    X_train = np.concatenate((X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], channels)), X_train_neg))
    Y_train = np.concatenate((Y_train, Y_train_neg))
    model.fit(X_train, Y_train)
    model.plot()
    model.save_model(json_path, weights_path)


def get_name(i, file):
    name = file['digitStruct/name']
    return ''.join([chr(char[0]) for char in file[name[i][0]].value])


def get_bounding_box(i, file):
    dict = {}
    item = file['digitStruct/bbox'][i].item()
    for dim in ['top', 'left', 'height', 'width', 'label']:
        keys = file[item][dim]
        values = [file[keys.value[i].item()].value[0][0]
                  for i in range(len(keys))] if len(keys) > 1 else [keys.value[0][0]]
        dict[dim] = values
    return dict['top'], dict['left'], dict['width'], dict['height'], dict['label']


def load_neg_data(path, grayscale=True):
    X = []
    Y = []
    path = os.path.join("%s" % FORMAT1_PATH, path)
    file = h5py.File(os.path.join(path, 'digitStruct.mat'))
    for i in range(file['/digitStruct/name'].shape[0]):
        name = get_name(i, file)
        x, y, w, h, labels = get_bounding_box(i, file)
        xp = [xc + hc for xc, hc in zip(x, h)]
        yp = [yc + wc for yc, wc in zip(y, w)]
        x = int(np.clip(np.min(x), 0, np.inf))
        y = int(np.clip(np.min(y), 0, np.inf))
        b_x = int(np.clip(np.max(xp), 0, np.inf))
        b_y = int(np.clip(np.max(yp), 0, np.inf))
        image = cv2.imread(os.path.join(path, name))
        reshape = (32, 32, 3)

        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            reshape = (32, 32, 1)

        if image.shape[0] - b_x > 10:
            mod_img = cv2.resize(image[b_x:, :], (32, 32))
            X.append(mod_img.reshape(reshape))
            Y.append(10)
        elif y > 10:
            mod_img = cv2.resize(image[:, :y], (32, 32))
            X.append(mod_img.reshape(reshape))
            Y.append(10)
        elif x > 10:
            mod_img = cv2.resize(image[0: x, :], (32, 32))
            X.append(mod_img.reshape(reshape))
            Y.append(10)
        elif image.shape[0] - b_y > 10:
            mod_img = cv2.resize(image[:, b_y:], (32, 32))
            X.append(mod_img.reshape(reshape))
            Y.append(10)
    return np.array(X), Y


def load_data_from_mat(file):
    data = loadmat(os.path.join(FORMAT2_PATH, file))
    return data['X'], data['y']


def load_data():
    x_train, y_train = load_data_from_mat('train_32x32.mat')
    x_test, y_test = load_data_from_mat('test_32x32.mat')
    return x_train.transpose((3, 0, 1, 2)), y_train[:, 0], x_test.transpose((3, 0, 1, 2)), y_test[:, 0]


def preprocess(grayscale=True):
    x_train, y_train, x_test, y_test = load_data()
    X_train = []
    X_test = []

    for image in x_train:
        if grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            X_train.append(gray)
        else:
            X_train.append(image.astype(np.float32))

    for image in x_test:
        if grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            X_test.append(gray)
        else:
            X_test.append(image.astype(np.float32))

    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    return X_train, y_train, np.array(X_test), np.array(y_test)
