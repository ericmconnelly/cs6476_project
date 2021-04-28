from models import CustomModel, VGG16Model, BaseModel
import numpy as np
import cv2
from scipy.io import loadmat
import os
import h5py

FORMAT1_PATH = "./data/format1/"
FORMAT2_PATH = "./data/format2/"
TEST_PATH = "./data/format1/test-images/"

def train_digit_recognition_model(model_instance: BaseModel, model_json_file, weights_file, grayscale=True):
    if grayscale:
        channels = 1
    else:
        channels = 3
    X_train, Y_train, X_test, Y_test = _load_and_preprocess(grayscale)
    X_train_neg, Y_train_neg = _load_neg_dataset('train', grayscale)
    X_train = np.array(X_train)
    X_train = np.concatenate((X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], channels)), X_train_neg))
    Y_train = np.concatenate((Y_train, Y_train_neg))
    model = model_instance
    model.fit(X_train, Y_train)
    model.plot()
    model.save_model(model_json_file, weights_file)
    X_test_neg, Y_test_neg = _load_neg_dataset('test', grayscale)
    X_test = np.concatenate((X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], channels)), X_test_neg))
    Y_test = np.concatenate((Y_test, Y_test_neg))


def _get_image_name(i, file):
    name = file['digitStruct/name']
    return ''.join([chr(char[0]) for char in file[name[i][0]].value])


def get_bounding_box(i, file):
    bbox_dict = {}
    item = file['digitStruct/bbox'][i].item()
    for attribute in ['top', 'left', 'height', 'width', 'label']:
        keys = file[item][attribute]
        values = [file[keys.value[i].item()].value[0][0]
                  for i in range(len(keys))] if len(keys) > 1 else [keys.value[0][0]]
        bbox_dict[attribute] = values
    return bbox_dict['top'], bbox_dict['left'], bbox_dict['width'], bbox_dict['height'], bbox_dict['label']


def _load_neg_dataset(path, grayscale=True):
    X = []
    Y = []
    path = os.path.join("%s" % FORMAT1_PATH, path)
    file = h5py.File(os.path.join(path, 'digitStruct.mat'))
    for i in range(file['/digitStruct/name'].shape[0]):
        name = _get_image_name(i, file)
        x, y, w, h, labels = get_bounding_box(i, file)
        xp = [xc + hc for xc, hc in zip(x, h)]
        yp = [yc + wc for yc, wc in zip(y, w)]
        t_x = int(np.clip(np.min(x), 0, np.inf))
        t_y = int(np.clip(np.min(y), 0, np.inf))
        b_x = int(np.clip(np.max(xp), 0, np.inf))
        b_y = int(np.clip(np.max(yp), 0, np.inf))
        image = cv2.imread(os.path.join(path, name))
        mod_reshape = (32, 32, 3)
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mod_reshape = (32, 32, 1)
        if t_x > 10:
            mod_img = cv2.resize(image[0: t_x, :], (32, 32))
            X.append(mod_img.reshape(mod_reshape))
            Y.append(10)
        elif image.shape[0] - b_x > 10:
            mod_img = cv2.resize(image[b_x:, :], (32, 32))
            X.append(mod_img.reshape(mod_reshape))
            Y.append(10)
        elif t_y > 10:
            mod_img = cv2.resize(image[:, :t_y], (32, 32))
            X.append(mod_img.reshape(mod_reshape))
            Y.append(10)
        elif image.shape[0] - b_y > 10:
            mod_img = cv2.resize(image[:, b_y:], (32, 32))
            X.append(mod_img.reshape(mod_reshape))
            Y.append(10)
    return np.array(X), Y


def _load_data(file):
    data = loadmat(os.path.join(FORMAT2_PATH, file))
    return data['X'], data['y']


def load_data():
    train_x, train_y = _load_data('train_32x32.mat')
    test_x, test_y = _load_data('test_32x32.mat')
    train_x = train_x.transpose((3, 0, 1, 2))
    test_x = test_x.transpose((3, 0, 1, 2))
    train_y = train_y[:, 0]
    test_y = test_y[:, 0]
    return train_x, train_y, test_x, test_y


def _load_and_preprocess(grayscale=True):
    train_x, train_y, test_x, test_y = load_data()
    X_train = []
    X_test = []

    for image in train_x:
        if grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            X_train.append(gray)
        else:
            X_train.append(image.astype(np.float32))

    for image in test_x:
        if grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            X_test.append(gray)
        else:
            X_test.append(image.astype(np.float32))

    train_y[train_y == 10] = 0
    test_y[test_y == 10] = 0
    Y_train = train_y
    Y_test = np.array(test_y)
    X_test = np.array(X_test)
    return X_train, Y_train, X_test, Y_test


# train_digit_recognition_model(CustomModel(), "custom_model.json", "weights.h5", True)
# train_digit_recognition_model(VGG16Model(), "vgg_16_pretrained.json", "vgg_16_pretrained.h5", False)
# train_digit_recognition_model(VGG16Model(False), "vgg_16.json", "vgg_16.h5", False)
