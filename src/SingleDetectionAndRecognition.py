import os
import cv2
import numpy as np
from models import CustomModel

PATH = 'data/correct-images'


def get_prediction(images):
    pred_images = []
    for image in images:
        image = cv2.resize(image, (32, 32)).reshape((32, 32, 1))
        pred_images.append(image)
    labels = []
    for i, val in enumerate(CustomModel.predict("custom_model.json", "weights.h5",
                                                np.array(pred_images))):
        label = np.argmax(val)
        if np.amax(val) >= 0.5 and label != 10:
            labels.append(label)
        else:
            labels.append(None)
    return labels


def classify(o_image):
    images = []
    if len(o_image.shape) > 2:
        image = cv2.cvtColor(o_image.copy(), cv2.COLOR_BGR2GRAY)
    else:
        image = o_image.copy()
    image_mser = cv2.MSER_create(_max_variation=0.225)
    rois, _ = image_mser.detectRegions(image)
    bounding_boxes = []
    for roi in rois:
        x2, y2 = np.amax(roi, axis=0)
        x1, y1 = np.amin(roi, axis=0)
        if (x2 - x1 <= 0.5 * image.shape[1] or y2 - y1 <= 0.5 * image.shape[0]) \
                and (x2 - x1 >= 0.05 * image.shape[1] and y2 - y1 >= 0.05 * image.shape[0]) \
                and (x2 - x1 <= (y2 - y1) * 2):

            img = image[y1: y2, x1: x2]
            if img.size > 0:
                images.append(img)
                bounding_boxes.append((x1, y1, x2, y2))

    if len(images) == 0:
        return o_image

    labels = get_prediction(images)
    nms_arr = []
    for i, label in enumerate(labels):
        if label is None:
            continue
        x1, y1, x2, y2 = bounding_boxes[i]
        nms_arr.append((x1, y1, x2, y2, label))
    output = perform_nms(np.array(nms_arr), 0.05)
    for x1, y1, x2, y2, label in output:
        o_image = cv2.rectangle(o_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        o_image = cv2.putText(
            o_image,
            str(label),
            org=(x1, y2 + 3),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            color=(0, 0, 255),
            thickness=2,
            fontScale=2
        )
    return o_image


def perform_nms(roi, threshold):
    if len(roi) == 0:
        return []

    roi = roi.astype(np.float32)
    final_roi_indices = []

    x1 = roi[:, 0]
    y1 = roi[:, 1]
    x2 = roi[:, 2]
    y2 = roi[:, 3]
    indices = np.argsort(y2)

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        final_roi_indices.append(i)
        overlap = (np.maximum(0, np.minimum(x2[i],
                                            x2[indices[:last]]) - np.maximum(x1[i], x1[indices[:last]]) + 1)
                   * np.maximum(0, np.minimum(y2[i], y2[indices[:last]]) - np.maximum(y1[i], y1[indices[:last]]) + 1)) / \
                  ((x2 - x1 + 1) * (y2 - y1 + 1))[indices[:last]]

        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > threshold)[0])))
    return roi[final_roi_indices].astype(np.int)
