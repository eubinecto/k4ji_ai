import numpy as np
import time
import os
import csv
import copy  # chap 9
import wave  # chap 11
import cv2  # chap 12
import matplotlib.pyplot as plt

from PIL import Image
from IPython.core.display import HTML


def relu(x):
    return np.maximum(x, 0)


def relu_derv(y):
    return np.sign(y)


def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))


def sigmoid_derv(y):
    return y * (1 - y)


def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))


def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)


def tanh(x):
    return 2 * sigmoid(2 * x) - 1


def tanh_derv(y):
    return (1.0 + y) * (1.0 - y)


def softmax(x):
    max_elem = np.max(x, axis=1)
    diff = (x.transpose() - max_elem).transpose()
    exp = np.exp(diff)
    sum_exp = np.sum(exp, axis=1)
    probs = (exp.transpose() / sum_exp).transpose()
    return probs


def softmax_cross_entropy_with_logits(labels, logits):
    probs = softmax(logits)
    return -np.sum(labels * np.log(probs + 1.0e-10), axis=1)


def softmax_cross_entropy_with_logits_derv(labels, logits):
    return softmax(logits) - labels


def load_csv(path, skip_header=True):
    with open(path) as csvfile:
        csvreader = csv.reader(csvfile)
        headers = None
        if skip_header:
            headers = next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    return rows, headers


def onehot(xs, cnt):
    return np.eye(cnt)[np.array(xs).astype(int)]


def vector_to_str(x, fmt='%.2f', max_cnt=0):
    if max_cnt == 0 or len(x) <= max_cnt:
        return '[' + ','.join([fmt] * len(x)) % tuple(x) + ']'
    v = x[0:max_cnt]
    return '[' + ','.join([fmt] * len(v)) % tuple(v) + ',...]'


def load_image_pixels(imagepath, resolution, input_shape):
    img = Image.open(imagepath)
    resized = img.resize(resolution)
    return np.array(resized).reshape(input_shape)


def draw_images_horz(xs, image_shape=None):
    show_cnt = len(xs)
    fig, axes = plt.subplots(1, show_cnt, figsize=(5, 5))
    for n in range(show_cnt):
        img = xs[n]
        if image_shape:
            x3d = img.reshape(image_shape)
            img = Image.fromarray(np.uint8(x3d))
        axes[n].imshow(img)
        axes[n].axis('off')
    plt.draw()
    plt.show()


def show_select_results(est, ans, target_names, max_cnt=0):
    for n in range(len(est)):
        pstr = vector_to_str(100 * est[n], '%2.0f', max_cnt)
        estr = target_names[np.argmax(est[n])]
        astr = target_names[np.argmax(ans[n])]
        rstr = 'O'
        if estr != astr:
            rstr = 'X'
        print('추정확률분포 {} => 추정 {} : 정답 {} => {}'
              .format(pstr, estr, astr, rstr))


def list_dir(path):
    filenames = os.listdir(path)
    filenames.sort()
    return filenames
