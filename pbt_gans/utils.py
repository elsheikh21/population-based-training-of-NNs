"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""
from __future__ import division
import math
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import os, gzip

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim

try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle

def one_hot(x, n):
    x = np.array(x)
    assert x.ndim == 1
    return np.eye(n)[x]

def load_cifar10(dataset, num_training=49000, num_validation=1000, num_test=10000):
    '''
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the neural net classifier.
    '''
    # Load the raw CIFAR-10 data
    X_train, y_train, X_test, y_test = load(os.path.join('dataset', 'cifar-10-batches-py'))

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    X_train = X_train.astype(np.float64) / 255.
    X_val = X_val.astype(np.float64) / 255.
    X_test = X_test.astype(np.float64) / 255.

    y_train = one_hot(y_train, 10)
    y_val = one_hot(y_val, 10)
    y_test = one_hot(y_test, 10)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

#    mean_image = np.mean(X_train, axis=0)
#    std = np.std(X_train)

#    X_train -= mean_image
#    X_val -= mean_image
#    X_test -= mean_image
#
#    X_train /= std
#    X_val /= std
#    X_test /= std

    return np.concatenate((X_train, X_val, X_test)), np.concatenate((y_train, y_val, y_test))

def load_CIFAR_batch(filename):
    ''' load single batch of cifar '''
    with open(filename, 'rb') as f:
        # unpickle a python2 object in python3
        datadict = pickle.load(f) #, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y

def load(ROOT):
    ''' load all of cifar '''
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

def load_mnist(dataset_name):
    data_dir = os.path.join("./data", dataset_name)

    def extract_data(filename, num_data, head_size, data_size):
        with gzip.open(filename) as bytestream:
            bytestream.read(head_size)
            buf = bytestream.read(data_size * num_data)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
        return data

    data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
    trX = data.reshape((60000, 28, 28, 1))

    data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
    trY = data.reshape((60000))

    data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
    teX = data.reshape((10000, 28, 28, 1))

    data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
    teY = data.reshape((10000))

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, crop)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def save_images_2(images, size, image_path):
    return imsave(images, size, image_path)

def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

""" Drawing Tools """
# borrowed from https://github.com/ykwon0407/variational_autoencoder/blob/master/variational_bayes.ipynb
def save_scattered_image(z, id, z_range_x, z_range_y, name='scattered_image.jpg'):
    N = 10
    plt.figure(figsize=(8, 6))
    plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(N, 'jet'))
    plt.colorbar(ticks=range(N))
    axes = plt.gca()
    axes.set_xlim([-z_range_x, z_range_x])
    axes.set_ylim([-z_range_y, z_range_y])
    plt.grid(True)
    plt.savefig(name)

# borrowed from https://gist.github.com/jakevdp/91077b0cae40f8f8244a
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def rescale(images):
    # rescale from 0-1 to 0-255
    batch_img_list = []
    for idx in range(64):
        vMin = np.amin(images[idx])
        vMax = np.amax(images[idx])
        img_arr = images[idx].reshape(32*32*3,1) # flatten
        for i, v in enumerate(img_arr):
            img_arr[i] = (v-vMin)/(vMax-vMin)
        img_arr = img_arr.reshape(32,32,3) # M*N*3
        batch_img_list.append(img_arr)
    return np.array(batch_img_list)

def save_matplot_img(images, size, image_path):
    # value must set between 0. with 1.
    for idx in range(64):
        vMin = np.amin(images[idx])
        vMax = np.amax(images[idx])
        img_arr = images[idx].reshape(32*32*3,1) # flatten
        for i, v in enumerate(img_arr):
            img_arr[i] = (v-vMin)/(vMax-vMin)
        img_arr = img_arr.reshape(32,32,3) # M*N*3

        # matplot display
        plt.subplot(8,8,idx+1),plt.imshow(img_arr, interpolation='nearest')
        #plt.title("pred.:{}".format(np.argmax(self.data_y[0]),fontsize=10))
        plt.axis("off")

    plt.savefig(image_path)
