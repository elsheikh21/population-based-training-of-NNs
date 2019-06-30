import numpy as np
import observations
import tensorflow as tf


def load_data():
    train_data, test_data = observations.cifar10('data/cifar',)
    test_data = test_data[0], test_data[1].astype(
        np.uint8)  # Fix test_data dtype

    train = tf.data.Dataset.from_tensor_slices(
        train_data).repeat().shuffle(10000).batch(64)
    test = tf.data.Dataset.from_tensors(test_data).repeat()

    handle = tf.placeholder(tf.string, [])
    itr = tf.data.Iterator.from_string_handle(
        handle, train.output_types, train.output_shapes)
    inputs, labels = itr.get_next()

    return train, test, handle, inputs, labels
