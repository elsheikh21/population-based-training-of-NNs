import numpy as np
import observations
import tensorflow as tf


def load_data():
    """
    1. Load the CIFAR-10 data set.
    It consists of 32x32 RGB images in 10 classes, with 6000 images per class.
    There are 50000 training images and 10000 test images.

    2. Converts first row to int8 as they represent different classes
        airplane : 0, automobile : 1, bird : 2, cat : 3, deer : 4, dog : 5, frog : 6, horse : 7, ship : 8, truck : 9

    3. Creates a dataset with a separate element for each row of
        the input tensor,
        repeating dataset indefinitely and shuffling it,
        then forming batches of 64 images

    4. Creating an iterator to iterate over the inputs and labels

    Returns:
        tuples of train, test, handler, labels

    """
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
