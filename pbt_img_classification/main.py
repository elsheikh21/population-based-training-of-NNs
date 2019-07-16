from functools import lru_cache
import tensorflow.contrib as tfc
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from dataset_handler import load_data
from plotting import plot

tf.reset_default_graph()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train, test, handle, inputs, labels = load_data()


def make_handle(sess, dataset):
    # To enumerate through the dataset, it will return an uninitialized iterator
    iterator = dataset.make_initializable_iterator()

    handle, _ = sess.run([iterator.string_handle(), iterator.initializer])
    return handle


# normalize the inputs and flatten them
inputs = tf.cast(inputs, tf.float32) / 255.0
inputs = tf.layers.flatten(inputs)
# cast labels to integers each representing a class
labels = tf.cast(labels, tf.int32)


class Model:
    """
    Creating a model with a given id and name scope to be able
    to debug the graph later.
    All models are created with a regularizer by default

    Models are formed of 2 dense layer 1024 units each, RELU activation,
    and the last layer consists of 10 units representing 10 classes,
    all are regularized using L1 Regularizer.
    We take the probability distributions produced by the logits layer,
    transform them to categorical to get likelihood of each class.
    And compute accuracy, and loss add to it the regularizer loss and optimize
    that using Adam optimizer

    L1 Regularizer is created to change one unit to its half or double the
    effective scale

    PBT part, it is implemented to add noise of uniform random distribution of
    mean 0.0 and standard deviation of 0.5

    """

    def __init__(self, model_id: int, regularize=True):
        self.model_id = model_id
        self.name_scope = tf.get_default_graph().get_name_scope()

        # Regularization
        if regularize:
            l1_reg = self._create_regularizer()
        else:
            l1_reg = None

        # Network and loglikelihood
        logits = self._create_network(l1_reg)
        # We maximize the loglikelihood of the data as a training objective
        distr = tf.distributions.Categorical(logits)
        loglikelihood = distr.log_prob(labels)

        # Define accuracy of prediction
        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(prediction, labels), tf.float32))

        # Loss and optimization
        self.loss = -tf.reduce_mean(loglikelihood)
        # Retrieve all weights and hyper-parameter variables of this model
        trainable = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name_scope + '/')
        # The loss to optimize is the negative loglikelihood
        # + the l1-regularizer
        reg_loss = self.loss + tf.losses.get_regularization_loss()
        self.optimize = tf.train.AdamOptimizer().minimize(reg_loss,
                                                          var_list=trainable)

    def _create_network(self, l1_reg):
        # Our deep neural network will have two hidden layers with
        # plenty of units
        hidden = tf.layers.dense(inputs, 1024, activation=tf.nn.relu,
                                 kernel_regularizer=l1_reg)
        hidden = tf.layers.dense(hidden, 1024, activation=tf.nn.relu,
                                 kernel_regularizer=l1_reg)
        logits = tf.layers.dense(hidden, 10,
                                 kernel_regularizer=l1_reg)
        return logits

    def _create_regularizer(self):
        # We will define the l1 regularizer scale in log2 space
        # This allows changing one unit to half or
        # double the effective l1 scale
        self.l1_scale = tf.get_variable('l1_scale',
                                        [],
                                        tf.float32,
                                        trainable=False,
                                        initializer=tf.constant_initializer(np.log2(1e-5)))
        # We define a 'perturb' operation that adds some noise
        # to our regularizer scale. We will use this perturbation
        # during exploration in our population based training
        noise = tf.random_normal([], stddev=0.5)
        self.perturb = self.l1_scale.assign_add(noise)

        return tfc.layers.l1_regularizer(2 ** self.l1_scale)

    @lru_cache(maxsize=None)
    def copy_from(self, other_model):
        '''
        This method is used for exploitation. We copy all weights
        and hyper-parameters, from other_model to this model
        '''
        my_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, self.name_scope + '/')
        their_weights = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, other_model.name_scope + '/')
        assign_ops = [mine.assign(theirs).op for mine,
                      theirs in zip(my_weights, their_weights)]
        return tf.group(*assign_ops)


def create_model(*args, **kwargs):
    with tf.variable_scope(None, 'model'):
        return Model(*args, **kwargs)


# Iterate for 50K epochs
ITERATIONS = 50_000

# Creating a list of accuracy, all are assigned to zeros for simplicity
nonreg_accuracy_hist = np.zeros((ITERATIONS // 100,))
mdl = create_model(0, regularize=False)

# configuring GPU options
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                      allow_soft_placement=True,
                                      log_device_placement=True)) as sess:
    train_handle = make_handle(sess, train)
    test_handle = make_handle(sess, test)
    sess.run(tf.global_variables_initializer())

    feed_dict = {handle: train_handle}
    test_feed_dict = {handle: test_handle}
    for i in tqdm(range(ITERATIONS), desc="Training"):
        # Training
        sess.run(mdl.optimize, feed_dict)
        # Evaluate
        if i % 100 == 0:
            nonreg_accuracy_hist[i //
                                 100] = sess.run(mdl.accuracy,
                                                 test_feed_dict)

# Population Based Training
# Creating a population of 10 Networks,
# assign best and worst percentiles to 30%
# Each will perform 500 steps each iteration for 100 iterations
POPULATION_SIZE = 10
BEST_THRES = 3
WORST_THRES = 3
POPULATION_STEPS = 500
ITERATIONS = 100

# Creating lists for holding data for plotting later
accuracy_hist = np.zeros((POPULATION_SIZE, POPULATION_STEPS))
l1_scale_hist = np.zeros((POPULATION_SIZE, POPULATION_STEPS))

best_accuracy_hist = np.zeros((POPULATION_STEPS,))
best_l1_scale_hist = np.zeros((POPULATION_STEPS,))

# Each model is created
models = [create_model(i) for i in tqdm(
    range(POPULATION_SIZE), desc="Creating models")]


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                      allow_soft_placement=True,
                                      log_device_placement=True)) as sess:
    train_handle = make_handle(sess, train)
    test_handle = make_handle(sess, test)
    sess.run(tf.global_variables_initializer())

    feed_dict = {handle: train_handle}
    test_feed_dict = {handle: test_handle}
    for i in tqdm(range(POPULATION_STEPS), desc="PBT"):
        # Copy best
        sess.run([m.copy_from(models[0]) for m in models[-WORST_THRES:]])
        # Perturb others
        sess.run([m.perturb for m in models[BEST_THRES:]])
        # Training
        for _ in tqdm(range(ITERATIONS), desc="Training"):
            sess.run([m.optimize for m in models], feed_dict)
        # Evaluate
        l1_scales = sess.run({m: m.l1_scale for m in models})
        accuracies = sess.run({m: m.accuracy for m in models}, test_feed_dict)
        models.sort(key=lambda m: accuracies[m], reverse=True)
        # Logging
        best_accuracy_hist[i] = accuracies[models[0]]
        best_l1_scale_hist[i] = l1_scales[models[0]]
        for m in models:
            l1_scale_hist[m.model_id, i] = l1_scales[m]
            accuracy_hist[m.model_id, i] = accuracies[m]

plot(best_accuracy_hist, nonreg_accuracy_hist, l1_scale_hist)
