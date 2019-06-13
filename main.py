import tensorflow as tf
import tensorflow.contrib as tfc
import numpy as np
import matplotlib.pyplot as plt
import observations
from functools import lru_cache
from tqdm import tqdm


tf.reset_default_graph()

train_data, test_data = observations.cifar10('data/cifar',)
test_data = test_data[0], test_data[1].astype(np.uint8)  # Fix test_data dtype

train = tf.data.Dataset.from_tensor_slices(
    train_data).repeat().shuffle(10000).batch(64)
test = tf.data.Dataset.from_tensors(test_data).repeat()


handle = tf.placeholder(tf.string, [])
itr = tf.data.Iterator.from_string_handle(
    handle, train.output_types, train.output_shapes)
inputs, labels = itr.get_next()


def make_handle(sess, dataset):
    iterator = dataset.make_initializable_iterator()
    handle, _ = sess.run([iterator.string_handle(), iterator.initializer])
    return handle


inputs = tf.cast(inputs, tf.float32) / 255.0
inputs = tf.layers.flatten(inputs)
labels = tf.cast(labels, tf.int32)


class Model:
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
        # We maximixe the loglikelihood of the data as a training objective
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
        # The loss to optimize is the negative loglikelihood + the l1-regularizer
        reg_loss = self.loss + tf.losses.get_regularization_loss()
        self.optimize = tf.train.AdamOptimizer().minimize(reg_loss, var_list=trainable)

    def _create_network(self, l1_reg):
        # Our deep neural network will have two hidden layers with plenty of units
        hidden = tf.layers.dense(inputs, 1024, activation=tf.nn.relu,
                                 kernel_regularizer=l1_reg)
        hidden = tf.layers.dense(hidden, 1024, activation=tf.nn.relu,
                                 kernel_regularizer=l1_reg)
        logits = tf.layers.dense(hidden, 10,
                                 kernel_regularizer=l1_reg)
        return logits

    def _create_regularizer(self):
        # We will define the l1 regularizer scale in log2 space
        # This allows changing one unit to half or double the effective l1 scale
        self.l1_scale = tf.get_variable('l1_scale', [], tf.float32, trainable=False,
                                        initializer=tf.constant_initializer(np.log2(1e-5)))
        # We define a 'pertub' operation that adds some noise to our regularizer scale
        # We will use this pertubation during exploration in our population based training
        noise = tf.random_normal([], stddev=0.5)
        self.perturb = self.l1_scale.assign_add(noise)

        return tfc.layers.l1_regularizer(2 ** self.l1_scale)

    @lru_cache(maxsize=None)
    def copy_from(self, other_model):
        # This method is used for exploitation. We copy all weights and hyper-parameters
        # from other_model to this model
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


ITERATIONS = 50_000

nonreg_accuracy_hist = np.zeros((ITERATIONS // 100,))
model = create_model(0, regularize=False)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                      allow_soft_placement=True,
                                      log_device_placement=True)) as sess:
    train_handle = make_handle(sess, train)
    test_handle = make_handle(sess, test)
    sess.run(tf.global_variables_initializer())

    feed_dict = {handle: train_handle}
    test_feed_dict = {handle: test_handle}
    for i in tqdm(range(ITERATIONS)):
        # Training
        sess.run(model.optimize, feed_dict)
        # Evaluate
        if i % 100 == 0:
            nonreg_accuracy_hist[i //
                                 100] = sess.run(model.accuracy, test_feed_dict)


POPULATION_SIZE = 10
BEST_THRES = 3
WORST_THRES = 3
POPULATION_STEPS = 500
ITERATIONS = 100

accuracy_hist = np.zeros((POPULATION_SIZE, POPULATION_STEPS))
l1_scale_hist = np.zeros((POPULATION_SIZE, POPULATION_STEPS))
best_accuracy_hist = np.zeros((POPULATION_STEPS,))
best_l1_scale_hist = np.zeros((POPULATION_STEPS,))

models = [create_model(i) for i in range(POPULATION_SIZE)]

with tf.Session() as sess:
    train_handle = make_handle(sess, train)
    test_handle = make_handle(sess, test)
    sess.run(tf.global_variables_initializer())

    feed_dict = {handle: train_handle}
    test_feed_dict = {handle: test_handle}
    for i in tqdm(range(POPULATION_STEPS), desc='PBT'):
        # Copy best
        sess.run([m.copy_from(models[0]) for m in models[-WORST_THRES:]])
        # Perturb others
        sess.run([m.perturb for m in models[BEST_THRES:]])
        # Training
        for _ in range(ITERATIONS):
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
