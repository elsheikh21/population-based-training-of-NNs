#-*- coding: utf-8 -*-
from __future__ import division
import operator
import os
import time
import tensorflow as tf
import numpy as np
import random
import re
import scipy.misc

from utils import *
from inception import *

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
#import tflib.inception_score

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)

class GAN(object):
    model_name = "GAN"     # name for checkpoint

    def __init__(self, worker_idx=-1, batch_size=64, z_dim=128, epochs=100):
        self.worker_idx = worker_idx
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.epochs = epochs

        # parameters
        self.input_height = 32
        self.input_width = 32
        self.output_height = 32
        self.output_width = 32

        self.z_dim = z_dim  # dimension of noise-vector
        self.c_dim = 3  # color dimension
        self.output_dim = self.output_height * self.output_width * self.c_dim

        # checkpoint dir
        self.checkpoint_dir = 'checkpoint'
        self.log_dir = 'logs'
        self.image_dir = 'images/{}'.format(worker_idx)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # train
        self.learning_rate_D = tf.get_variable('learning_rate_D', initializer=tf.constant(2e-4))
        self.learning_rate_G = tf.get_variable('learning_rate_G', initializer=tf.constant(2e-4))

        self.beta1 = 0.5

        # test
        self.sample_num = 64  # number of generated images to be saved

#        # load cifar10
#        self.data_X, self.data_y = load_cifar10('cifar10', preprocessing=False)
#        self.data_X = np.reshape(self.data_X, [-1, 32*32*3])
#        print("Shape of cifar10 X: {}".format(self.data_X.shape))
#        print("Shape of cifar10 Y: {}".format(self.data_y.shape))

        DATA_DIR = 'dataset/cifar-10-batches-py'
        DATA_SIZE = 60000

        self.train_gen, self.dev_gen = lib.cifar10.load(self.batch_size, data_dir=DATA_DIR)
        def inf_train_gen():
            while True:
                for images,_ in self.train_gen():
                    yield images

        self.gen = inf_train_gen()

        # get number of batches for a single epoch
        self.num_batches = DATA_SIZE // self.batch_size

#        # load pretrained inception network (code from tensorflow / openAI)
        self.init_inception()
        self.inception_score = tf.Variable(0, dtype=tf.float32)

    def Discriminator(self, inputs):
        # architecture from: https://github.com/igul222/improved_wgan_training
        output = tf.reshape(inputs, [-1, 3, 32, 32])

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, self.z_dim, 5, output, stride=2)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', self.z_dim, 2*self.z_dim, 5, output, stride=2)
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
        output = LeakyReLU(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*self.z_dim, 4*self.z_dim, 5, output, stride=2)
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
        output = LeakyReLU(output)

        output = tf.reshape(output, [-1, 4*4*4*self.z_dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*self.z_dim, 1, output)

        return tf.reshape(output, [-1])

    def Generator(self, n_samples, noise=None):
        # architecture from: https://github.com/igul222/improved_wgan_training
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*self.z_dim, noise)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4*self.z_dim, 4, 4])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*self.z_dim, 2*self.z_dim, 5, output)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*self.z_dim, self.z_dim, 5, output)
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', self.z_dim, 3, 5, output)

        output = tf.tanh(output)

        return tf.reshape(output, [-1, self.output_dim])

    def build_model(self):

        self.inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.output_dim])
        real_data = 2*((tf.cast(self.inputs, tf.float32)/255.)-.5)
        G = self.Generator(self.batch_size)

        D_real = self.Discriminator(real_data)
        D_fake = self.Discriminator(G)

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        d_loss_fake =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        self.d_loss = (d_loss_real + d_loss_fake) / 2.

        self.g_optim = tf.train.AdamOptimizer(self.learning_rate_G, beta1=0.5).minimize(self.g_loss, var_list=lib.params_with_name('Generator'))
        self.d_optim = tf.train.AdamOptimizer(self.learning_rate_D, beta1=0.5).minimize(self.d_loss, var_list=lib.params_with_name('Discriminator'))


        self.z = tf.constant(np.random.normal(size=(self.batch_size, 128)).astype('float32'))
        self.fake_images = self.Generator(self.batch_size, noise=self.z)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        d_lr_sum = tf.summary.scalar("d_learning_rate", self.learning_rate_D)
        g_lr_sum = tf.summary.scalar("g_learning_rate", self.learning_rate_G) 
        d_log_lr_sum = tf.summary.scalar("d_log_learning_rate", tf.log(self.learning_rate_D))
        g_log_lr_sum = tf.summary.scalar("g_log_learning_rate", tf.log(self.learning_rate_G))

        inception_sum = tf.summary.scalar("inception_score", self.inception_score)
        self.inception_sum = tf.summary.merge([inception_sum])
        self.samples_100 = self.Generator(100)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum, g_lr_sum, g_log_lr_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum, d_lr_sum, d_log_lr_sum])

        # define explore graph
        coin_flip_D = tf.cast(tf.random_uniform(shape=[], minval=0, maxval=1+1, dtype=tf.int32), tf.float32)
        coin_flip_G = tf.cast(tf.random_uniform(shape=[], minval=0, maxval=1+1, dtype=tf.int32), tf.float32)

        self.explore_learning_D = tf.assign(self.learning_rate_D, 2*self.learning_rate_D*coin_flip_D + 0.5*self.learning_rate_D*(1-coin_flip_D))  
        self.explore_learning_G = tf.assign(self.learning_rate_G, 2*self.learning_rate_G*coin_flip_G + 0.5*self.learning_rate_G*(1-coin_flip_G))  

        # update inception tensor
        self.inception_value = tf.placeholder(tf.float32)
        self.update_inception = tf.assign(self.inception_score, self.inception_value)

        # variable mappings due to code merge

        self.disc_cost = self.d_loss 
        self.gen_cost = self.g_loss
        self.disc_train_op = self.d_optim 
        self.real_data_int =  self.inputs
        self.gen_train_op = self.g_optim

    def train(self):

        # Dataset iterators

        DATA_DIR = 'dataset/cifar-10-batches-py'
        train_gen, dev_gen = lib.cifar10.load(64, data_dir=DATA_DIR)
        def inf_train_gen():
            while True:
                for images,_ in train_gen():
                    yield images

        self.mon_sess.run(tf.initialize_all_variables())
        gen = inf_train_gen()

        for iteration in xrange(100000):
            start_time = time.time()
            # Train generator
            if iteration > 0:
                _ = self.mon_sess.run(self.gen_train_op)
            # Train critic
            disc_iters = 1
            for i in xrange(disc_iters):
                _data = gen.next()
                _disc_cost, _ = self.mon_sess.run([self.disc_cost, self.disc_train_op], feed_dict={self.real_data_int: _data})

            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)

            # Calculate inception score every 1K iters
            if iteration % 1000 == 999:
                inception_score = self.get_inception_score()
                lib.plot.plot('inception score', inception_score[0])

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 99:
                dev_disc_costs = []
                for images,_ in dev_gen():
                    _dev_disc_cost = self.mon_sess.run(self.disc_cost, feed_dict={self.real_data_int: images}) 
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                self.generate_image(iteration, _data)


            # Save logs every 100 iters
            if (iteration < 5) or (iteration % 100 == 99):
                lib.plot.flush()

            lib.plot.tick()

    def step(self, idx, epoch, verbose=True):

        start_time = time.time()
        self.counter = epoch*self.num_batches + idx

        # Train generator
        _gen_cost, _, summary_str = self.mon_sess.run([self.gen_cost, self.gen_train_op, self.g_sum])
        self.writer.add_summary(summary_str, self.counter)

        # Train critic
        _data = self.gen.next()
        _disc_cost, _, summary_str = self.mon_sess.run([self.disc_cost, self.disc_train_op, self.d_sum], feed_dict={self.real_data_int: _data})
        self.writer.add_summary(summary_str, self.counter)

        # Calculate dev loss and generate samples every 100 iters
        if self.counter % 100 == 99:
            dev_disc_costs = []
            for images,_ in self.dev_gen():
                _dev_disc_cost = self.mon_sess.run(self.disc_cost, feed_dict={self.real_data_int: images}) 
                dev_disc_costs.append(_dev_disc_cost)
                self.generate_image(self.counter, _data)

        if verbose:
            # display training status
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                  % (epoch, idx, self.num_batches, time.time() - start_time, _disc_cost, _gen_cost))

    def eval(self):
        """evaluate the inception score"""
        return self.get_inception_score()

    def exploit(self, worker_idx, score, strategy="TS"):
        """exploit using Truncation Selection (TS) or Binary Tournament (BS)"""

        do_explore = False

        if strategy == "TS":
            # rank all agents, if agent is in the bottom 20% of the population
            # sample another agent uniformly from the top 20% of the population
            # copying its weights and hyperparameters
            num_workers = len(os.listdir(self.checkpoint_dir)) + 1 # for p
            percentile_20 = int(np.ceil(num_workers * 0.2))

            p = ('current', score) 
            ranked_list = self.rank_workers(p)

            # (worker, score)
            top_20 = ranked_list[-percentile_20:]
            bottom_20 = ranked_list[:percentile_20]
            
            # worker
            top_20 = [i[0] for i in top_20]
            bottom_20 = [i[0] for i in bottom_20]

            if 'current' in bottom_20:
                exploit_idx = random.choice(top_20)

                if exploit_idx != 'current':
                    do_explore = True
                    print("Worker {} (EXPLOIT): inheriting Worker {}'s weights/hyperparams".format(worker_idx, exploit_idx))
                    self.load(exploit_idx) 
            else:
                print("Worker {} (EXPLOIT): is not in the bottom 20, no action".format(worker_idx))
 
        elif strategy == "BS":
            raise NotImplementedError
        else:
            raise ValueError

        return do_explore

    def explore(self, worker_idx):
        self.mon_sess.run([self.explore_learning_D, self.explore_learning_G]) 
        print("Worker {} (EXPLORE)".format(worker_idx))

    def rank_workers(self, p):
        """exploit takes (h,w,p,P)
        the p is not in P!
        """

        num_workers = len(os.listdir(self.checkpoint_dir))
        ranked_dict = {}

        regex = re.compile('(\d+)_(\d*\.?\d*)')
        for i in range(num_workers):
            cpkt_dir = os.path.join(self.checkpoint_dir, str(i))
            if not os.path.exists(cpkt_dir):
                os.makedirs(cpkt_dir)
            for f in os.listdir(cpkt_dir):
                m = regex.match(f)
                if m:
                    ranked_dict[i] = float(m.group(2))
        ranked_dict[p[0]] = p[1]

        # technically a list of tuples
        print(ranked_dict)
        ranked_list = sorted(ranked_dict.items(), key=operator.itemgetter(1))
        return ranked_list
        

    def save(self, worker_idx, score):
        # update inception tensor for summary
        self.mon_sess.run(self.update_inception, {self.inception_value: score})

        worker_dir = os.path.join(self.checkpoint_dir, str(worker_idx))
        if not os.path.exists(worker_dir):
            os.makedirs(worker_dir)

        name = '{}_{}_{}.model'.format(worker_idx, score, self.counter)
        self.saver.save(self.get_session(), os.path.join(worker_dir, name))

        # tensorboard
        summary_str = self.mon_sess.run(self.inception_sum)
        self.writer.add_summary(summary_str, self.counter)


    def load(self, worker_idx):
        
        worker_dir = os.path.join(self.checkpoint_dir, str(worker_idx))
        ckpt = tf.train.get_checkpoint_state(worker_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.mon_sess, os.path.join(worker_dir, ckpt_name))
            print("Successfully loaded checkpoint from Worker {}!".format(worker_idx))
        else:
            print("Could not find checkpoint")

    def load_saved_session(self):
        print("Loading Initial Checkpoints...") 
        epoch = 0
        idx = 0

        worker_dir = os.path.join(self.checkpoint_dir, str(self.worker_idx))
        ckpt = tf.train.get_checkpoint_state(worker_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.mon_sess, os.path.join(worker_dir, ckpt_name))


            regex = re.compile('(\d+)_(\d*\.?\d*)_(\d+)')
            counter = int(regex.match(ckpt_name).group(3))

            epoch = counter // self.num_batches
            idx = counter - epoch * self.num_batches


            print("Successfully loaded checkpoint from epoch {} idx {}".format(epoch, idx))
        else:
            print("Could not find checkpoint")

        return epoch, idx

    def generate_image(self, frame, true_dist):
        samples = self.mon_sess.run(self.fake_images)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((-1, 3, 32, 32)), 'images/{}/samples_{}.jpg'.format(self.worker_idx, frame))

    def init_inception(self):
        return init_inception(self)

    def get_inception_score(self):
        all_samples = []
        for i in xrange(10):
            all_samples.append(self.mon_sess.run(self.samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255./2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
        return get_inception_score(self, list(all_samples))#lib.inception_score.get_inception_score(list(all_samples))

    def get_session(self):
        """
        MonitoredTrainingSession only supports hooks and not custom saver objects
        The control for these hooks is before_run and after_run, which is not enough control (https://github.com/tensorflow/tensorflow/issues/8425)
        """
        session = self.mon_sess
        while type(session).__name__ != 'Session':
            session = session._sess
        return session
