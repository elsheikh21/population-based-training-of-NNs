import argparse
import os
import pickle
import numpy as np
import sys
import time
import tensorflow as tf
from tqdm import tqdm

#from gan_class import *
#from wgan_class_pbt_model import *
#from gan_class_v2 import *
from gan_class_sequential import *


from inception import *

# for GAN v1
from utils import *

# for GAN v2
import tflib_defs as lib
import tflib_defs.ops.linear
import tflib_defs.ops.conv2d
import tflib_defs.ops.batchnorm
import tflib_defs.ops.deconv2d
import tflib_defs.save_images
import tflib_defs.cifar10
#import tflib.inception_score
import tflib.plot

from tqdm import tqdm


tf.logging.set_verbosity(tf.logging.INFO)


def main(_):

    lib.print_model_settings(locals().copy())
    gpu_options = tf.GPUOptions(allow_growth=True)

    num_workers = 5
    graph_list = []
    sess_list = []
    gan_list = []

    for i in range(num_workers):
        graph = tf.Graph()
        graph_list.append(graph)


        sess_list.append(tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)))

    # load the dataset once

    data_X, data_y = load_cifar10('cifar10')
    data_X = np.reshape(data_X, [-1, 32*32*3])

    for i in tqdm(range(num_workers), desc='Creating GANs population'):
        graph = graph_list[i]

        with graph.as_default():
            mon_sess = sess_list[i]

            gan = GAN(worker_idx=i, epochs=200, data_X=data_X, data_y=data_y)
            gan.mon_sess = mon_sess

            gan.build_model()
            gan.mon_sess.run(tf.global_variables_initializer())
            # use filesystem for population
            gan.saver = tf.train.Saver(max_to_keep=1)

            # restore session
            start_epoch, start_idx = gan.load_saved_session()

            # log each worker separately for tensorboard
            gan.writer = tf.summary.FileWriter(os.path.join(
                gan.log_dir, str(i)), tf.get_default_graph())

            # show all variables
            show_all_variables()

            # add gan object
            gan_list.append(gan)

    #        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
        ready_freq = 2000
        im_save_freq = 100
        update_freq = 1000

    for epoch in tqdm(range(start_epoch, gan.epochs), desc='Training'):
        for idx in range(start_idx, gan.num_batches):
            for i in range(num_workers):

                graph = graph_list[i]
                gan = gan_list[i]

                with graph.as_default():
                    mon_sess = sess_list[i]

                    counter = epoch*gan.num_batches + idx

                    start_idx = 0  # so the next loop doesn't start from here

                    if idx == 0:  # print out loss once per epoch
                        gan.step(idx, epoch, verbose=True)
                    else:
                        gan.step(idx, epoch, verbose=False)

                    # inception score takes ~5s, so there is a tradeoff
                    if counter % ready_freq == 0:
                        inception_score, _ = gan.eval()
                        print("Worker {} with Inception Score {}".format(
                            i, inception_score))

                        do_explore = gan.exploit(
                            worker_idx=i, score=inception_score)

                        if do_explore:
                            gan.explore(i)
#                            inception_score, _ = gan.eval()
#                            print("Worker {} with Inception Score {}".format(FLAGS.task_index, inception_score))

                    if counter % update_freq == 0:
                        # update checkpoint (ideally checkpoint every idx)
                        inception_score, _ = gan.eval()
                        gan.save(worker_idx=i, score=inception_score)
                        print("Worker {} with Inception Score {}".format(
                            i, inception_score))


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser()

    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )

    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
