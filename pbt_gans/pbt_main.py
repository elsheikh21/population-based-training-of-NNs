import argparse
import os
import pickle
import numpy as np
import sys
import time
import tensorflow as tf

#from gan_class import *
#from wgan_class_pbt_model import *
from gan_class_v2 import *

from inception import *

# for GAN v1
from utils import *

# for GAN v2
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.cifar10
#import tflib.inception_score
import tflib.plot

tf.logging.set_verbosity(tf.logging.INFO)

def main(_):
    # we need to provide all ps and worker info to each server so they are aware of each other
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    
    # create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    
    # create and start a server for the local task.
    
#    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#    gpu_options = tf.GPUOptions(allow_growth=True)

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
    server = tf.train.Server(cluster,
                            job_name=FLAGS.job_name,
                            task_index=FLAGS.task_index,
                            config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
                            
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":

        # explictely place weights and hyperparameters on the worker servers to prevent sharing
        # otherwise replica_device_setter will put them on the ps
        with tf.device("/job:worker/task:{}".format(FLAGS.task_index)):

            # [step 1] initialize all worker variables
            # pbt-hyperparams and weights MUST go here
            gan = GAN(worker_idx=FLAGS.task_index, epochs=200)

            # [step 2] define graph here
            # we don't define on the ps because we don't share weights
            gan.build_model()

            # use filesystem for population
            gan.saver = tf.train.Saver(max_to_keep=1)

            # log each worker separately for tensorboard
            gan.writer = tf.summary.FileWriter(os.path.join(gan.log_dir, str(FLAGS.task_index)), tf.get_default_graph()) 

            # show all variables
            show_all_variables()
            

#            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
#            gpu_options = tf.GPUOptions(allow_growth=True)

        gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
        with tf.train.MonitoredTrainingSession(master=server.target,
                                            is_chief=True, config=tf.ConfigProto(gpu_options=gpu_options)) as gan.mon_sess:

            ready_freq = 5000
            im_save_freq = 100
            update_freq = 1000

            # restore session
            start_epoch, start_idx = gan.load_saved_session()

            for epoch in range(start_epoch, gan.epochs):

                for idx in range(start_idx, gan.num_batches):
                    counter = epoch*gan.num_batches + idx

                    start_idx = 0 # so the next loop doesn't start from here

                    if idx == 0: # print out loss once per epoch 
                        gan.step(idx, epoch, verbose=True)
                    else:
                        gan.step(idx, epoch, verbose=False)

                    # inception score takes ~5s, so there is a tradeoff
                    if counter % ready_freq == 0:
                        inception_score, _ = gan.eval()
                        print("Worker {} with Inception Score {}".format(FLAGS.task_index, inception_score))

                        do_explore = gan.exploit(worker_idx=FLAGS.task_index, score=inception_score)

                        if do_explore:
                            gan.explore(FLAGS.task_index)
#                            inception_score, _ = gan.eval()
#                            print("Worker {} with Inception Score {}".format(FLAGS.task_index, inception_score))

                    if counter % update_freq == 0:
                        # update checkpoint (ideally checkpoint every idx)
                        inception_score, _ = gan.eval()
                        gan.save(worker_idx=FLAGS.task_index, score=inception_score)
 

if __name__ == "__main__":
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

