import argparse
import sys
import cv2
import time
sys.path.append("Wrapped Game Code/")
import pong_fun as game # whichever is imported "as game" will be used
import dummy_game
#import tetris_fun as game
import random
import numpy as np
from collections import deque

import tensorflow as tf

#####################################################################################################

# Create Local Game Variables
GAME = 'pong' # the name of the game being played for log files
ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 500. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 25000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others

FLAGS = None

#####################################################################################################

# Define layers and make the network configuration

def weight_variable(shape):
    initial = tf.random_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

#####################################################################################################


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):

        # Build model...
#####################################################################################################
        #sess = tf.InteractiveSession()
        print('Checkpoint 1 reached')
        s, readout, h_fc1 = createNetwork()
        a = tf.placeholder("float", [None, ACTIONS])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        #train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        global_step = tf.contrib.framework.get_or_create_global_step()
        train_op = tf.train.AdagradOptimizer(0.01).minimize(cost, global_step=global_step)
        game_state = game.GameState()

        D =  deque()
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal, bar1_score, bar1_score = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

        #saver = tf.train.Saver()
        #var_list = tf.contrib.framework.list_variables(config.pre_model_dir)
        #for v in var_list:
            #print(v)

        epsilon = INITIAL_EPSILON
        t = 0
        episode = 0
        tick = time.time()
        print('Step 1 complete')

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="",
                                           hooks=hooks) as mon_sess:
        print('Step 2 started')
        #mon_sess.run(tf.initialize_all_variables())
        print('Step 2 complete')
        while not mon_sess.should_stop():
            
            print('Step 2 Done')
            while True:
                # choose an action epsilon greedily
                #print('Checkpoint 3 reached')
                readout_t = readout.eval(feed_dict = {s : [s_t]}, session=mon_sess)[0]
                a_t = np.zeros([ACTIONS])
                action_index = 0

                if random.random() <= epsilon or t <= OBSERVE:
                        action_index = random.randrange(ACTIONS)
                        a_t[action_index] = 1
                else:
                        action_index = np.argmax(readout_t)
                        a_t[action_index] = 1

                if epsilon > FINAL_EPSILON and t > OBSERVE:
                        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                for i in range(0, K):
                        # run the selected action and observe next state and reward
                        x_t1_col, r_t, terminal, bar1_score, bar1_score  = game_state.frame_step(a_t)
                        if(terminal == 1):
                            episode +=1
                        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
                        ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
                        x_t1 = np.reshape(x_t1, (80, 80, 1))
                        s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

                        # store the transition in D
                        D.append((s_t, a_t, r_t, s_t1, terminal))
                        if len(D) > REPLAY_MEMORY:
                            D.popleft()
                # only train if done observing
                if t > OBSERVE:
                        # sample a minibatch to train on
                        minibatch = random.sample(D, BATCH)

                        # get the batch variables
                        s_j_batch = [d[0] for d in minibatch]
                        a_batch = [d[1] for d in minibatch]
                        r_batch = [d[2] for d in minibatch]
                        s_j1_batch = [d[3] for d in minibatch]

                        y_batch = []
                        readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch}, session=mon_sess)
                        for i in range(0, len(minibatch)):
                            # if terminal only equals reward
                            if minibatch[i][4]:
                                y_batch.append(r_batch[i])
                            else:
                                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                        train_op.run(feed_dict = {
                            y : y_batch,
                            a : a_batch,
                            s : s_j_batch}, session=mon_sess)

                # update the old values
                s_t = s_t1
                t += 1

                # save progress every 10000 iterations
                #if t % 10000 == 0:
                    #saver.save(mon_sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

                # print info
                state = ""
                if t <= OBSERVE:
                        state = "observe"
                elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                        state = "explore"
                else:
                        state = "train"
                #print ("TIMESTEP", t, "/episode", episode, "/ LINES", game_state.total_lines, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
                if r_t!= 0:
                        print ("TIMESTEP", t, "/episode",episode, "/bar1_score", bar1_score, "/bar2_score", bar2_score, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
                if((bar1_score - bar2_score) > 16 )
                    print("GAME_ENDS_in_Time",int(time.time() -tick))
                    break;
                if((bar1_score - bar2_score) > 14 ):
                    print("GAME_MID_in_Time",int(time.time() -tick))
        # Run a training step asynchronously.
        # See <a href="./../api_docs/python/tf/train/SyncReplicasOptimizer"><code>tf.train.SyncReplicasOptimizer</code></a> for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
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
