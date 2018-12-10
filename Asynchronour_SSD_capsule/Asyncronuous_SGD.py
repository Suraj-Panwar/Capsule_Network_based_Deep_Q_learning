import argparse
import cv2
import sys
import time 
import random
import numpy as np

import tensorflow as tf
from collections import deque

import pong_fun as game # Pygame Environment
from capsule_fun import *  # function for capsule network

#####################################################################################################

epsilon = 1e-9
iter_routing = 2
train_freq = 20

GAME = 'pong'
ACTIONS = 6
GAMMA = 0.99
OBSERVE = 1000.
EXPLORE = 5000.
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 1.0
REPLAY_MEMORY = 25000
BATCH = 32
K = 1

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
        print('Checkpoint 1 reached:-Build model')
        s, coeff, readout = createNetwork()
        # define the cost function
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
        x_t, r_0, terminal, bar1_score, bar2_score = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)  
        
        epsilon = INITIAL_EPSILON
        b_IJ1 = np.zeros((1, 1152, 10, 1, 1)).astype(np.float32) # batch_size=1
        b_IJ2 = np.zeros((BATCH, 1152, 10, 1, 1)).astype(np.float32) # batch_size=BATCH
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
                readout_t = readout.eval(feed_dict = {s:s_t.reshape((1,84,84,4)), coeff:b_IJ1}, session=mon_sess)
                #readout_t = readout.eval(feed_dict = {s : [s_t]}, session=mon_sess)[0]
                a_t = np.zeros([ACTIONS])
                action_index = 0
                
                if random.random() <= epsilon or t <= OBSERVE:
                    action_index = random.randrange(ACTIONS)
                    a_t[action_index] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1
                # scale down epsilon
                if epsilon > FINAL_EPSILON and t > OBSERVE:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                
                for i in range(0, K):
                    x_t1_col, r_t, terminal, bar1_score, bar2_score = game_state.frame_step(a_t)
                    if(terminal == 1):
                        episode +=1
                    x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (84, 84)), cv2.COLOR_BGR2GRAY)
                    ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
                    x_t1 = np.reshape(x_t1, (84, 84, 1))
                    s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)
                    # store the transition in D
                    D.append((s_t, a_t, r_t, s_t1, terminal))
                    if len(D) > REPLAY_MEMORY:
                        D.popleft()

                    D.append((s_t, a_t, r_t, s_t1, terminal))
                    if len(D) > REPLAY_MEMORY:
                        D.popleft()
                if t > OBSERVE and t%train_freq==0:
                    minibatch = random.sample(D, BATCH)
                    # get the batch variables
                    s_j_batch = [d[0] for d in minibatch]
                    a_batch = [d[1] for d in minibatch]
                    r_batch = [d[2] for d in minibatch]
                    s_j1_batch = [d[3] for d in minibatch]
                    y_batch = []
                    readout_j1_batch = readout.eval(feed_dict = {s:s_j1_batch, coeff:b_IJ2 }, session=mon_sess)
                    for i in range(0, len(minibatch)):
                        if minibatch[i][4]:
                            y_batch.append(r_batch[i])
                        else:
                            y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))
                    
                    train_op.run(feed_dict = {
                        y : y_batch,
                        a : a_batch,
                        s : s_j_batch,
                        coeff: b_IJ2}, session=mon_sess)
                s_t = s_t1
                t += 1
        
                if r_t!= 0:
                    print ("Timestep", t,"/ Score", bar1_score)
        
                if( (bar1_score - bar2_score) > 13): 
                    print("Game_Ends_in Time:",int(time.time() - tick))
                    break;   
                if( (bar1_score - bar2_score) > 11):
                    print("Game_Between_in Time:",int(time.time() - tick))
                    

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
