import argparse
import sys
import cv2
import time
import pong_fun as game 
import random
import numpy as np
from collections import deque
import pandas as pd
import tensorflow as tf
from conv_fun import *
FLAGS = None

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  n_replica = len(worker_hosts )
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
	actions = 6 
	gamma = 0.99 
	observe = 500
	EXPLORE = 1000
        s, readout, h_fc1 = createNetwork()
        a = tf.placeholder("float", [None, actions])
        y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
        cost = tf.reduce_mean(tf.square(y - readout_action))
        #train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
        global_step = tf.contrib.framework.get_or_create_global_step() 
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        opt = tf.train.SyncReplicasOptimizer(opt,
                                             replicas_to_aggregate=n_replica,
                                             total_num_replicas=n_replica,
                                             use_locking=True)
        train_op = opt.minimize(cost, global_step= global_step)
        init_op = tf.global_variables_initializer()
        game_state = game.GameState()

        D =  deque()
        do_nothing = np.zeros(actions)
        do_nothing[0] = 1
        x_t, r_0, terminal, bar1_score, bar2_score = game_state.frame_step(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)

        #saver = tf.train.Saver()
        #var_list = tf.contrib.framework.list_variables(config.pre_model_dir)
        #for v in var_list:
            #print(v)
	FINAL_EPSILON = 0.05 
	INITIAL_EPSILON = 1.0 
	replay_memory = 25000 
	batch = 32
	k = 1
        epsilon = INITIAL_EPSILON
        t = 0
        dct = {}
        episode = 0; cost_list = []; q_values = [];
        tick = time.time()
        print('Step 1 complete')

    # The StopAtStepHook handles stopping after running given steps.
    # hooks=[tf.train.StopAtStepHook(last_step=1000000)]
    
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    ''' with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="",
                                           hooks=hooks) as mon_sess:'''
    is_chief=(FLAGS.task_index == 0)
    # You can create the hook which handles initialization and queues.
    sync_replicas_hook = opt.make_session_run_hook(is_chief)
    
    with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief,hooks=[sync_replicas_hook]) as mon_sess:
        print('Step 2 started')
        mon_sess.run(init_op)
        print('Step 2 complete')
        while not mon_sess.should_stop():
            mon_sess.run(init_op)
            print("Variables initialized ...")
            print('Step 2 Done')
            while True:
                #print('Checkpoint 3 reached')
                readout_t = readout.eval(feed_dict = {s : [s_t]}, session=mon_sess)[0]
                a_t = np.zeros([actions])
                action_index = 0

                if random.random() <= epsilon or t <= observe:
                        action_index = random.randrange(actions)
                        a_t[action_index] = 1
                else:
                        action_index = np.argmax(readout_t)
                        a_t[action_index] = 1

                if epsilon > FINAL_EPSILON and t > observe:
                        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                for i in range(0, k):
                        x_t1_col, r_t, terminal, bar1_score, bar2_score  = game_state.frame_step(a_t)
                        if(terminal == 1):
                            episode +=1
                        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
                        ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
                        x_t1 = np.reshape(x_t1, (80, 80, 1))
                        s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)

                        D.append((s_t, a_t, r_t, s_t1, terminal))
                        if len(D) > replay_memory:
                            D.popleft()
                if t > observe:
                    minibatch = random.sample(D, batch)

                    s_j_batch = [d[0] for d in minibatch]
                    a_batch = [d[1] for d in minibatch]
                    r_batch = [d[2] for d in minibatch]
                    s_j1_batch = [d[3] for d in minibatch]

                    y_batch = []
                    readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch}, session=mon_sess)
                    for i in range(0, len(minibatch)):
                        if minibatch[i][4]:
                            y_batch.append(r_batch[i])
                        else:
                            y_batch.append(r_batch[i] + gamma * np.max(readout_j1_batch[i]))
                    mon_sess.run(train_op,feed_dict = {
                        y : y_batch,
                        a : a_batch,
                        s : s_j_batch})
                    if(t%10000 == 0):
                        q_values.append(np.max(readout_t))
                        loss =  mon_sess.run(cost,feed_dict = {y : y_batch,a : a_batch,s : s_j_batch})
                        cost_list.append(loss)
                    if(t%20000==0):
                        dct["q_values"] = q_values
                        dct["cost"] = cost_list
                        dct["time_step"] = t
                        dct["bar1_score"] = bar1_score
                        dct["bar2_score"] = bar2_score
                        result  = pd.DataFrame(dct)
                        result.to_csv("/home/dolaram/Syn_CNN/output.csv")
    
                        
                s_t = s_t1
                t += 1
                    
                if r_t!= 0:
                        print ("TIMESTEP", t, "/episode",episode, "/bar1_score", bar1_score, "/bar2_score", bar2_score, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
                if((bar1_score - bar2_score) > 15 ):
                    print("GAME_ENDS_in_Time",int(time.time() -tick))
			
                    break;
                if((bar1_score - bar2_score) > 14 ):
                    print("GAME_MID_in_Time",int(time.time() -tick))



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

