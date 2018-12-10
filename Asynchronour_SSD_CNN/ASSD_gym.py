import tensorflow as tf
import cv2
import sys
import gym
import random
import time 
import numpy as np
from collections import deque
env = gym.envs.make("Pong-v0")
from matplotlib import pyplot as plt
ACTIONS = 4 #env.action_space.n # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 500. # timesteps to observe before training
EXPLORE = 10000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
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

sess = tf.InteractiveSession()
s, readout, h_fc1 = createNetwork()
# define the cost function
a = tf.placeholder("float", [None, ACTIONS])
y = tf.placeholder("float", [None])
readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
cost = tf.reduce_mean(tf.square(y - readout_action))
train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
# store the previous observations in replay memory
D = deque()
obs = env.reset()
obs, r_0, done, _ = env.step(0)
x_t = obs[34:-16,:,:]
#x_t, r_0, terminal, bar1_score, bar2_score = game_state.frame_step(do_nothing)
x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)
# saving and loading networks
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
epsilon = INITIAL_EPSILON
t = 0
episode = 0
bar1_score =0
bar2_score =0
tick = time.time()
obs = env.reset()
q_values = []
cost_list = []
while True:
    env.render()
    # choose an action epsilon greedily
    readout_t = readout.eval(feed_dict = {s : [s_t]})[0]
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
        # run the selected action and observe next state and reward
        obs, r_t, done, _ = env.step(action_index)# game_state.frame_step(a_t)
        x_t1_col = obs[34:-16,:,:]
        if(r_t==1):
            bar1_score += 1
        if(r_t==-1):
            bar2_score += 1
        if(done ==1):
            obs = env.reset()
            episode +=1
            bar1_score =0
            bar2_score =0
            obs = env.reset()
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_col, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:,:,0:3], axis = 2)
        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, done))
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
        readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
        for i in range(0, len(minibatch)):
            # if terminal only equals reward
            if minibatch[i][4]:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

        # perform gradient step
        train_step.run(feed_dict = {y : y_batch,a : a_batch, s : s_j_batch})
    # update the old values
    s_t = s_t1
    t += 1
    # save progress every 10000 iterations
    if t % 10000 == 0:
        q_values.append(np.max(readout_t))
        loss  = sess.run(cost,feed_dict = {y : y_batch,a : a_batch, s : s_j_batch})
        cost_list.append(loss)
    if r_t!= 0:
        print ("TIMESTEP", t, "/ episode", episode, "/ bar1_score", bar1_score, "/ bar2_score", bar2_score, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
    if(bar1_score - bar2_score > 10): 
        print("Game_Ends_in Time:",int(time.time() - tick))
        break;
print("Game_Ends_in Time:",int(time.time() - tick))
print("____________ END HERE _____________")
