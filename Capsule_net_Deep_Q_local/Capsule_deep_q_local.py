#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import cv2
import sys
sys.path.append("Wrapped Game Code/")
import pong_fun as game # whichever is imported "as game" will be used
import dummy_game
#import tetris_fun as game
import random
import time 
import numpy as np
from collections import deque


# In[2]:


epsilon = 1e-9
iter_routing = 3
train_freq = 20


# In[3]:


def squash(vector):
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
def routing(input, b_IJ):
    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    W = tf.get_variable('Weight', shape=(1, 1152, 160, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))
    biases = tf.get_variable('bias', shape=(1, 1, 10, 16, 1))
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, 160, 1, 1])
    #assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]

    u_hat = reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, 1152, 10, 16, 1])
    #assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                #assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                #assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                #assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v
    return(v_J)
# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)
# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)


# In[4]:


GAME = 'pong' # the name of the game being played for log files
ACTIONS = 6 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 5000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 1 # only select an action every Kth frame, repeat prev for others


# In[5]:


def createNetwork():
    # input layer
    s= tf.placeholder("float", [None, 84, 84, 4])
    coeff = tf.placeholder(tf.float32, shape=(None, 1152, 10, 1, 1))
    ####################### New Network COnfiguration #####################    
    w_initializer, b_initializer = tf.random_normal_initializer(0., 0.01), tf.constant_initializer(0.01)
    w1 = tf.get_variable('w1',[8, 8, 4, 256],initializer=w_initializer)
    b1 = tf.get_variable('b1',[256],initializer=b_initializer)
    # Convolution Layer
    # Conv1, [batch_size, 20, 20, 256]
    l1 = tf.nn.conv2d(s, w1, strides=[1, 4, 4, 1], padding="VALID")
    conv1 = tf.nn.relu(tf.nn.bias_add(l1, b1))
    conv1 = tf.reshape(conv1,[-1,20,20,256])
    capsules = tf.contrib.layers.conv2d(conv1, 32 * 8,kernel_size=9, stride=2, padding="VALID",
                    activation_fn = tf.nn.relu,
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform=False),
                    biases_initializer=tf.constant_initializer(0))
    
    capsules = tf.reshape(capsules, (-1, 1152, 8, 1)) #Reshape to(batch_szie, 1152, 8, 1)
    capsules = squash(capsules)
    input_fc = tf.reshape(capsules, shape=(-1, 1152, 1, capsules.shape[-2].value, 1))
    caps2 = routing(input_fc, coeff)
    vector_j = tf.reshape(caps2, shape=(-1, 160))
    fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=30, activation_fn=tf.nn.relu)
    q_eval = tf.contrib.layers.fully_connected(fc1, num_outputs=ACTIONS, activation_fn=None)
    #output = tf.nn.softmax(logits = fc2)
    #argmax_idx = tf.to_int32(tf.argmax(output, axis=1))
    readout = q_eval
    return s, coeff, readout


# In[6]:


def trainNetwork(s, coeff, readout, sess):
    tick = time.time()
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()
    # store the previous observations in replay memory
    D = deque()
    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, bar1_score, bar2_score = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis = 2)  
    # saving and loading networks
    saver = tf.train.Saver()
    #sess.run(tf.initialize_all_variables())
    sess.run(tf.global_variables_initializer())
    b_IJ1 = np.zeros((1, 1152, 10, 1, 1)).astype(np.float32) # batch_size=1
    b_IJ2 = np.zeros((BATCH, 1152, 10, 1, 1)).astype(np.float32) # batch_size=BATCH
    epsilon = INITIAL_EPSILON
    t = 0
    episode = 0
    while "pigs" != "fly":
        # choose an action epsilon greedily
        # readout_t = readout.eval(feed_dict = {s : [s_t].reshape((1,80,80,4))})[0]
        
        readout_t = readout.eval(feed_dict = {s:s_t.reshape((1,84,84,4)), coeff:b_IJ1})
        
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
        
        # only train if done observing
        if t > OBSERVE and t%train_freq==0:
            # sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = readout.eval(feed_dict = {s:s_j1_batch, coeff:b_IJ2 })
            #readout_j1_batch = readout.eval(feed_dict = {s : s_j1_batch})
            for i in range(0, len(minibatch)):
                # if terminal only equals reward
                if minibatch[i][4]:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step
            train_step.run(feed_dict = {
                y : y_batch,
                a : a_batch,
                s : s_j_batch,
                coeff: b_IJ2})

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"
        #print ("TIMESTEP", t, "/ STATE", state, "/ LINES", game_state.total_lines, "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))
        if r_t!= 0:
            print ("TIMESTEP", t, "/ EPISODE", episode, "/ bar1_score", bar1_score, "/ bar2_score", bar2_score, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))

        if(bar1_score - bar2_score > 17): 
            print("Game_Ends_in Time:",int(time.time() - tick))
            break;   
        if(bar1_score - bar2_score > 15):
            print("Game_Mid_in Time:",int(time.time() - tick))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''


# In[7]:


# tf.reset_default_graph()
# sess = tf.InteractiveSession()
# s_, coeff, readout = createNetwork()
# print(s_)
# trainNetwork(s_, coeff, readout, sess)


# In[8]:


def playGame():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    s, coeff, readout = createNetwork()
    trainNetwork(s, coeff, readout, sess)


# In[9]:


def main():
    playGame()

if __name__ == "__main__":
    tick = time.time()
    main()
    print("Game_Ends_in Time:",int(time.time() - tick))
    print("____________ END HERE _____________")


# In[ ]:





# In[ ]:





# In[ ]:



