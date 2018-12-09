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
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
def routing(input, b_IJ):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    W = tf.get_variable('Weight', shape=(1, 1152, 160, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.01))
    biases = tf.get_variable('bias', shape=(1, 1, 10, 16, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
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

    # printing
#     a_file = open("logs_" + GAME + "/readout.txt", 'w')
#     h_file = open("logs_" + GAME + "/hidden.txt", 'w')

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
    ####################################################################################
    """do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, bar1_score, bar2_score = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t_1 = np.stack((x_t, x_t, x_t, x_t), axis = 2)
    
    b_IJ1 = np.zeros((1, 1152, 10, 1, 1)).astype(np.float32) # batch_size=1
    b_IJ2 = np.zeros((BATCH, 1152, 10, 1, 1)).astype(np.float32) # batch_size=BATCH
    
    rr = readout.eval(feed_dict = {s_ :s_t_1.reshape((1,84,84,4)), s:st_i, coeff:b_IJ1})
    #readout = readout.eval(feed_dict = {s_ :s_t_1.reshape((1,84,84,4)), s:st_i, coeff:b_IJ1})
    print("Shape of first network",rr.shape)"""    
    ####################################################################################
    #print("Shape of first network",conv1_t)
#     checkpoint = tf.train.get_checkpoint_state("saved_networks")
#     if checkpoint and checkpoint.model_checkpoint_path:
#         saver.restore(sess, checkpoint.model_checkpoint_path)
#         print( "Successfully loaded:", checkpoint.model_checkpoint_path)
#     else:
#         print ("Could not find old network weights")
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


# In[1]:


225402/3600


# In[2]:


225402-3600*62


# In[3]:


2202/60


# In[10]:


518751/20


# In[1]:


202006-3600*56


# In[2]:


406/60


# In[1]:


202006/3600


# In[16]:


q_values = [1.530984e-01,1.564839e-01,6.165820e-01,1.002289e+00,1.091462e+00,1.097284e+00,1.161262e+00, 1.198849e+00,1.224219e+00,1.250956e+00, 1.281951e+00,1.345693e+00, 1.361393e+00,1.490082e+00,1.503976e+00,1.504488e+00, 1.641450e+00,1.640933e+00,1.705964e+00,1.730145e+00,1.752013e+00,1.748811e+00, 1.782931e+00,1.792710e+00, 1.818257e+00,1.822933e+00, 1.804847e+00,1.8227,1.813930e+00,1.860576e+00, 1.880579e+00, 1.869156e+00, 1.919491e+00,1.913734e+00,1.877738e+00,1.942900e+00,1.900447e+00,1.914032e+00]#,1.923226e+00,2.207653e+00,2.197628e+00,1.875603e+00, 1.869130e+00,1.959915e+00,2.372653e+00, 2.320155e+00, 2.460180e+00,2.161158e+00,2.220545e+00, 2.195042e+00,1.982264e+00]


# In[1]:


from matplotlib import pyplot as plt


# In[18]:


plt.plot(q_values)


# In[1]:


q_values1 = [0.0123329964,0.0103768017,0.0109969731,0.0118582118,0.0110366894,0.0120392106,0.0081759347,0.0080730841,0.005814835,0.0034184759,-0.0006518066,-0.0015277872,-0.0057894643,-0.0026164083,-0.0036992161,
-0.016304791,-0.0180712789,-0.0303907339,-0.0821117982,-0.1101859659,-0.1182276607,-0.0969664678,-0.123342298,-0.2122781426,-0.1713319123,-0.0026164083,-0.0036992161,-0.016304791,
-0.0180712789,0.0080712789,0.0180712789,0.0180712789,6.165820e-01,7.165820e-01,8.165820e-01,1.464839e-01,6.165820e-01,1.002289e+00,1.091462e+00,1.097284e+00,1.161262e+00, 1.198849e+00,1.224219e+00,1.250956e+00, 1.281951e+00,1.490082e+00,1.503976e+00,1.504488e+00, 1.641450e+00,1.640933e+00,1.705964e+00,1.730145e+00,1.752013e+00,1.748811e+00, 1.804847e+00,1.8227,1.813930e+00,1.860576e+00, 1.880579e+00, 1.913734e+00,1.877738e+00]


# In[4]:


plt.plot(q_values1)


# In[2]:


score = [-17,-18,-18,-16,-17,-15,-15,-15,-15,-15,-15,-14,-13,-12,-10,-5,-6,-7,-6,-10,-5,-4,-2,0,3,-1,5,1,10,7,8,13,15,14,15,17,16,15,15,16,17,17,17,18,19]


# In[3]:


plt.plot(score)


# In[29]:


cost_capsule = [0.02701401,0.028786866,0.036477614,0.0060465923,0.043763548,0.010776486,0.008192269,0.010940961,0.030792711
                ,0.036387745,0.01559204,0.02088072,0.023226066,0.020197935,0.049995854,0.0444438,0.05156593,0.044140372
                ,0.03947793,0.024624988,0.023623686,0.020794272,0.0148373535,0.0509583,0.03287832,0.009071955,0.028208429
                ,0.0072530126,0.053203996,0.0062716636,0.007196473,0.029647475,0.0076366756,0.029146846,0.0048923437,0.0027569286
                ,0.005318172,0.0050067995,0.0045762174,0.15842143
                ,0.05486014,0.028587535,0.05722414,0.005370695,0.14684866,0.028189054
                ,0.0050085858,0.006993169,0.029938692,0.033542663
                ,0.0020201309,0.034135193,0.0019902664,0.0050085858,0.006993169
                ,0.007196473,0.0050085858,0.0050085858,0.0050085858
               ,0.005318172,0.0050067995,0.0045762174,0.005318172,0.0050067995,0.0045762174,0.005318172,0.0050067995,0.0045762174
               ,0.0020201309,0.034135193,0.0019902664,0.0050085858,0.006993169]


# In[30]:


plt.plot(cost_capsule)


# In[7]:


aAA


# In[15]:


a


# In[31]:


import pandas as pd
df = pd.read_csv("output.csv")


# In[33]:


plt.plot(df["cost"])


# In[34]:


df["cost"]


# In[ ]:




