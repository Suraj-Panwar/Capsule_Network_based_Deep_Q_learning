import tensorflow as tf
import numpy as np
epsilon = 1e-9
iter_routing = 2
ACTIONS = 6 
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
    readout = q_eval
    return s, coeff, readout

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
    
def reduce_sum(input_tensor, axis=None, keepdims=False):
    return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
   
def softmax(logits, axis=None):
    return tf.nn.softmax(logits, axis=axis)
