import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import math


def leaky_relu(x, leak=0.2, name='relu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def res_block(name, inputs, n_filters, n_layers=2, residual_alpha=1.0, leaky_relu_alpha=0.2, strides=1,
              pad='same', channel='channels_last'):
    with tf.variable_scope(name):
        next_input = inputs
        for i in range(n_layers):
            with tf.variable_scope('conv' + str(i)) as scope:
                normed = layers.layer_norm(next_input)
                nonlinear = leaky_relu(normed, leak=leaky_relu_alpha)
                conv = tf.layers.conv1d(
                    inputs=nonlinear,
                    filters=n_filters,
                    kernel_size=5,
                    strides=strides,
                    padding=pad,
                    activation=None,
                    kernel_initializer=layers.xavier_initializer(),
                    bias_initializer=tf.zeros_initializer(), data_format=channel
                )
                next_input = conv

        return next_input * residual_alpha + inputs


def conv1d(name, input, n_filters):
    with tf.variable_scope(name):
        conv_input = tf.layers.conv1d(
            inputs=input,
            filters=n_filters,
            kernel_size=5,
            strides=1,
            padding='same',
            activation=leaky_relu,
            kernel_initializer=layers.xavier_initializer(),
            bias_initializer=tf.zeros_initializer(),
            data_format='channels_first')

    return conv_input

def conv1d_(input_,output_channels,dilation =1,filter_width = 1,casual = False,name =""):
    with tf.variable_scope(name):
        w = tf.get_variable("w", [1, filter_width,input_.get_shape()[-1],output_channels],
                                 initializer=tf.random_normal_initializer(stddev=0.02),
                                 dtype=tf.float32)
        bias = tf.get_variable("b", [output_channels],
                               initializer=tf.constant_initializer(0.0))
        if casual:
            padding = [[0,0],[(filter_width)-1*dilation,0],[0,0]]
            padded = tf.pad(input_,padding)
            input_expand = tf.expand_dims(padded,dim=1)
            out = tf.nn.atrous_conv2d(input_expand,w,rate=dilation,padding='VALID')+bias
        else:
            input_expand = tf.expand_dims(input_, dim=1)
            out = tf.nn.atrous_conv2d(input_expand, w, rate=dilation, padding='SAME') + bias

        return tf.squeeze(out,[1])

def dil_residual_blk(input_,dilation,layer_no,channel_,filter_w,casual = True,train=True,residual_alpha=1.0, leaky_relu_alpha=0.2):
    next_input = input_
    name_ = "layer_{}".format(layer_no,dilation)
    with tf.variable_scope(name_) as scope:
        normed = layers.layer_norm(next_input)
        nonlinear = leaky_relu(normed, leak=leaky_relu_alpha)
        conv1_ = conv1d_(nonlinear,channel_,name='conv1')

        conv1 = layers.layer_norm(conv1_)
        nonlinear2 = leaky_relu(conv1, leak=leaky_relu_alpha)
        dil_conv = conv1d_(nonlinear2, channel_,dilation,filter_w, name='dilated_conv')

        dil_conv = layers.layer_norm(dil_conv)
        nonlinear3 = leaky_relu(dil_conv,leak=leaky_relu_alpha)

        conv2_ = conv1d_(nonlinear3,2*channel_,name='conv2')

        return input_ + conv2_

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("w", [shape[1], output_size],
                                 initializer=tf.random_normal_initializer(stddev=stddev),
                                 dtype=tf.float32)
        bias = tf.get_variable("b", [output_size],
                               initializer=tf.constant_initializer(bias_start))
    return tf.matmul(input_, matrix) + bias

# MMD
def compute_kernel( x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]

    dim = tf.shape(x)[1]

    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))

    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))


def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)

    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

def KL_Loss(z_mean,z_log):
    kl = -0.5*tf.reduce_sum(1+z_log-tf.square(z_mean)-tf.exp(z_log),1)

    return tf.reduce_mean(kl)


def MSE(real_, fake_):
    d = (real_ - fake_)
    d2 = tf.multiply(d, d) * 2.0
    loss = tf.reduce_sum(d2, 1)
    loss = tf.reduce_mean(loss)

    return loss

def cycle_consistency_loss(real_,fake_):
    return tf.reduce_mean(tf.abs(real_-fake_))

def distance(x1,y1,x2,y2):
    dist = math.sqrt( ((x2-x1)**2)+ ((y2-y1)**2))
    return dist

def get_feature(real_,mean_):
    BASKET_RIGHT = np.array([88, 25] * 100)
    BASKET_RIGHT = np.reshape(BASKET_RIGHT, newshape=[100, 2])

    en_ball = []
    b_feat = []
    tmp_ball = -1

    data = real_

    data[:, :, [0, 2, 4, 6, 8, 10]] = (data[:, :, [0, 2, 4, 6, 8, 10]] * mean_[1]) + mean_[0]
    data[:, :, [1, 3, 5, 7, 9, 11]] = (data[:, :, [1, 3, 5, 7, 9, 11]] * mean_[3]) + mean_[2]

    for x in range(len(data)):
        for i in range(100):
            tmp = []
            ballx = data[x, i, 0]
            bally = data[x, i, 1]

            p1x = data[x, i, 2]
            p1y = data[x, i, 3]
            p1d = distance(ballx, bally, p1x, p1y)

            p2x = data[x, i, 4]
            p2y = data[x, i, 5]
            p2d = distance(ballx, bally, p2x, p2y)

            p3x = data[x, i, 6]
            p3y = data[x, i, 7]
            p3d = distance(ballx, bally, p3x, p3y)

            p4x = data[x, i, 8]
            p4y = data[x, i, 9]
            p4d = distance(ballx, bally, p4x, p4y)

            p5x = data[x, i, 10]
            p5y = data[x, i, 11]
            p5d = distance(ballx, bally, p5x, p5y)

            basketd = distance(ballx, bally, BASKET_RIGHT[i, 0], BASKET_RIGHT[i, 1])

            tmp.append(p1d)
            tmp.append(p2d)
            tmp.append(p3d)
            tmp.append(p4d)
            tmp.append(p5d)
            tmp.append(basketd)

            p = tmp.index(min(tmp))
            has_ball = p
            if p == 5:
                en_ball.append([0, 0, 0, 0, 0])
            else:
                if tmp[p] < 3:
                    if p == 0:
                        en_ball.append([1, 0, 0, 0, 0])
                    elif p == 1:
                        en_ball.append([0, 1, 0, 0, 0])
                    elif p == 2:
                        en_ball.append([0, 0, 1, 0, 0])
                    elif p == 3:
                        en_ball.append([0, 0, 0, 1, 0])
                    elif p == 4:
                        en_ball.append([0, 0, 0, 0, 1])
                    elif p == 5:
                        en_ball.append([0, 0, 0, 0, 0])
                    else:
                        en_ball.append([0, 0, 0, 0, 0])

                elif has_ball is not tmp_ball:
                    en_ball.append([0, 0, 0, 0, 0])
                    pass
                else:
                    en_ball.append([0, 0, 0, 0, 0])

        b_feat.append(en_ball)
        en_ball = []

    b_feat = np.array(b_feat)

    return b_feat