from __future__ import print_function
import tensorflow as tf
import numpy as np
# from scipy.misc import imread, imresize
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
from tflearn.data_utils import shuffle

import pickle
from tflearn.data_utils import image_preloader
import h5py
import math
# import logging
import random
import time

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def random_flip_right_to_left(image_batch):
    '''
    This function will flip the images randomly.
    Input: batch of images [batch, height, width, channels]
    Output: batch of images flipped randomly [batch, height, width, channels]
    '''
    result = []
    for n in range(image_batch.shape[0]):
        if bool(random.getrandbits(1)):  ## With 0.5 probability flip the image
            result.append(image_batch[n][:, ::-1, :])
        else:
            result.append(image_batch[n])
    return result


class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.last_layer_parameters = []  ## Parameters in this list will be optimized when only last layer is being trained
        self.parameters = []  ## Parameters in this list will be optimized when whole BCNN network is finetuned
        self.convlayers()  ## Create Convolutional layers
        self.fc_layers()  ## Create Fully connected layer
        self.weight_file = weights

    def convlayers(self):
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean
            print('Adding Data Augmentation')

        # conv1_1
        with tf.variable_scope("conv1_1"):
            weights = tf.get_variable("W", [3, 3, 3, 64], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv1_2
        with tf.variable_scope("conv1_2"):
            weights = tf.get_variable("W", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [64], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv1_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool1')

        # conv2_1
        with tf.variable_scope("conv2_1"):
            weights = tf.get_variable("W", [3, 3, 64, 128], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv2_2
        with tf.variable_scope("conv2_2"):
            weights = tf.get_variable("W", [3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [128], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv2_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool2')

        # conv3_1
        with tf.variable_scope("conv3_1"):
            weights = tf.get_variable("W", [3, 3, 128, 256], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv3_2
        with tf.variable_scope("conv3_2"):
            weights = tf.get_variable("W", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv3_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv3_3
        with tf.variable_scope("conv3_3"):
            weights = tf.get_variable("W", [3, 3, 256, 256], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [256], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv3_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv3_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool3')

        # conv4_1
        with tf.variable_scope("conv4_1"):
            weights = tf.get_variable("W", [3, 3, 256, 512], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool3, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv4_2
        with tf.variable_scope("conv4_2"):
            weights = tf.get_variable("W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv4_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv4_3
        with tf.variable_scope("conv4_3"):
            weights = tf.get_variable("W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv4_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv4_3 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

        # conv5_1
        with tf.variable_scope("conv5_1"):
            weights = tf.get_variable("W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.pool4, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_1 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv5_2
        with tf.variable_scope("conv5_2"):
            weights = tf.get_variable("W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv5_1, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_2 = tf.nn.relu(conv + biases)
            self.parameters += [weights, biases]

        # conv5_3
        with tf.variable_scope("conv5_3"):
            weights = tf.get_variable("W", [3, 3, 512, 512], initializer=tf.contrib.layers.xavier_initializer(),
                                      trainable=True)
            # Create variable named "biases".
            biases = tf.get_variable("b", [512], initializer=tf.constant_initializer(0.1), trainable=True)
            conv = tf.nn.conv2d(self.conv5_2, weights, strides=[1, 1, 1, 1], padding='SAME')
            self.conv5_3 = tf.nn.relu(conv + biases)

            self.parameters += [weights, biases]
            #self.special_parameters = [weights, biases]

        self.pool5 = tf.nn.max_pool(self.conv5_3,
                                    ksize=[1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                               trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                               trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_initial_weights(self, session):
        '''weight_dict contains weigths of VGG16 layers'''
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            #print
            print("i:%d,k:%d" %(i,k))
            print(np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))


if __name__ == '__main__':

    '''
    Load Training and Validation Data
    '''
    train_data = h5py.File('../train_test/new_train_448.h5', 'r')
    val_data = h5py.File('../train_test/new_val_448.h5', 'r')

    print('Input data read complete')

    X_train, Y_train = train_data['X'], train_data['Y']
    X_val, Y_val = val_data['X'], val_data['Y']
    print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)

    '''Shuffle the data'''
    X_train, Y_train = shuffle(X_train, Y_train)
    X_val, Y_val = shuffle(X_val, Y_val)
    print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)

    sess = tf.Session()  ## Start session to create training graph

    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 100])

    # print 'Creating graph'
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    print('VGG network created')

    # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))

    print([_.name for _ in vgg.parameters])

    learning_rate = 0.001

    # optimizer =tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

    # check_op = tf.add_check_numerics_ops()


    correct_prediction = tf.equal(tf.argmax(vgg.fc3l, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    vgg.load_initial_weights(sess)
    print([_.name for _ in vgg.parameters])

    batch_size = 2

    for v in tf.trainable_variables():
        print("Trainable variables", v)
    print('Starting training')

    lr = 0.001
    finetune_step = -1

    #######queenie edited##############################

    val_batch_size = 10
    total_val_count = len(X_val)
    correct_val_count = 0
    val_loss = 0.0
    total_val_batch = int(total_val_count / val_batch_size)

    #######queenie edited#############################################


    for i in range(total_val_batch):
        batch_val_x, batch_val_y = X_val[i * val_batch_size:i * val_batch_size + val_batch_size], Y_val[
                                                                                                  i * val_batch_size:i * val_batch_size + val_batch_size]
        val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y})

        pred = sess.run(num_correct_preds, feed_dict={imgs: batch_val_x, target: batch_val_y})
        correct_val_count += pred

    print("##############################")
    print("Validation Loss -->", val_loss)
    print("correct_val_count, total_val_count", correct_val_count, total_val_count)
    print("Validation Data Accuracy -->", 100.0 * correct_val_count / (1.0 * total_val_count))
    print("##############################")

    validation_accuracy_buffer = []

    for epoch in range(2):  ####queenie edited, origin: 100########################
        avg_cost = 0.
        total_batch = int(333 / batch_size)
        X_train, Y_train = shuffle(X_train, Y_train)

        for i in range(2):
            batch_xs, batch_ys = X_train[i * batch_size:i * batch_size + batch_size], Y_train[
                                                                                      i * batch_size:i * batch_size + batch_size]
            batch_xs = random_flip_right_to_left(batch_xs)

            start = time.time()
            # sess.run([optimizer,check_op], feed_dict={imgs: batch_xs, target: batch_ys})
            sess.run([optimizer], feed_dict={imgs: batch_xs, target: batch_ys})
            if i % 20 == 0:
                print('Full BCNN finetuning, time to run optimizer for batch size 16:', time.time() - start, 'seconds')

            cost = sess.run(loss, feed_dict={imgs: batch_xs, target: batch_ys})

            if i % 20 == 0:
                print('Learning rate: ', (str(lr)))
                if epoch <= finetune_step:
                    print("Training last layer of BCNN_DD")
                else:
                    print("Fine tuning all BCNN_DD")

                print("Epoch:", '%03d' % (epoch + 1), "Step:", '%03d' % i, "Loss:", str(cost))
                print("Training Accuracy -->",
                      accuracy.eval(feed_dict={imgs: batch_xs, target: batch_ys}, session=sess))
                # print(sess.run(vgg.fc3l, feed_dict={imgs: batch_xs, target: batch_ys}))

        val_batch_size = 10
        total_val_count = len(X_val)
        correct_val_count = 0
        val_loss = 0.0
        total_val_batch = int(total_val_count / val_batch_size)
        for i in range(total_val_batch):
            batch_val_x, batch_val_y = X_val[i * val_batch_size:i * val_batch_size + val_batch_size], Y_val[
                                                                                                      i * val_batch_size:i * val_batch_size + val_batch_size]
            val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y})

            pred = sess.run(num_correct_preds, feed_dict={imgs: batch_val_x, target: batch_val_y})
            correct_val_count += pred

        print("##############################")
        print("Validation Loss -->", val_loss)
        print("correct_val_count, total_val_count", correct_val_count, total_val_count)
        print("Validation Data Accuracy -->", 100.0 * correct_val_count / (1.0 * total_val_count))
        print("##############################")

        if epoch > 40:

            validation_accuracy_buffer.append(100.0 * correct_val_count / (1.0 * total_val_count))
            ## Check if the validation accuracy has stopped increasing
            if len(validation_accuracy_buffer) > 10:
                index_of_max_val_acc = np.argmax(validation_accuracy_buffer)
                if index_of_max_val_acc == 0:
                    break
                else:
                    del validation_accuracy_buffer[0]
