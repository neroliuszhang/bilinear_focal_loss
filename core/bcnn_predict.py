from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import csv

import h5py
import math
import random
import pickle
import os

slim=tf.contrib.slim
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
            # mean = tf.constant([110.61220668946963, 127.15494646868024, 135.13440353131017], dtype=tf.float32,
            #                    shape=[1, 1, 1, 3], name='img_mean')
            #images = self.imgs - mean
            images=tf.cast(self.imgs,tf.float32)*(1./255)-0.5
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
            self.special_parameters = [weights, biases]

        ''' Reshape conv5_3 from [batch_size, height, width, number_of_filters]
           to [batch_size, number_of_filters, height, width]'''
        self.conv5_3 = tf.transpose(self.conv5_3, perm=[0, 3, 1, 2])

        ''' Reshape conv5_3 from [batch_size, number_of_filters, height*width] '''
        self.conv5_3 = tf.reshape(self.conv5_3, [-1, 512, 784])

        ''' A temporary variable which holds the transpose of conv5_3 '''
        conv5_3_T = tf.transpose(self.conv5_3, perm=[0, 2, 1])

        '''Matrix multiplication [batch_size,512,784] x [batch_size,784,512] '''
        self.phi_I = tf.matmul(self.conv5_3, conv5_3_T)

        '''Reshape from [batch_size,512,512] to [batch_size, 512*512] '''
        self.phi_I = tf.reshape(self.phi_I, [-1, 512 * 512])
        print('Shape of phi_I after reshape', self.phi_I.get_shape())

        self.phi_I = tf.divide(self.phi_I, 784.0)
        print('Shape of phi_I after division', self.phi_I.get_shape())

        '''Take signed square root of phi_I'''
        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I), tf.sqrt(tf.abs(self.phi_I) + 1e-12))
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        '''Apply l2 normalization'''
        self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, dim=1)
        print('Shape of z_l2', self.z_l2.get_shape())

    def fc_layers(self):
        with tf.variable_scope('fc-new') as scope:
            fc3w = tf.get_variable('W', [512 * 512, 13], initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=True)
            # fc3b = tf.Variable(tf.constant(1.0, shape=[100], dtype=tf.float32), name='biases', trainable=True)
            fc3b = tf.get_variable("b", [13], initializer=tf.constant_initializer(0.1), trainable=True)
            self.fc3l = tf.nn.bias_add(tf.matmul(self.z_l2, fc3w), fc3b)
            self.last_layer_parameters += [fc3w, fc3b]




if __name__ == '__main__':

    '''
    Load Training and Validation Data
    '''




    sess = tf.Session()  ## Start session to create training graph

    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 13])

    # print 'Creating graph'
    checkpoint_file=tf.train.latest_checkpoint("/home/goerlab/Bilinear-CNN-TensorFlow/core/model/20180110/checkpoint/")
    print("checkpoint file:%s" %(checkpoint_file))
    vgg = vgg16(imgs, 'all_layers2_epoch_0.npz', sess)

    print('VGG network created')



    print([_.name for _ in vgg.parameters])

    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

    # check_op = tf.add_check_numerics_ops()
    softmax = tf.nn.softmax(vgg.fc3l)
    correct_prediction = tf.argmax(vgg.fc3l, 1)

    predict_probability=tf.reduce_max(tf.nn.softmax(vgg.fc3l),axis=1)
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

    # sess.run(tf.global_variables_initializer())

    #sess=tf.Session()
    #sess.run(tf.global_variables_initializer())

    #vgg.load_initial_weights(sess)
    variables_to_restore=slim.get_variables_to_restore()
    print("variables_to_restore:")
    print(variables_to_restore)
    saver=tf.train.Saver(variables_to_restore)

    saver.restore(sess,checkpoint_file)
    image_dir="/media/goerlab/My Passport/Welder_detection/dataset/20180109/Data/val/"
    csv_file = open("./test_detail.csv","wb")
    writer=csv.writer(csv_file)
    writer.writerow(["labels","prediction","probability","filename"])
    for dirc in os.listdir(image_dir):
        subdir=image_dir+dirc+'/'
        for subdirc in os.listdir(subdir):


            file_name=subdir+subdirc
            print(file_name)
            file_image=cv2.imread(file_name)
            file_image=cv2.resize(file_image,(448,448))




            trans_image=np.asarray(file_image).reshape((1,448,448,3))
            real_predict,real_proba,real_softmax=sess.run([correct_prediction,predict_probability,softmax],feed_dict={imgs:trans_image})
            writer.writerow([str(dirc),real_predict[0],real_proba[0],file_name])
            print(real_predict)
            print(real_proba)
            print(real_softmax)
            # print(real_softmax)
#
