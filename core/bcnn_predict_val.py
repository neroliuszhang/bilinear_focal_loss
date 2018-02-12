from __future__ import print_function
import tensorflow as tf
import numpy as np
# from scipy.misc import imread, imresize
import tflearn
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import os
from tflearn.data_utils import shuffle
from PIL import Image
import pickle
from tflearn.data_utils import image_preloader
import h5py
import math
# import logging
import random
import time
import cv2
import csv
from tflearn import data_utils

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
            #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            #images = self.imgs - mean
            images = tf.cast(self.imgs, tf.float32) * (1. / 255) - 0.5
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

        self.conv5_3 = tf.transpose(self.conv5_3, perm=[0, 3, 1, 2])
        ''' Reshape conv5_3 from [batch_size, height, width, number_of_filters] 
                                                                        to [batch_size, number_of_filters, height, width]'''

        self.conv5_3 = tf.reshape(self.conv5_3, [-1, 512, 784])
        ''' Reshape conv5_3 from [batch_size, number_of_filters, height*width]
                                                                        '''

        conv5_3_T = tf.transpose(self.conv5_3, perm=[0, 2, 1])
        ''' A temporary variable which holds the transpose of conv5_3 
            from [batch_size,number_of_filters,height*width] to [batch_size,height*width,numer_of_filters]'''

        self.phi_I = tf.matmul(self.conv5_3, conv5_3_T)
        '''Matrix multiplication [batch_size,512,784] x [batch_size,784,512] '''

        self.phi_I = tf.reshape(self.phi_I, [-1, 512 * 512])
        '''Reshape from [batch_size,512,512] to [batch_size, 512*512] '''
        print('Shape of phi_I after reshape', self.phi_I.get_shape())

        self.phi_I = tf.divide(self.phi_I, 784.0)
        print('Shape of phi_I after division', self.phi_I.get_shape())

        self.y_ssqrt = tf.multiply(tf.sign(self.phi_I), tf.sqrt(tf.abs(self.phi_I) + 1e-12))
        '''Take signed square root of phi_I'''
        print('Shape of y_ssqrt', self.y_ssqrt.get_shape())

        self.z_l2 = tf.nn.l2_normalize(self.y_ssqrt, dim=1)
        '''Apply l2 normalization'''
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
    #train_data = h5py.File('/home/goerlab/Bilinear-CNN-TensorFlow/train_test_small/new_train_488.h5', 'r')
    val_data = h5py.File('/home/goerlab/Bilinear-CNN-TensorFlow/dataset/one_pic/data/new_val_224.h5', 'r')

    #print('Input data read complete')

    #X_train, Y_train = train_data['X'], train_data['Y']
    X_val, Y_val = val_data['X'], val_data['Y']
    #print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)

    '''Shuffle the data'''
    #X_train, Y_train = shuffle(X_train, Y_train)
    X_val, Y_val = shuffle(X_val, Y_val)
    #print("Data shapes -- (train, val, test)", X_train.shape, X_val.shape)

    sess = tf.Session()  ## Start session to create training graph

    imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
    target = tf.placeholder("float", [None, 13])
    target = tf.placeholder("float", [None, 13])

    # print 'Creating graph'
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    print('VGG network created')

    # Defining other ops using Tensorflow
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=vgg.fc3l, labels=target))
    #
    # print([_.name for _ in vgg.parameters])
    #
    # learning_rate = 0.001

    # optimizer =tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

    # check_op = tf.add_check_numerics_ops()
    confidence_score=tf.reduce_max(tf.nn.softmax(vgg.fc3l),axis=1)
    prediction=tf.argmax(vgg.fc3l,1)
    label=tf.argmax(target,1)
    correct_prediction = tf.equal(tf.argmax(vgg.fc3l, 1), tf.argmax(target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    num_correct_preds = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
    checkpoint_file=tf.train.latest_checkpoint("/home/goerlab/Bilinear-CNN-TensorFlow/core/model/20180110/checkpoint/")

    variables_to_restore = slim.get_variables_to_restore()
    #W1=variables_to_restore[0]
    print("variables_to_restore:")
    print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)

    saver.restore(sess, checkpoint_file)



    #val_batch_size = 10#####This is origin
    val_batch_size=1
    total_val_count = len(X_val)##########This is origin
    #total_val_count=1

    correct_val_count = 0
    val_loss = 0.0
    total_val_batch = int(total_val_count / val_batch_size)

    # for i in range(total_val_batch):
    #     print("total val batch:%d" %(i))
    #     batch_val_x, batch_val_y = X_val[i * val_batch_size:i * val_batch_size + val_batch_size], Y_val[
    #                                                                                               i * val_batch_size:i * val_batch_size + val_batch_size]
    #     val_loss += sess.run(loss, feed_dict={imgs: batch_val_x, target: batch_val_y})
    #
    #     pred,real_prediction,real_label,real_confidence_score = sess.run([num_correct_preds,prediction,label,confidence_score], feed_dict={imgs: batch_val_x, target: batch_val_y})
    #     # print(batch_val_x)
    #     print(real_prediction)
    #     # print(real_label)
    #     print(real_confidence_score)
    #     correct_val_count += pred
    #
    # print("##############################")
    # print("Validation Loss -->", val_loss)
    # print("correct_val_count, total_val_count", correct_val_count, total_val_count)
    # print("Validation Data Accuracy -->", 100.0 * correct_val_count / (1.0 * total_val_count))
    # print("##############################")

        #validation_accuracy_buffer = []
    num_of_class=13
    confusion_matrix=[[0 for col in range(num_of_class)] for row in range(num_of_class)]

    csv_dir="/home/goerlab/Bilinear-CNN-TensorFlow/core/model/20180110/record/"
    csvfile=open(csv_dir+"test_detail.csv","wb")
    writer=csv.writer(csvfile)
    writer.writerow(["labels","prediction","confidence_score","file_name"])
    image_dir = "/media/goerlab/My Passport/Welder_detection/dataset/20180109/Data/val/"
    cnt=0
    for dirc in os.listdir(image_dir):
        subdir=image_dir+dirc+'/'
        if not os.path.isfile(subdir):
            for subdirc in os.listdir(subdir):


                file_name=subdir+subdirc
                print(file_name)
                file_image=data_utils.load_image(file_name)
                file_image=data_utils.resize_image(file_image,448,448)
                #file_image=cv2.imread(file_name)
                #file_image=cv2.resize(file_image,(448,448))



                #file_image=file_image*(1./255)-0.5
                trans_image=np.asarray(file_image).reshape((1,448,448,3))
                #trans_image=trans_image(1./255)-0.5

                real_predict,real_fc3l,real_confidence_score=sess.run([prediction,vgg.fc3l,confidence_score],feed_dict={imgs:trans_image})
                #writer.writerow([str(dirc),real_predict[0],real_proba[0],file_name])
                # print(trans_image)
                real_label=int(dirc)
                print("label:%d, prediction:%d, confidence score:%f" %(real_label,real_predict,real_confidence_score))
                #print(confusion_matrix)
                writer.writerow([real_label,real_predict[0],real_confidence_score[0],file_name])
                confusion_matrix[real_label][real_predict]+=1
                cnt+=1
                # print(real_predict)
                # #print(real_fc3l)
                #
                # print(real_confidence_score)
                # print(real_proba)
                # print(real_softmax)

    print("confusion matrix:")
    #print(confusion_matrix)
    for i in range(num_of_class):
        print(confusion_matrix[i])
    precision_result = [0 for i in range(num_of_class)]
    recall_result = [0 for i in range(num_of_class)]
    # for i in range(num_of_class):
    #       precision_result[i]=confusion_matrix[i][i]/
    precision_sum = map(sum, zip(*confusion_matrix))

    # print("precision_sum:")
    # print(precision_sum)
    for i in range(num_of_class):
        if precision_sum[i]==0:
            precision_sum[i]=1
        precision_result[i] = 1.0*confusion_matrix[i][i] / precision_sum[i]

    print("average_precision:")
    print(precision_result)
    print("mean_average_precision:")
    print(sum(precision_result)/num_of_class)

    # print("mean_average_precision:")
    # print(sum(precision_result)/num_of_class)

    #print("recall_sum:")
    recall_sum = map(sum, confusion_matrix)
    #print(recall_sum)

    for i in range(num_of_class):
        if recall_sum[i]==0:
            recall_sum[i]=1
        recall_result[i] = 1.0*confusion_matrix[i][i] / recall_sum[i]

    print("recall:")
    print(recall_result)
    print("mean_recall:")
    print(sum(recall_result)/num_of_class)
    real_accuracy=0
    for i in range(num_of_class):

        real_accuracy+=confusion_matrix[i][i]

    accuracy_final=1.0*real_accuracy/sum(recall_sum)
    print(real_accuracy)
    print(sum(recall_sum))
    print(accuracy_final)
    print("accuracy:%d/%d = %f" %(real_accuracy,sum(recall_sum),accuracy_final))