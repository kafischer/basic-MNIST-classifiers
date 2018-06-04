# Implement a 4-layer CNN using tensorflow
#
# Author: Kevin Fischer
# Date: 4/18/2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None


def conv_2d(x, W, b):
    """
    Implements 2d convolutional layer, including bias and ReLu
    nonlinearity.

    Parameters
    ----------
    x : tensor
        Input tensor of shape (samples, height, height, channels)
    W : tensor
        Input tensor of shape (window, window, channels, channels_new)
    b : tensor
        Input tensor of shape (channels_new,)

    Returns
    -------
    Output tensor of shape (samples, height, height, channels_new)
    """
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def max_pool_2x2(x):
    """
    Implements 2x2 max pooling.

    Parameters
    ----------
    x : tensor
        Input tensor of shape (samples, height, height, channels),
        height is necessarily is an even integer

    Returns
    -------
    Output tensor of shape (samples, height/2, height/2, channels)
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def init_weight(shape):
    """
    Generates random tensor for a given shape and standard deviation 0.1.

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    Output tensor of shape
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def init_bias(shape):
    """
    Generates bias tensor for a given shape with values 0.1.

    Parameters
    ----------
    shape : tuple

    Returns
    -------
    bias tensor of shape
    """
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv_net(x):
    """
    Generates computation graph for the 4-layer convolutional network

    Parameters
    ----------
    x : tensor
        Input tensor

    Returns
    -------
    Computation graph
    """

    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolutional layer #1
    # Size [-1, 28, 28, 1] --> [-1, 28, 28, 32]
    W_c1 = init_weight([10, 10, 1, 32])
    b_c1 = init_bias([32])
    x = conv_2d(x, W_c1, b_c1)
    # Size [-1, 28, 28, 1] --> [-1, 14, 14, 32]
    x = max_pool_2x2(x)

    # Convolutional layer #2
    # Size [-1, 14, 14, 32] --> [-1, 14, 14, 16]
    W_c2 = init_weight([5, 5, 32, 16])
    b_c2 = init_bias([16])
    x = conv_2d(x, W_c2, b_c2)
    # Size [-1, 14, 14, 32] --> [-1, 7, 7, 16]
    x = max_pool_2x2(x)

    # Fully connected layer #1
    n_fc1 = 7 * 7 * 16
    x = tf.reshape(x, [-1, n_fc1])
    W_fc1 = init_weight([n_fc1, 1024])
    b_fc1 = init_bias([1024])
    x = tf.nn.relu(tf.add(tf.matmul(x, W_fc1), b_fc1))
    # Fully connected layer #2 (final layer)
    W_fc2 = init_weight([1024, 10])
    b_fc2 = init_bias([10])
    y = tf.add(tf.matmul(x, W_fc2), b_fc2)

    return y


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    y = conv_net(x)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.AdagradOptimizer(1e-2).minimize(cross_entropy)
    # metrics, accuracy/loss
    loss = -tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), axis=1)
    loss = tf.reduce_mean(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # train
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    report_metrics = True
    for i in range(200):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print('Epoch', i)
        train_loss_value = sess.run(loss, feed_dict={x: batch_xs,
                                                     y_: batch_ys})
        train_loss_list.append(train_loss_value)
        if report_metrics:
            test_loss_value = sess.run(loss, feed_dict={x: mnist.test.images,
                                                        y_: mnist.test.labels})
            test_loss_list.append(test_loss_value)
            train_accuracy_value = sess.run(accuracy,
                                            feed_dict={x: batch_xs,
                                                       y_: batch_ys})
            test_accuracy_value = sess.run(accuracy,
                                           feed_dict={x: mnist.test.images,
                                                      y_: mnist.test.labels})
            train_accuracy_list.append(train_accuracy_value)
            test_accuracy_list.append(test_accuracy_value)
            print('  training loss {}, test loss {}'.format(
                train_loss_value, test_loss_value))
            print('  training accuracy {}, test accuracy {}'.format(
                train_accuracy_value, test_accuracy_value))
        else:
            print(' training loss', train_loss_value)
    np.savetxt('training_data/train_loss_part2.2.txt',
               np.array(train_loss_list))
    if report_metrics:
        np.savetxt('training_data/test_loss_part2.2.txt',
                   np.array(test_loss_list))
        np.savetxt('training_data/train_accuracy_part2.2.txt',
                 np.array(train_accuracy_list))
        np.savetxt('training_data/test_accuracy_part2.2.txt',
                 np.array(test_accuracy_list))
    # test trained model
    print('\nAccuracy on test set',
          sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))
    y_pred = sess.run(y, feed_dict={x: mnist.test.images})
    confusion = np.zeros((10, 10))
    for m in range(y_pred.shape[0]):
        confusion[np.argmax(y_pred[m]),
                  np.argmax(mnist.test.labels[m])] += 1
    np.savetxt('training_data/test_confusion_part2.2.txt', confusion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
