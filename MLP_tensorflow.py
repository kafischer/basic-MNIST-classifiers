# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2018 Kevin A. Fischer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    # metrics
    loss = -tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)), axis=1)
    loss = tf.reduce_mean(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    for i in range(200):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        train_loss_value = sess.run(loss, feed_dict={x: batch_xs,
                                                     y_: batch_ys})
        test_loss_value = sess.run(loss, feed_dict={x: mnist.test.images,
                                                    y_: mnist.test.labels})
        train_loss_list.append(train_loss_value)
        test_loss_list.append(test_loss_value)
        train_accuracy_value = sess.run(accuracy, feed_dict={x: batch_xs,
                                                             y_: batch_ys})
        test_accuracy_value = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                            y_: mnist.test.labels})
        train_accuracy_list.append(train_accuracy_value)
        test_accuracy_list.append(test_accuracy_value)
        print('Epoch', i)
        print('  training loss {}, test loss {}'.format(
            train_loss_value, test_loss_value))
        print('  training accuracy {}, test accuracy {}'.format(
            train_accuracy_value, test_accuracy_value))
    np.savetxt('training_data/train_loss_part2.1.txt', np.array(train_loss_list))
    np.savetxt('training_data/test_loss_part2.1.txt', np.array(test_loss_list))
    np.savetxt('training_data/train_accuracy_part2.1.txt',
               np.array(train_accuracy_list))
    np.savetxt('training_data/test_accuracy_part2.1.txt',
               np.array(test_accuracy_list))
    y_pred = sess.run(y, feed_dict={x: mnist.test.images})
    confusion = np.zeros((10, 10))
    for m in range(y_pred.shape[0]):
        confusion[np.argmax(y_pred[m]),
                  np.argmax(mnist.test.labels[m])] += 1
    np.savetxt('training_data/test_confusion_part2.1.txt', confusion)

    # Test trained model
    print('\nAccuracy on test set',
          sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)