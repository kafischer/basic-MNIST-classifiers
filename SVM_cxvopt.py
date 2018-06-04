# Classify MNIST digits using a SVM implemented from a quadratic programming
# package.
#
# Author: Kevin Fischer
# Date: 4/20/2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import sys
import tensorflow as tf

from scipy.spatial.distance import pdist, squareform, cdist
from cvxopt import matrix, solvers
from itertools import combinations
from multiprocessing import Pool, cpu_count

from tensorflow.examples.tutorials.mnist import input_data
FLAGS = None

# decay length of rbf kernel
gamma = 0.05


def svm(x, y):
    """
    Solve the SVM dual problem with rbf kernel and l1 regularization constant C

    minimize     1/2 alpha.T K alpha + alpha.T
    subject to   0 <= alpha <= C
                 y.T alpha = 0

    For images x1 and x2, the rbf kernel is given by

        exp(-gamma || y1 - y2 ||^2)

    Parameters
    ----------
    x : ndarray
        Input array of images shape (samples, 784)
    y : ndarray
        Input array of labels shape (samples, 1)

    Returns
    -------
    Output ndarray of Lagrange multipliers, alpha
    """

    C = 5  # regularization constant
    n_samples = x.shape[0]
    Y = np.dot(y.reshape(-1, 1), y.reshape(1, -1))  # outer product
    K = np.exp(-gamma * squareform(pdist(x, 'sqeuclidean')))  # rbf kernel
    P = matrix(K * Y)
    q = matrix(-np.ones((n_samples, 1)))

    # alphas >= 0
    G_std = matrix(np.diag(-np.ones(n_samples)))
    h_std = matrix(np.zeros(n_samples))
    # alphas <= C
    G_slack = matrix(np.diag(np.ones(n_samples)))
    h_slack = matrix(np.ones(n_samples) * C)
    # stack G and h matrices together
    G = matrix(np.vstack((G_std, G_slack)))
    h = matrix(np.vstack((h_std, h_slack)))

    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    # minimize     1/2 x.T P x + q.T x
    # subject to   G x <= h = 0 / C
    #               A x = b = 0
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    return np.array(sol['x'])


def train_classifier(x_train, y_train_, x_test, num0, num1):
    """
    Train a binary SVC, which decides whether an image contains class num0 or
    class num1.

    Parameters
    ----------
    x_train : ndarray
        Input array of images shape (samples, 784)
    y_train_ : ndarray
        Input array of labels shape (samples, 1)
    x_test : ndarray
        Input array of images shape (samples, 784)
    num0 : int
        Input int of class number.
    num1 : int
        Input int of class number.

    Returns
    -------
    Output ndarray of Lagrange multipliers, alpha
    """

    # select out only examples between class num0/num1
    x0 = x_train[y_train_[:, num0] > 0]  # get x samples of class num0
    y0_ = np.ones(x0.shape[0])  # num0 gets label +1
    x1 = x_train[y_train_[:, num1] > 0]  # get x samples of class num1
    y1_ = -np.ones(x1.shape[0])  # num1 gets label -1
    x = np.concatenate([x0, x1])
    y_ = np.concatenate([y0_, y1_])
    # train the binary SVM
    alphas = svm(x, y_).reshape(-1, 1)
    alphas *= y_.reshape(-1, 1)  # alpha always appears with y_
    # extract intercept b by finding average between
    # min value of class num0 and max value of class num1
    K = np.exp(-gamma * cdist(x, x[y_ > 0], 'sqeuclidean'))
    w_min = np.min(np.sum(alphas * K, axis=0))
    K = np.exp(-gamma * cdist(x, x[y_ < 0], 'sqeuclidean'))
    w_max = np.max(np.sum(alphas * K, axis=0))
    b = -(w_min + w_max) / 2  # intercept
    # compute classification accuracy on the training set (should be 1.0)
    K_predict = np.exp(-gamma * squareform(pdist(x, 'sqeuclidean')))
    predict = np.sign(np.sum(alphas * K_predict, axis=0) + b)
    accuracy = np.mean(np.equal(predict, y_))
    print('Classifier {} vs {}, training accuracy {}'.format(
        num0, num1, accuracy))
    # compute binary predictions on entire MNIST test set
    K_predict = np.exp(-gamma * cdist(x, x_test, 'sqeuclidean'))
    predict = np.sign(np.sum(alphas * K_predict, axis=0) + b)
    predict[predict > 0] *= num0  # label class num0 by its number
    predict[predict < 0] *= -num1  # label class num1 by its number
    return np.array(predict.reshape(1, -1), dtype='int')  # return predictions


def main(_):
    """
    Use 10 choose 2 combinations of binary SVCs to classify the MNIST digits,
    which may take on values 0-9. The prediction is determined by the most
    popular digit voted among the binary classifiers.
    """

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    accuracy_list = []
    nbatch_list = range(200, 8000, 200)
    for nbatch in nbatch_list:  # number training samples
        # 10 choose 2 possible classifiers
        classifiers = combinations(range(10), 2)

        x_train = mnist.train.images[:nbatch, :]
        y_train_ = mnist.train.labels[:nbatch, :]
        # get test data
        x_test = mnist.test.images
        y_test = mnist.test.labels

        predictions = []
        ncpus = cpu_count() - 1
        multiprocess = True
        # train
        if multiprocess and ncpus > 1:
            # multiprocess
            with Pool() as p:
                # dispatch each classifier training task to a separate thread
                c_list = list(classifiers)
                n = len(c_list)
                num0, num1 = list(zip(*c_list))
                args = zip([x_train]*n, [y_train_]*n, [x_test]*n, num0, num1)
                predictions = p.starmap(train_classifier, args)
        else:
            # serial process
            for (num0, num1) in classifiers:
                predictions.append(
                    train_classifier(x_train, y_train_, x_test, num0, num1))
        # compare predictions in test set to labels
        y_pred = np.zeros(y_test.shape)
        for i, pred in enumerate(np.concatenate(predictions).T):
            # for each image, pick the most popular digit voted among the
            # binary classifiers
            y_pred[i, np.argmax(np.bincount(pred))] = 1
        # compute test accuracy
        accuracy = np.equal(np.argmax(y_pred, 1), np.argmax(y_test, 1))
        accuracy = np.mean(accuracy)
        print('\n\n{} training examples, test accuracy {}\n\n'.format(
            nbatch, accuracy))
        accuracy_list.append(accuracy)
    np.savetxt('training_data/test_accuracy_svm.txt', np.array(accuracy_list))
    confusion = np.zeros((10, 10))
    for m in range(y_pred.shape[0]):
        confusion[np.argmax(y_pred[m]), np.argmax(y_test[m])] += 1
    np.savetxt('training_data/test_confusion_svm.txt', confusion)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)