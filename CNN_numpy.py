# Implement a 4-layer CNN using numpy only (no tensorflow)
#
# Author: Kevin Fischer
# Date: 4/19/2018

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from multiprocessing import Pool, cpu_count

FLAGS = None


class SumDict(dict):
    """Child of dict that implements a summation operation between
    dictionaries. Entries with the same key are added together."""

    def __add__(self, y):
        return SumDict({x: self.get(x, 0) + y.get(x, 0)
                        for x in set(self).union(y)})


class NNTools(object):
    """Provides a suite of tools for convolutional neural networks."""

    @staticmethod
    def loss(y, y_):
        """
        Implements cross entropy loss for predictions y and labels y_.

        Parameters
        ----------
        y : ndarrray
            Input array of shape (samples, 10)
        y_ : ndarrray
            Input array of shape (samples, 10)

        Returns
        -------
        Output float loss function.
        """
        return np.mean(-np.sum(y_ * np.log(y), axis=1))

    @staticmethod
    def conv_2d(x, W, b=None):
        """
        Implements 2d convolutional layer, including bias and ReLu
        nonlinearity. If the bias is absent, the ReLu is excluded.

        Parameters
        ----------
        x : ndarray
            Input array of shape (samples, height, height, channels)
        W : ndarray
            Input array of shape (window, window, channels, channels_new)
        b : ndarray
            Input array of shape (channels_new,)

        Returns
        -------
        y : ndarray
            Output array of shape (samples, height, height, channels_new)
        """

        (n_x, n_h, n_h, n_c) = x.shape
        (n_f, n_f, n_c, n_c_new) = W.shape

        # zero pad x for 'SAME' padding, only pad height axes of x
        n_pad_l = int((n_f - 1) / 2)
        n_pad_h = int(np.ceil((n_f - 1) / 2))
        pad = [(0, 0), (n_pad_l, n_pad_h), (n_pad_l, n_pad_h), (0, 0)]
        x_pad = np.pad(x, pad, mode='constant')
        # initialize output array
        y = np.zeros((n_x, n_h, n_h, n_c_new))
        for i in range(n_h):
            for j in range(n_h):
                for k in range(n_c_new):
                    # get filter for only one output channel at a time
                    filter = W[:, :, :, k]
                    # get x slice for sliding action
                    x_window = x_pad[:, i:i+n_f, j:j+n_f, :]
                    if b is None:
                        bias = 0
                    else:
                        bias = b[k]
                    # perform kernel operation
                    y[:, i, j, k] = \
                        np.einsum('bxyq,xyq->b', x_window, filter) + bias
        if b is None:
            return y
        else:
            return np.maximum(y, 0)

    @staticmethod
    def conv_2d_backward(x, delta, W_shape):
        """
        Implements backward convolution for computing parameter derivatives of
        the convolutional layers.

        Parameters
        ----------
        x : ndarray
            Input array of shape (samples, height, height, channels)
        delta : ndarray
            Input array of shape (samples, height, height, channels_new)
        W_shape : ndarray
            Input array of shape (window, window, channels, channels_new)

        Returns
        -------
        dW : ndarray
            Output array of shape (window, window, channels, channels_new)
        """

        (n_x, n_h, n_h, n_c) = x.shape
        (n_f, n_f, n_c, n_c_new) = W_shape

        # zero pad x for 'SAME' padding, only pad height axes of x
        n_pad_l = int((n_f - 1) / 2)
        n_pad_h = int(np.ceil((n_f - 1) / 2))
        pad = [(0, 0), (n_pad_l, n_pad_h), (n_pad_l, n_pad_h), (0, 0)]
        x_pad = np.pad(x, pad, mode='constant')
        # initialize output array
        dW = np.zeros(W_shape)
        for m in range(n_f):
            for n in range(n_f):
                # get intput window to convolve
                x_window = x_pad[:, m:m+n_h, n:n+n_h, :]
                # perform kernel operation, averages over the total number of
                # samples
                dW[m, n, :, :] = \
                    np.einsum('kijl,kijo->ol', delta, x_window) / n_x
        return dW

    @staticmethod
    def max_pool_2x2(x):
        """
        Implements 2x2 max pooling. Also returns mask for which activations
        matter in backpropagation.

        Parameters
        ----------
        x : ndarray
            Input array of shape (samples, height, height, channels),
            height is necessarily is an even integer

        Returns
        -------
        a : ndarray
            Output array of shape (samples, height/2, height/2, channels)

        mask : ndarray
            Output array of shape (samples, height, height, channels)
        """

        (n_x, n_h, n_h, n_c) = x.shape
        assert(n_h/2 == int(n_h/2))  # check n_h is even
        n_h_new = int(n_h/2)  # new height is halved
        a = np.zeros((n_x, n_h_new, n_h_new, n_c))
        mask = np.zeros(x.shape)

        # run over all indices
        for l in range(n_x):
            for i in range(n_h_new):
                for j in range(n_h_new):
                    for k in range(n_c):
                        # get 2x2 window
                        x_window = x[l, 2*i:2*i+2, 2*j:2*j+2, k]
                        # find which indices the max came from
                        amax = np.argmax(x_window)
                        imax = 2*i + (amax > 1)
                        jmax = 2*j + amax % 2
                        ymax = x[l, imax, jmax, k]
                        a[l, i, j, k] = ymax  # set max as output

                        # account for ReLu activations from previous payer
                        if ymax > 0:
                            # remember where the max came from
                            mask[l, imax, jmax, k] = 1
        return a, mask

    @staticmethod
    def init_weight(shape):
        """
        Generates random array for a given shape and standard deviation 0.1.
        The difference between using full normal and truncated normal
        distributions is marginal.

        Parameters
        ----------
        shape : tuple

        Returns
        -------
        Output ndarray of shape
        """
        stddev = 0.1
        return np.random.randn(*shape) * stddev

    @staticmethod
    def init_bias(shape):
        """
        Generates bias array for a given shape with values 0.1.

        Parameters
        ----------
        shape : tuple

        Returns
        -------
        bias ndarray of shape
        """
        return 0.1 * np.ones(shape)


class Network:
    """Implements a two-layer convolutional neutral network with two
    fully connected output layers."""

    def __init__(self):
        # convolutional layer #1
        #   size [-1, 28, 28, 1] --> [-1, 14, 14, 32]
        # convolutional layer #2
        #   size [-1, 14, 14, 32] --> [-1, 7, 7, 16]
        # fully connected layer #1
        #   size [-1, 7, 7, 16] --> [-1, 1024]
        # Fully connected layer #2 (final layer)
        #   size [-1, 1024] --> [-1, 10]

        # initialize weights & biases

        # convolutional layers
        W_c1 = NNTools().init_weight([10, 10, 1, 32])
        b_c1 = NNTools().init_bias([32])
        W_c2 = NNTools().init_weight([5, 5, 32, 16])
        b_c2 = NNTools().init_bias([16])

        # fully connected layers
        W_f1 = NNTools().init_weight([7 * 7 * 16, 1024])
        b_f1 = NNTools().init_bias([1024])
        W_f2 = NNTools().init_weight([1024, 10])
        b_f2 = NNTools().init_bias([10])

        # store parameters as dictionaries
        self.params = \
            dict(W_c1=W_c1, W_c2=W_c2, W_f1=W_f1, W_f2=W_f2,
                 b_c1=b_c1, b_c2=b_c2, b_f1=b_f1, b_f2=b_f2)
        self.historical_grads = \
            dict(W_c1=0, W_c2=0, W_f1=0, W_f2=0,
                 b_c1=0, b_c2=0, b_f1=0, b_f2=0)

    def propagate(self, x, y_):
        """
        Perform both the forward and propagation steps of the network, using
        the parameters of the instantiated Network() model.

        Parameters
        ----------
        x : ndarray
            Input array shape (samples, 28 * 28)
        y_ : ndarray
            Input array shape (samples, 10)

        Returns
        -------
        y : ndarrray
            Output array of shape (samples, 10)
        grads : SumDict
            Output SumDict of gradients
        """

        params = self.params  # get parameters dictionary

        # forward propagate

        # we're working with images each 28 x 28 pixels, only one layer
        x = x.reshape([-1, 28, 28, 1])

        # convolutional layer #1
        z_c1 = NNTools().conv_2d(x, params['W_c1'], params['b_c1'])
        a_c1, mask1 = NNTools().max_pool_2x2(z_c1)
        # convolutional layer #2
        z_c2 = NNTools().conv_2d(a_c1, params['W_c2'], params['b_c2'])
        a_c2, mask2 = NNTools().max_pool_2x2(z_c2)
        # fully connected layer #1
        a_f0 = a_c2.reshape([-1, params['W_f1'].shape[0]])
        z_f1 = np.dot(a_f0, params['W_f1']) + params['b_f1']
        a_f1 = np.maximum(z_f1, 0)
        # fully connected layer #2
        z_f2 = np.dot(a_f1, params['W_f2']) + params['b_f2']
        exp_z = np.exp(np.clip(z_f2, None, 100))
        y = exp_z / np.reshape(np.sum(exp_z, axis=1), (exp_z.shape[0], 1))

        # backward propagate

        # fully connected layer #2
        delta = y - y_
        db_f2 = np.mean(delta, axis=0)
        dW_f2 = np.dot(a_f1.transpose(), delta) / a_f1.shape[0]
        # fully connected layer #1
        delta = np.dot(delta, params['W_f2'].transpose()) * (z_f1 > 0)
        db_f1 = np.mean(delta, axis=0)
        dW_f1 = np.dot(a_f0.transpose(), delta) / a_f0.shape[0]
        delta = np.dot(delta, params['W_f1'].transpose())
        # convolutional layer #2
        delta = delta.reshape([-1, 7, 7, 16])
        #   through maxpool
        delta = mask2 * delta.repeat(2, axis=1).repeat(2, axis=2)
        db_c2 = np.mean(np.sum(delta, axis=(1, 2)), axis=0)
        dW_c2 = NNTools().conv_2d_backward(a_c1, delta, params['W_c2'].shape)
        W_deconv = np.swapaxes(params['W_c2'], 2, 3)  # swaps new/prev # of
        # channels
        delta = NNTools().conv_2d(delta, W_deconv)
        # convolutional layer #1
        #   through maxpool
        delta = mask1 * delta.repeat(2, axis=1).repeat(2, axis=2)
        db_c1 = np.mean(np.sum(delta, axis=(1, 2)), axis=0)
        dW_c1 = NNTools().conv_2d_backward(x, delta, params['W_c1'].shape)

        grads = SumDict(
            dict(W_c1=dW_c1, W_c2=dW_c2, W_f1=dW_f1, W_f2=dW_f2,
                 b_c1=db_c1, b_c2=db_c2, b_f1=db_f1, b_f2=db_f2)
        )

        return y, SumDict(grads)

    def update_params(self, grads, eta):
        """
        Perform gradient descent using the adaptive gradient method.

        Parameters
        ----------
        grads : dict
            Input dictionary with parameters of the network
        eta : float
            Learning rate

        Returns
        -------
        y : ndarrray
            Output array of shape (samples, 10)
        """

        eps = 1e-10
        Gtt = self.historical_grads
        for key in Gtt.keys():
            # normalize gradients by their past histories
            Gtt[key] += grads[key] ** 2
            self.params[key] -= eta * grads[key] / (eps + np.sqrt(Gtt[key]))

    @staticmethod
    def forward_pass(x, params):
        """
        Estimates labels y from forward propagation of the network.

        Parameters
        ----------
        x : ndarray
            Input array shape (samples, 28 * 28)
        params : dict
            Input dictionary with parameters of the network

        Returns
        -------
        y : ndarrray
            Output array of shape (samples, 10)
        """

        x = x.reshape([-1, 28, 28, 1])
        # convolutional layer #1
        x = NNTools().conv_2d(x, params['W_c1'], params['b_c1'])
        x, _ = NNTools().max_pool_2x2(x)
        # convolutional layer #2
        x = NNTools().conv_2d(x, params['W_c2'], params['b_c2'])
        x, _ = NNTools().max_pool_2x2(x)
        # fully connected layer #1
        x = x.reshape([-1, params['W_f1'].shape[0]])
        x = np.maximum(np.dot(x, params['W_f1']) + params['b_f1'], 0)
        # fully connected layer #2 (final layer)
        x = np.dot(x, params['W_f2']) + params['b_f2']
        x = np.exp(np.clip(x, None, 100))
        y = x / np.reshape(np.sum(x, axis=1), (x.shape[0], 1))  # softmax

        return y


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    x_test = mnist.test.images
    y_test_ = mnist.test.labels
    cnn = Network()  # instantiate the CNN
    ncpus = cpu_count() - 1
    multiprocess = True
    # train
    loss_list = []
    for i in range(200):
        x, y_ = mnist.train.next_batch(100)
        if multiprocess and ncpus > 1:
            # break up x and y_ into chunks
            xs = np.array_split(x, ncpus)
            y_s = np.array_split(y_, ncpus)
            # multiprocess
            with Pool() as p:
                results = p.starmap(cnn.propagate, zip(xs, y_s))
            ys, grads_multi = zip(*results)  # unzip results
            y = np.concatenate(ys)  # get back to single array (samples, 10)
            # add up all the gradients together
            grads = SumDict()
            for grad in grads_multi:
                grads = grads + grad
        else:
            y, grads = cnn.propagate(x, y_)
        cnn.update_params(grads, eta=1e-2)
        # this is already a bit slower than the tf implementation, so just
        # compute training loss...
        loss_list.append(NNTools.loss(y, y_))
        print('Epoch {}, training loss {}'.format(i, loss_list[-1]))
    np.savetxt('training_data/training_loss_part2.3.txt', np.array(loss_list))
    # Test trained model
    y_test = cnn.forward_pass(x_test, cnn.params)
    accuracy = np.equal(np.argmax(y_test, 1), np.argmax(y_test_, 1))
    accuracy = np.mean(accuracy)
    print('\nAccuracy on test set', accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
