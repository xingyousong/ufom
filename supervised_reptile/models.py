"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow.compat.v1 as tf

DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

# pylint: disable=R0903
class OmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):

        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))

        for index in range(4):

            if optim_kwargs['orth_norm'] and index > 0:
                kernel = tf.Variable(initial_value=tf.random.normal([9*64, 64]), trainable=True)
                kernel = compute_cwy(kernel)
                kernel = tf.reshape(kernel, [3, 3, 64, 64])
                bias = tf.Variable(initial_value=tf.zeros([1, 1, 1, 64]), trainable=True)
 
                out = tf.nn.conv2d(out, kernel, 2, 'SAME')/3 + bias
            else:
                out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')/3

            if not optim_kwargs['nobatchnorm']:
                out = tf.layers.batch_normalization(out, training=True)

            if optim_kwargs['nonlin'] == 'softplus':
                out = tf.math.softplus(out*optim_kwargs['temp'])/optim_kwargs['temp']
            else:
                out = tf.nn.relu(out)

        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
        #                                                           logits=self.logits)
        self.loss = get_cross_entropy(self.logits, self.label_ph, num_classes)
        self.predictions = tf.argmax(self.logits, axis=-1)

        self.learning_rate = optim_kwargs['learning_rate']
        optimizer_instance = optimizer(self.learning_rate)
        self.minimize_op = optimizer_instance.minimize(self.loss)
        self.optimizer = optimizer_instance

# pylint: disable=R0903
class MiniImageNetModel:
    """
    A model for Mini-ImageNet classification.
    """
    def __init__(self, num_classes, optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph
        for index in range(4):

            if optim_kwargs['orth_norm'] and index > 0:
                kernel = tf.Variable(initial_value=tf.random.normal([9*32, 32]), trainable=True)
                kernel = compute_cwy(kernel)
                kernel = tf.reshape(kernel, [3, 3, 32, 32])
                bias = tf.Variable(initial_value=tf.zeros([1, 1, 1, 32]), trainable=True)
 
                out = tf.nn.conv2d(out, kernel, 1, 'SAME')/3 + bias
            else:
                out = tf.layers.conv2d(out, 32, 3, padding='same')/3

            if optim_kwargs['batchnorm']:
                out = tf.layers.batch_normalization(out, training=True)

            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')

            if optim_kwargs['nonlin'] == 'softplus':
                out = tf.math.softplus(out*optim_kwargs['temp'])/optim_kwargs['temp']
            else:
                out = tf.nn.relu(out)

        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
        #                                                           logits=self.logits)
        self.loss = get_cross_entropy(self.logits, self.label_ph, num_classes)
        self.predictions = tf.argmax(self.logits, axis=-1)
  
        self.learning_rate = optim_kwargs['learning_rate']
        optimizer_instance = optimizer(self.learning_rate)
        self.minimize_op = optimizer_instance.minimize(self.loss)
        self.optimizer = optimizer_instance


def compute_cwy(params):

    params = params/tf.math.sqrt(tf.reduce_sum(params**2, axis=0, keepdims=True))

    S = tf.linalg.band_part(tf.matmul(params, params, transpose_a=True), 0, -1) - \
        0.5*tf.eye(params.shape[1])

    result = tf.eye(params.shape[0], num_columns=params.shape[1]) - tf.matmul(params,
        tf.matmul(tf.linalg.inv(S), params[:params.shape[1]], transpose_b=True))

    return result

def get_cross_entropy(logits, labels, num_classes):

    logits = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
    labels = tf.one_hot(labels, num_classes)

    result = -tf.reduce_sum(logits*labels, axis=1)

    return result
