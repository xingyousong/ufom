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
        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))

        
        #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
        #                                                           logits=self.logits)
        self.loss = get_cross_entropy(self.logits, self.label_ph, num_classes)
        self.predictions = tf.argmax(self.logits, axis=-1)

        optimizer_instance = optimizer(**optim_kwargs)

        self.learning_rate = optim_kwargs['learning_rate']
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
        for _ in range(4):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
            out = tf.nn.relu(out)
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        #self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
        #                                                           logits=self.logits)
        self.loss = get_cross_entropy(self.logits, self.label_ph, num_classes)
        self.predictions = tf.argmax(self.logits, axis=-1)
 
        optimizer_instance = optimizer(**optim_kwargs)
 
        self.learning_rate = optim_kwargs['learning_rate']
        self.minimize_op = optimizer_instance.minimize(self.loss)
        self.optimizer = optimizer_instance


def get_cross_entropy(logits, labels, num_classes):

    logits = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
    labels = tf.one_hot(labels, num_classes)

    result = -tf.reduce_sum(logits*labels)

    return result
