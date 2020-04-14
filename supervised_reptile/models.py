"""
Models for supervised meta-learning.
"""

from functools import partial

import numpy as np
import tensorflow as tf

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

            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')

            if not optim_kwargs['nobatchnorm']:
 
                if optim_kwargs['precond']:
                    out -= tf.math.reduce_mean(out, axis=(0, 1, 2), keepdims=True)
                    out = precond_grads(out)

                out = tf.layers.batch_normalization(out, scale=False, training=True)

            if optim_kwargs['nonlin'] == 'softplus':
                out = tf.math.softplus(out*optim_kwargs['temp'])/optim_kwargs['temp']
            else:
                out = tf.nn.relu(out)

        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = get_cross_entropy(self.logits, self.label_ph, num_classes)
        self.predictions = tf.argmax(self.logits, axis=-1)

        self.learning_rate = optim_kwargs['learning_rate']
        optimizer_instance = optimizer(self.learning_rate)

        self.gvs = optimizer_instance.compute_gradients(self.loss)

        self.minimize_op = optimizer_instance.apply_gradients(self.gvs)
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

            out = tf.layers.conv2d(out, 32, 3, padding='same')

            if not optim_kwargs['nobatchnorm']:
 
                if optim_kwargs['precond']:
                    out -= tf.math.reduce_mean(out, axis=(0, 1, 2), keepdims=True)
                    out = precond_grads(out)

                out = tf.layers.batch_normalization(out, scale=False, training=True)

            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')

            if optim_kwargs['nonlin'] == 'softplus':
                out = tf.math.softplus(out*optim_kwargs['temp'])/optim_kwargs['temp']
            else:
                out = tf.nn.relu(out)

        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))
        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = get_cross_entropy(self.logits, self.label_ph, num_classes)
        self.predictions = tf.argmax(self.logits, axis=-1)
  
        self.learning_rate = optim_kwargs['learning_rate']
        optimizer_instance = optimizer(self.learning_rate)
 
        self.gvs = optimizer_instance.compute_gradients(self.loss)

        self.minimize_op = optimizer_instance.apply_gradients(self.gvs)
        self.optimizer = optimizer_instance


@tf.custom_gradient
def precond_grads(x):

    def grad(dy):

        std_values = tf.math.reduce_std(x, axis=(0, 1, 2), keepdims=True)

        return dy*tf.reduce_min(std_values)

    return x, grad

def get_cross_entropy(logits, labels, num_classes):

    logits = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
    labels = tf.one_hot(labels, num_classes)

    result = -tf.reduce_sum(logits*labels, axis=1)

    return result
