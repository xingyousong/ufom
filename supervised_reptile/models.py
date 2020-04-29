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

        self.clip_grads = optim_kwargs['clip_grads']
        self.clip_grad_value = optim_kwargs['clip_grad_value']

        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))

        if optim_kwargs['mlp']:
            hidden_sizes = [256, 128, 64, 64]

        for index in range(optim_kwargs['n_layers']):

            if optim_kwargs['mlp']:
                out = tf.layers.dense(out, hidden_sizes[index])
            else:
                out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')

            out = tf.layers.batch_normalization(out, training=True)
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
 
        self.clip_grads = optim_kwargs['clip_grads']
        self.clip_grad_value = optim_kwargs['clip_grad_value']

        self.input_ph = tf.placeholder(tf.float32, shape=(None, 84, 84, 3))
        out = self.input_ph

        for index in range(optim_kwargs['n_layers']):
            out = tf.layers.conv2d(out, 32, 3, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.layers.max_pooling2d(out, 2, 2, padding='same')
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

def get_cross_entropy(logits, labels, num_classes):

    logits = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
    labels = tf.one_hot(labels, num_classes)

    result = -tf.reduce_sum(logits*labels, axis=1)

    return result
