import tensorflow as tf
import numpy as np

from functools import partial


DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)


class NAS:

    def __init__(self, input_shape, hidden_size, output_size, bin_size, emb_size,
            optimizer=DEFAULT_OPTIMIZER, **optim_kwargs):

        input_size = np.prod(input_shape)
        total_size = input_size + hidden_size + output_size

        nonlinearities = [tf.identity, tf.nn.relu, tf.math.sigmoid]

        self.embeddings = tf.Variable(tf.random.normal([total_size, emb_size]), name='embedding',
            trainable=True)
        self.in_weights = tf.Variable(tf.random.normal([total_size]), name='in_weights',
            trainable=True)
        self.out_weights = tf.Variable(tf.random.normal([total_size]), name='out_weights',
            trainable=True)
        self.biases = tf.Variable(tf.zeros([total_size]), name='biases', trainable=True)
        self.nonlin_logweights = tf.Variable(tf.random.normal([total_size, len(nonlinearities)]),
            name='nonlin_logweights', trainable=True)

        embeddings = self.embeddings/tf.math.sqrt(tf.reduce_sum(self.embeddings**2, axis=1,
            keepdims=True))

        self.input_ph = tf.placeholder(tf.float32, shape=(None,) + input_shape)

        inputs = tf.reshape(self.input_ph, [-1, input_size])*1000

        omegas = sample_stiefel_matrix(emb_size, emb_size)

        r_features = tf.matmul(embeddings, omegas)*2*np.sqrt(2)*np.pi
        r_features = tf.concat([tf.math.sin(r_features), tf.math.cos(r_features)], axis=1)

        nonlin_probs = tf.math.exp(self.nonlin_logweights - tf.math.reduce_logsumexp(
            self.nonlin_logweights, axis=1, keepdims=True))

        aggr_r_features = tf.reduce_sum(r_features[:input_size], axis=0)
        aggr_weighted_r_features = tf.matmul(inputs, r_features[:input_size]*\
            self.out_weights[:input_size, None])

        slices = [slice(x, x + bin_size) for x in range(input_size, input_size + \
            hidden_size, bin_size)] + [slice(input_size + hidden_size, None)]
        on_output_flags = [False]*(hidden_size//bin_size) + [True]

        self.logits = None

        for cur_slice, on_output in zip(slices, on_output_flags):

            hidden_inputs = self.biases[None, cur_slice] + self.in_weights[None, cur_slice]*\
                tf.matmul(aggr_weighted_r_features, r_features[cur_slice], transpose_b=True)/\
                tf.matmul(aggr_r_features[None, :], r_features[cur_slice], transpose_b=True)

            hidden_inputs = tf.layers.batch_normalization(hidden_inputs, training=True)

            xs = 0.0

            for nonlin_index, nonlinearity in enumerate(nonlinearities):
                xs = xs + nonlinearity(hidden_inputs)*nonlin_probs[None, cur_slice, nonlin_index]

            if on_output:
                self.logits = xs
            else:

                aggr_r_features = aggr_r_features + tf.reduce_sum(r_features[cur_slice], axis=0)
                aggr_weighted_r_features = aggr_weighted_r_features + \
                    tf.matmul(xs, r_features[cur_slice]*self.out_weights[cur_slice, None])

        self.logits_min = tf.reduce_min(self.logits)
        self.logits_max = tf.reduce_max(self.logits)

        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = get_cross_entropy(self.logits, self.label_ph, output_size)
        self.predictions = tf.argmax(self.logits, axis=-1)

        optimizer_obj = optimizer(optim_kwargs['learning_rate'])
        gvs = optimizer_obj.compute_gradients(self.loss)

        self.g_norm = tf.math.sqrt(tf.reduce_sum(gvs[0][0]**2))

        self.minimize_op = optimizer_obj.apply_gradients(gvs)


def sample_stiefel_matrix(height, width):

  subspace = tf.random.normal([height, width])
  subspace = tf.linalg.band_part(subspace, -1, 0)
  subspace = subspace/tf.math.sqrt(tf.reduce_sum(subspace**2, axis=0, keepdims=True))

  S = tf.linalg.band_part(tf.matmul(subspace, subspace, transpose_a=True), 0, -1) - \
    0.5*tf.eye(width)

  result = tf.eye(height, num_columns=width) - tf.matmul(subspace,
      tf.matmul(tf.linalg.inv(S), subspace[:width], transpose_b=True))

  return result

def get_cross_entropy(logits, labels, num_classes):

    logits = logits - tf.reduce_logsumexp(logits, axis=-1, keepdims=True)
    labels = tf.one_hot(labels, num_classes)

    result = -tf.reduce_sum(logits*labels, axis=1)

    return result
