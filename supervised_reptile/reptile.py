"""
Supervised Reptile learning and evaluation on arbitrary
datasets.
"""

import random

import tensorflow as tf
import numpy as np

from .variables import (interpolate_vars, average_vars, subtract_vars, add_vars, scale_vars,
                        VariableState)


class Reptile:
    """
    A meta-learning session.

    Reptile can operate in two evaluation modes: normal
    and transductive. In transductive mode, information is
    allowed to leak between test samples via BatchNorm.
    Typically, MAML is used in a transductive manner.
    """
    def __init__(self, session, variables=None, transductive=False, pre_step_op=None, model=None):
        self.session = session
        self._model_state = VariableState(self.session, variables or tf.trainable_variables())
        self._full_state = VariableState(self.session,
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        self._transductive = transductive
        self._pre_step_op = pre_step_op

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size, on_eval_iter=None):
        """
        Perform a Reptile training step.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.
          meta_step_size: interpolation coefficient.
          meta_batch_size: how many inner-loops to run.
        """
        old_vars = self._model_state.export_variables()
        new_vars = []
        for _ in range(meta_batch_size):
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            for batch in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                inputs, labels = zip(*batch)
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            new_vars.append(self._model_state.export_variables())
            self._model_state.import_variables(old_vars)
        new_vars = average_vars(new_vars)
        self._model_state.import_variables(interpolate_vars(old_vars, new_vars, meta_step_size))

        return None

    def evaluate(self,
                 dataset,
                 input_ph,
                 label_ph,
                 minimize_op,
                 predictions,
                 num_classes,
                 num_shots,
                 inner_batch_size,
                 inner_iters,
                 replacement):
        """
        Run a single evaluation of the model.

        Samples a few-shot learning task and measures
        performance.

        Args:
          dataset: a sequence of data classes, where each data
            class has a sample(n) method.
          input_ph: placeholder for a batch of samples.
          label_ph: placeholder for a batch of labels.
          minimize_op: TensorFlow Op to minimize a loss on the
            batch specified by input_ph and label_ph.
          predictions: a Tensor of integer label predictions.
          num_classes: number of data classes to sample.
          num_shots: number of examples per data class.
          inner_batch_size: batch size for every inner-loop
            training iteration.
          inner_iters: number of inner-loop iterations.
          replacement: sample with replacement.

        Returns:
          The number of correctly predicted samples.
            This always ranges from 0 to num_classes.
        """
        train_set, test_set = _split_train_test(
            _sample_mini_dataset(dataset, num_classes, num_shots+1))
        old_vars = self._full_state.export_variables()
        for batch in _mini_batches(train_set, inner_batch_size, inner_iters, replacement):
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        self._full_state.import_variables(old_vars)
        return num_correct

    def _test_predictions(self, train_set, test_set, input_ph, predictions):
        if self._transductive:
            inputs, _ = zip(*test_set)
            return self.session.run(predictions, feed_dict={input_ph: inputs})
        res = []
        for test_sample in test_set:
            inputs, _ = zip(*train_set)
            inputs += (test_sample[0],)
            res.append(self.session.run(predictions, feed_dict={input_ph: inputs})[-1])
        return res


class MAML(Reptile):
    '''
    Not compatible with adaptive optimizers in `minimize_op`.
    '''

    def __init__(self, session, mode=None, model=None, tail_shots=None, variables=None,
            transductive=False, pre_step_op=None, mc_iters=None, compute_errors=None,
            hess_sum_approx=None):

        self.session = session

        if variables is None:
            variables = [y for x, y in model.gvs]

        self._model_state = VariableState(self.session, variables)
        self._full_state = VariableState(self.session,
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        self._transductive = transductive
        self._pre_step_op = pre_step_op

        self.mode = mode
        self.mc_iters = mc_iters
        self.compute_errors = compute_errors
        self.hess_sum_approx = hess_sum_approx

        self._learning_rate = model.learning_rate
        self.tail_shots = tail_shots

        loss_grads = [x for x, y in model.gvs]

        self._hess_prod_arg_phs = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape()) \
            for v in variables]

        second_grad_loss = tf.add_n([tf.reduce_sum(x*y) for x, y in zip(loss_grads,
            self._hess_prod_arg_phs) if x is not None])

        self._hess_prod = [x for x, y in model.optimizer.compute_gradients(second_grad_loss,
            var_list=variables)]

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size,
                   on_eval_iter=None):

        store_vals = (self.compute_errors or self.mode in ['MAML', 'EReptile'])

        if self.compute_errors:
            exact_updates = []

        updates = []

        init_vars = self._model_state.export_variables()

        for meta_batch_index in range(meta_batch_size):

            mini_dataset = list(_sample_mini_dataset(dataset, num_classes, num_shots))
            index_dataset = [(i, x[1]) for i, x in enumerate(mini_dataset)]

            index_mini_batches = [x[0] for x in self._mini_batches(index_dataset, inner_batch_size,
                inner_iters, replacement)]

            last_update = None

            if self.hess_sum_approx:
 
                self._model_state.import_variables(init_vars)

                for batch_index, index_batch in enumerate(index_mini_batches):

                    if batch_index == inner_iters - 1:
                        var_backup = self._model_state.export_variables()

                    batch = [mini_dataset[i] for i in index_batch]
                    inputs, labels = zip(*batch)

                    if self._pre_step_op:
                        self.session.run(self._pre_step_op)

                    self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})

                    if batch_index == inner_iters - 1:
                        hess_sum_last_update = subtract_vars(self._model_state.export_variables(),
                            var_backup)

                hess_sum_term = [np.zeros_like(x) for x in init_vars]

            if store_vals:
                var_states = [init_vars]

            if self.mc_iters > 0:
                mc_term = [np.zeros_like(x) for x in init_vars]
                repeat_iters = self.mc_iters
            else:
                repeat_iters = 1

            for iter_index in range(repeat_iters):

                self._model_state.import_variables(init_vars)

                if self.mc_iters > 0:

                    random_vector = [np.random.randn(*x.shape) for x in init_vars]
                    total_size = np.sum([x.size for x in random_vector])
                    random_vector = scale_vars(random_vector,
                        np.sqrt(total_size)/_norm(random_vector))
                    aggr_vector = random_vector

                    if self.hess_sum_approx:
                        hess_sum_mc_term = [np.zeros_like(x) for x in init_vars]

                for batch_index, index_batch in enumerate(index_mini_batches):

                    if batch_index == inner_iters - 1:
                        var_backup = self._model_state.export_variables()

                    batch = [mini_dataset[i] for i in index_batch]
                    inputs, labels = zip(*batch)

                    if self._pre_step_op:
                        self.session.run(self._pre_step_op)

                    if self.hess_sum_approx and iter_index == 0 and batch_index < inner_iters - 1:

                        hess_prod = self.session.run(self._hess_prod, feed_dict={x:y for x, y in \
                            zip([input_ph, label_ph] + self._hess_prod_arg_phs, [inputs, labels] + \
                            hess_sum_last_update)})

                        hess_sum_term = add_vars(hess_sum_term, hess_prod)

                    if self.mc_iters > 0:

                        if self.hess_sum_approx:

                            hess_prod = self.session.run(self._hess_prod,
                                feed_dict={x:y for x, y in zip([input_ph, label_ph] + \
                                self._hess_prod_arg_phs, [inputs, labels] + random_vector)})

                            hess_sum_mc_term = add_vars(hess_sum_mc_term, hess_prod)

                        hess_prod, _ = self.session.run([self._hess_prod, minimize_op],
                            feed_dict={x:y for x, y in zip([input_ph, label_ph] + \
                            self._hess_prod_arg_phs, [inputs, labels] + aggr_vector)})

                        hess_prod = scale_vars(hess_prod, self._learning_rate)
                        aggr_vector = subtract_vars(aggr_vector, hess_prod)

                    else:
                        self.session.run(minimize_op, feed_dict={input_ph: inputs,
                            label_ph: labels})

                    if iter_index == 0:

                        if batch_index == inner_iters - 1:
                            last_update = subtract_vars(self._model_state.export_variables(),
                                var_backup)
                        elif store_vals:
                            var_states.append(self._model_state.export_variables())

                if self.mc_iters > 0:

                    aggr_vector = subtract_vars(aggr_vector, random_vector)

                    if self.hess_sum_approx:
                        aggr_vector = add_vars(aggr_vector, scale_vars(hess_sum_mc_term,
                            self._learning_rate))

                    dot_product = _dot_product(aggr_vector, last_update)

                    cur_mc_term = scale_vars(last_update, dot_product)
                    mc_term = add_vars(mc_term, cur_mc_term)

            if self.mode in ['MAML', 'EReptile']:
                update = self._get_exact_batch_updates(input_ph, label_ph, minimize_op,
                    self._model_state.export_variables(), last_update, mini_dataset,
                    index_mini_batches, var_states)
            elif self.mode == 'FOML':
                update = last_update
            elif self.mode == 'Reptile':
                pass

            if self.mc_iters > 0:

                mc_term = scale_vars(mc_term, 1/self.mc_iters)

                update = add_vars(update, mc_term)

            if self.hess_sum_approx:
                update = subtract_vars(update, scale_vars(hess_sum_term, self._learning_rate))

            updates.append(update)

            if self.compute_errors:

                exact_update = self._get_exact_batch_updates(input_ph, label_ph, minimize_op,
                    last_update, mini_dataset, index_mini_batches, var_states)

                exact_updates.append(exact_update)

        update = average_vars(updates)

        if self.compute_errors:

            exact_update = average_vars(exact_updates)

            self._model_state.import_variables(add_vars(init_vars, scale_vars(exact_update,
                meta_step_size)))

            diff = subtract_vars(exact_update, update)
            error = _norm(diff)/_norm(exact_update)

            return error

        self._model_state.import_variables(add_vars(init_vars, scale_vars(update, meta_step_size)))

        return None

    def _get_exact_batch_updates(self, input_ph, label_ph, minimize_op, batch_update,
            mini_dataset, index_mini_batches, var_states):

        for batch_index, index_batch in enumerate(index_mini_batches[-2::-1]):

            batch = [mini_dataset[i] for i in index_batch]
            inputs, labels = zip(*batch)

            cur_vars = var_states[len(var_states) - 2 - batch_index]

            self._model_state.import_variables(cur_vars)

            if self._pre_step_op:
                self.session.run(self._pre_step_op)

            hess_prod = self.session.run(self._hess_prod, feed_dict={x:y for x, y in \
                zip([input_ph, label_ph] + self._hess_prod_arg_phs, [inputs, labels] + \
                batch_update)})

            hess_prod = scale_vars(hess_prod, self._learning_rate)

            batch_update = subtract_vars(batch_update, hess_prod)

        return batch_update
 
    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
        """
        Generate inner-loop mini-batches for the task.
        """
        if self.tail_shots is None:
            for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                yield value
            return
        train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
            yield batch
        yield tail

def _power_method(hess_prod_arg_phs, session, hess_prod, input_ph, label_ph, inputs, labels,
        iters=20):

    vector = [np.random.randn(*x.shape) for x in hess_prod_arg_phs]

    for _ in range(iters):

        old_vector = vector
        vector = session.run(hess_prod, feed_dict={x:y for x, y in \
            zip([input_ph, label_ph] + hess_prod_arg_phs, [inputs, labels] + \
            vector)})

        eigenvalue = _dot_product(old_vector, vector)
        norm = _norm(vector)
        vector = scale_vars(vector, 1/norm)

    return np.abs(eigenvalue)

def _dot_product(var1, var2):

    return np.sum([(x*y).sum() for x, y in zip(var1, var2)])

def _norm(var):

    return np.sqrt(_dot_product(var, var))


class FOML(Reptile):
    """
    A basic implementation of "first-order MAML" (FOML).

    FOML is similar to Reptile, except that you use the
    gradient from the last mini-batch as the update
    direction.

    There are two ways to sample batches for FOML.
    By default, FOML samples batches just like Reptile,
    meaning that the final mini-batch may overlap with
    the previous mini-batches.
    Alternatively, if tail_shots is specified, then a
    separate mini-batch is used for the final step.
    This final mini-batch is guaranteed not to overlap
    with the training mini-batches.
    """
    def __init__(self, *args, tail_shots=None, **kwargs):
        """
        Create a first-order MAML session.

        Args:
          args: args for Reptile.
          tail_shots: if specified, this is the number of
            examples per class to reserve for the final
            mini-batch.
          kwargs: kwargs for Reptile.
        """
        super(FOML, self).__init__(*args, **kwargs)
        self.tail_shots = tail_shots

    # pylint: disable=R0913,R0914
    def train_step(self,
                   dataset,
                   input_ph,
                   label_ph,
                   minimize_op,
                   num_classes,
                   num_shots,
                   inner_batch_size,
                   inner_iters,
                   replacement,
                   meta_step_size,
                   meta_batch_size, on_eval_iter=None):
        old_vars = self._model_state.export_variables()
        updates = []
        for _ in range(meta_batch_size):
            mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
            mini_batches = self._mini_batches(mini_dataset, inner_batch_size, inner_iters,
                                              replacement)
            for batch in mini_batches:
                inputs, labels = zip(*batch)
                last_backup = self._model_state.export_variables()
                if self._pre_step_op:
                    self.session.run(self._pre_step_op)
                self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
            updates.append(subtract_vars(self._model_state.export_variables(), last_backup))
            self._model_state.import_variables(old_vars)
        update = average_vars(updates)
        self._model_state.import_variables(add_vars(old_vars, scale_vars(update, meta_step_size)))

        return None

    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):
        """
        Generate inner-loop mini-batches for the task.
        """
        if self.tail_shots is None:
            for value in _mini_batches(mini_dataset, inner_batch_size, inner_iters, replacement):
                yield value
            return
        train, tail = _split_train_test(mini_dataset, test_shots=self.tail_shots)
        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
            yield batch
        yield tail

def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)

def _mini_batches(samples, batch_size, num_batches, replacement):
    """
    Generate mini-batches from some data.

    Returns:
      An iterable of sequences of (input, label) pairs,
        where each sequence is a mini-batch.
    """
    samples = list(samples)
    if replacement:
        for _ in range(num_batches):
            yield random.sample(samples, batch_size)
        return
    cur_batch = []
    batch_count = 0
    while True:
        random.shuffle(samples)
        for sample in samples:
            cur_batch.append(sample)
            if len(cur_batch) < batch_size:
                continue
            yield cur_batch
            cur_batch = []
            batch_count += 1
            if batch_count == num_batches:
                return

def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')
    return train_set, test_set
