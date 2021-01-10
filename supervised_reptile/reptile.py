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
        self.name = 'Reptile'
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
                   meta_batch_size,
                   frac_done):
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


class UnbMAML(Reptile):
    '''
    Not compatible with adaptive optimizers in `minimize_op`.
    '''

    def __init__(self, session, mode=None, model=None, tail_shots=None, variables=None,
            transductive=False, pre_step_op=None, exact_prob=None, on_resampling=None):

        self.name = 'UnbMAML'
        self.session = session

        if variables is None:
            variables = [y for x, y in model.gvs]

        self._model_state = VariableState(self.session, variables)
        self._full_state = VariableState(self.session,
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        self._transductive = transductive
        self._pre_step_op = pre_step_op

        self.mode = mode
        self.model = model
        self.on_resampling = on_resampling

        if exact_prob < 0:
            self.min_exact_prob = 0.05
            self.beta = 0.99
            self.D2_smoothed = 0.0
            self.V2_smoothed = 0.0
            self.smooth_steps = 0

        self.exact_prob = exact_prob

        self._learning_rate = model.learning_rate

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
                   replacement, # not used
                   meta_step_size,
                   meta_batch_size,
                   frac_done):

        updates = []

        init_vars = self._model_state.export_variables()

        f_calls = 0

        for meta_batch_index in range(meta_batch_size):

            if self.exact_prob >= -1e-8:
                exact_prob = self.exact_prob
            elif self.smooth_steps == 0:
                exact_prob = self.min_exact_prob
            else:

                D2 = self.D2_smoothed/(1 - np.power(self.beta, self.smooth_steps))
                V2 = self.V2_smoothed/(1 - np.power(self.beta, self.smooth_steps))

                D2 = D2/10

                Cdet = inner_iters + 1
                Crnd = ((inner_iters - 1)/2 + 1)*inner_iters

                if D2 >= 0.5*Crnd*V2/(Cdet + Crnd):
                    exact_prob = 1.0
                else:

                    a = (V2 - D2)*Crnd
                    b = -D2*Crnd
                    c = -2*D2*Cdet

                    p1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
                    p2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

                    if p1 > 0 and p1 < 1:
                        exact_prob = p1
                    else:
                        exact_prob = p2

                    if exact_prob < self.min_exact_prob:
                        exact_prob = self.min_exact_prob

            on_exact = (np.random.rand() < exact_prob)

            if on_exact:
                f_calls += (inner_iters - 1)*inner_iters//2 + inner_iters
                var_states = [init_vars]
            else:
                f_calls += inner_iters + 1

            if self.on_resampling:
                mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots)
                mini_batches = list(self._mini_batches(mini_dataset, inner_batch_size, inner_iters, False))
            else:
                train_batch, test_batch = _sample_train_test_batch(dataset, num_classes, num_shots)
                mini_batches = [train_batch]*(inner_iters - 1) + [test_batch]

            self._model_state.import_variables(init_vars)

            for batch in mini_batches[:-1]:

                inputs, labels = zip(*batch)

                if self._pre_step_op:
                    self.session.run(self._pre_step_op)

                self.session.run(minimize_op, feed_dict={input_ph: inputs,
                    label_ph: labels})

                if on_exact:
                    var_states.append(self._model_state.export_variables())

            update = self._get_test_update(input_ph, label_ph, mini_batches[-1], minimize_op)

            if on_exact:

                exact_update = self._get_exact_update(input_ph, label_ph, mini_batches,
                    var_states, minimize_op)

                update = add_vars(update, scale_vars(subtract_vars(exact_update, update),
                    1/exact_prob))

                if self.exact_prob < 0:

                    self.smooth_steps += 1

                    cur_D2 = sum([((fo - o)**2).sum() for fo, o in zip(update, exact_update)])
                    cur_V2 = sum([(o**2).sum() for o in exact_update])

                    self.D2_smoothed = self.beta*self.D2_smoothed + (1 - self.beta)*cur_D2
                    self.V2_smoothed = self.beta*self.V2_smoothed + (1 - self.beta)*cur_V2

            updates.append(update)

        update = average_vars(updates)
        self._model_state.import_variables(add_vars(init_vars, scale_vars(update, meta_step_size)))

        return f_calls

    def _get_test_update(self, input_ph, label_ph, test_batch, minimize_op):

        inputs, labels = zip(*test_batch)

        var_backup = self._model_state.export_variables()

        if self._pre_step_op:
            self.session.run(self._pre_step_op)

        self.session.run(minimize_op, feed_dict={input_ph: inputs,
            label_ph: labels})

        update = subtract_vars(self._model_state.export_variables(), var_backup)

        self._model_state.import_variables(var_backup)

        return update

    def _get_exact_update(self, input_ph, label_ph, mini_batches, var_states,
            minimize_op):

        self._model_state.import_variables(var_states[-1])
        update = self._get_test_update(input_ph, label_ph, mini_batches[-1], minimize_op)

        for var_index in range(len(var_states) - 2, -1, -1):

            inputs, labels = zip(*mini_batches[var_index])

            self._model_state.import_variables(var_states[var_index])

            if self._pre_step_op:
                self.session.run(self._pre_step_op)

            hess_prod = self.session.run(self._hess_prod, feed_dict={x:y for x, y in \
                zip([input_ph, label_ph] + self._hess_prod_arg_phs, [inputs, labels] + \
                update)})

            hess_prod = scale_vars(hess_prod, self._learning_rate)
            update = subtract_vars(update, hess_prod)

            if self.model.clip_grads:
                clip_value = self.model.clip_grad_value
                update = [np.clip(x, -clip_value, clip_value) for x in update]

        return update

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

        if self.on_resampling:
            train_set, test_set = _split_train_test(_sample_mini_dataset(dataset, num_classes,
                    num_shots + 1))
            mini_batches = list(_mini_batches(train_set, inner_batch_size, inner_iters, False))
        else:
            train_set, test_set = _sample_train_test_batch(dataset, num_classes, num_shots)
            mini_batches = [train_set]*(inner_iters - 1)
 
        old_vars = self._full_state.export_variables()
        for batch in mini_batches:
            inputs, labels = zip(*batch)
            if self._pre_step_op:
                self.session.run(self._pre_step_op)
            self.session.run(minimize_op, feed_dict={input_ph: inputs, label_ph: labels})
        test_preds = self._test_predictions(train_set, test_set, input_ph, predictions)
        num_correct = sum([pred == sample[1] for pred, sample in zip(test_preds, test_set)])
        self._full_state.import_variables(old_vars)
        return num_correct

    def _mini_batches(self, mini_dataset, inner_batch_size, inner_iters, replacement):

        train, tail = _split_train_test(mini_dataset, test_shots=1)

        for batch in _mini_batches(train, inner_batch_size, inner_iters - 1, replacement):
            yield batch

        yield tail

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
        self.name = 'FOML'
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
                   meta_batch_size,
                   frac_done):
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

def _sample_train_test_batch(dataset, num_classes, num_shots):

    mini_dataset = _sample_mini_dataset(dataset, num_classes, num_shots + 1)

    return _split_train_test(mini_dataset)

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
