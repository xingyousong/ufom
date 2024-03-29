"""
Training helpers for supervised meta-learning.
"""

import os
import time
import numpy as np

import tensorflow as tf

from .reptile import Reptile
from .variables import weight_decay

# pylint: disable=R0913,R0914
def train(sess,
          model,
          train_set,
          test_set,
          save_dir,
          num_classes=5,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=10,
          weight_decay_rate=1,
          time_deadline=None,
          train_shots=None,
          transductive=False,
          reptile_fn=Reptile,
          log_fn=print):
    """
    Train a model on a dataset.
    """

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    saver = tf.train.Saver()
    reptile = reptile_fn(sess,
                         model=model,
                         transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))

    if train_shots is None:
        train_shots = num_shots

    accuracy_ph = tf.placeholder(tf.float32, shape=())
    acc_summary = tf.summary.scalar('accuracy', accuracy_ph)

    total_f_calls = 0
    i = 0
    f_calls_ph = tf.placeholder(tf.int32, shape=())
    f_calls_summary = tf.summary.scalar('f_calls', f_calls_ph)

    #merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())

    full_f_calls_per_iter = meta_batch_size*(inner_iters + 1 + inner_iters*(inner_iters - 1)//2)
    max_f_calls = meta_iters*full_f_calls_per_iter

    while total_f_calls < max_f_calls:

        frac_done = total_f_calls / max_f_calls
        cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size

        result = reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                num_classes=num_classes, num_shots=train_shots,
                inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                replacement=replacement, meta_step_size=cur_meta_step_size,
                meta_batch_size=meta_batch_size, frac_done=frac_done)

        if result is not None:
            total_f_calls += result 
        elif reptile.name == 'Reptile':
            total_f_calls += inner_iters*meta_batch_size
        elif reptile.name == 'FOML':
            total_f_calls += (inner_iters + 1)*meta_batch_size

        if i%eval_interval == 0:

            summary = sess.run(f_calls_summary, feed_dict={f_calls_ph: total_f_calls})
            train_writer.add_summary(summary, i)

            accuracies = []

            for dataset, writer in [(train_set, train_writer), (test_set, test_writer)]:

                correct = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                           model.minimize_op, model.predictions,
                                           num_classes=num_classes, num_shots=num_shots,
                                           inner_batch_size=eval_inner_batch_size,
                                           inner_iters=eval_inner_iters, replacement=replacement)

                summary = sess.run(acc_summary, feed_dict={accuracy_ph: correct/num_classes})

                writer.add_summary(summary, i)
                writer.flush()
                accuracies.append(correct / num_classes)

            log_fn('batch %d: train=%f test=%f' % (i, accuracies[0], accuracies[1]))

        if i % 100 == 0 or i == meta_iters-1:
            saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
        if time_deadline is not None and time.time() > time_deadline:
            break

        i += 1
