"""
Train a model on Omniglot.
"""

import random
import numpy as np

import tensorflow as tf

from supervised_reptile.args import argument_parser, model_kwargs, train_kwargs, evaluate_kwargs
from supervised_reptile.eval import evaluate
from supervised_reptile.models import Cifar100Model
from supervised_reptile.cifar100 import read_dataset, split_dataset, augment_dataset
from supervised_reptile.train import train

DATA_DIR = None#'data/omniglot'

def main():
    """
    Load data and train a model on it.
    """
    args = argument_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(train_set)#augment_dataset(train_set))
    test_set = list(test_set)

    model = Cifar100Model(args.classes, **model_kwargs(args))

    with tf.Session() as sess:
        if not args.pretrained:
            print('Training...')
            train(sess, model, train_set, test_set, args.checkpoint, **train_kwargs(args))
        else:
            print('Restoring from checkpoint...')
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        print('Evaluating...')
        eval_kwargs = evaluate_kwargs(args)
        print('Train accuracy: ' + str(evaluate(sess, model, train_set, **eval_kwargs)))
        print('Test accuracy: ' + str(evaluate(sess, model, test_set, **eval_kwargs)))

if __name__ == '__main__':
    tf.disable_eager_execution()
    main()
