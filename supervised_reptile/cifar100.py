"""
Loading and augmenting the Omniglot dataset.

To use these APIs, you should prepare a directory that
contains all of the alphabets from both images_background
and images_evaluation.
"""

import os
import random

from PIL import Image
import numpy as np
import tensorflow_datasets as tfds

def read_dataset(data_dir):
    """
    Iterate over the characters in a data directory.

    Args:
      data_dir: a directory of alphabet directories.

    Returns:
      An iterable over Characters.

    The dataset is unaugmented and not split up into
    training and test sets.
    """

    train_dataset = list(tfds.as_numpy(tfds.load('cifar100', split='train')))
    test_dataset = list(tfds.as_numpy(tfds.load('cifar100', split='test')))
    dataset = train_dataset + test_dataset

    class_images = [[] for _ in range(100)]

    for obj in dataset:
      class_images[int(obj['label'])].append(obj['image'])

    for cur_images in class_images:
      yield Character(np.asarray(cur_images))

def split_dataset(dataset, num_train=80):
    """
    Split the dataset into a training and test set.

    Args:
      dataset: an iterable of Characters.

    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]

def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.

    Args:
      dataset: an iterable of Characters.

    Returns:
      An iterable of augmented Characters.
    """
    yield from dataset
    #for character in dataset:
    #    for rotation in [0, 90, 180, 270]:
    #        yield Character(character.dir_path, rotation=rotation)

# pylint: disable=R0903
class Character:
    """
    A single character class.
    """
    def __init__(self, images):
        self.images = images

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        return self.images[np.random.choice(len(self.images), size=num_images,
            replace=False)].astype(float)/255
