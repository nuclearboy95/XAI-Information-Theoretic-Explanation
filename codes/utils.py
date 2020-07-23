import json
from contextlib import contextmanager
from itertools import count, product
from tqdm import tqdm
import numpy as np
import _pickle as p
import tensorflow as tf
import functools

__all__ = ['task', 'load_json', 'get_imagenet_classname', 'runner', 'save_binary', 'ranges', 'pad_images',
           'preprocess_imagenet', 'lazy_property']


def lazy_property(f):
    attribute = '_cache_' + f.__name__

    @property
    @functools.wraps(f)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, f(self))
        return getattr(self, attribute)

    return decorator


def pad_images(images, K):
    pad_width = ((0, 0), (K, K), (K, K), (0, 0))
    return np.pad(images, pad_width, mode='constant')


def ranges(*args):
    generators = [range(arg) for arg in args]
    return product(*generators)


def load_json(fpath, encoding=None):
    with open(fpath, 'r', encoding=encoding) as f:
        return json.load(f)


def save_binary(d, fpath):
    with open(fpath, 'wb') as f:
        p.dump(d, f)


def get_imagenet_classname(class_id):
    d1 = load_json('data/class_index2uid.json')
    uid = d1[str(class_id)]
    d2 = load_json('data/uid2human.json')
    name = d2[uid]
    return name


@contextmanager
def task(_):
    yield


def runner(sess, ops, verbose=True):
    i_batch_g = count()

    if verbose:
        i_batch_g = tqdm(i_batch_g)

    for i_batch in i_batch_g:
        try:
            result_batch = sess.run(ops)
            yield (i_batch, result_batch)

        except tf.errors.OutOfRangeError:
            return


MEAN_IMAGENET = [0.485, 0.456, 0.406]
STD_IMAGENET = [0.229, 0.224, 0.225]


def preprocess_imagenet(x):
    x = tf.cast(x, tf.float32)
    x /= 255.
    x -= tf.constant(MEAN_IMAGENET)
    x /= tf.constant(STD_IMAGENET)
    return x
