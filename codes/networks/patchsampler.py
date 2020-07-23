import tensorflow as tf
import numpy as np
import os
from ..utils import preprocess_imagenet, lazy_property


class PatchSampler:
    @property
    def name(self):
        return 'PatchSampler_K%d_C%d' % (self.k, self.c)

    @lazy_property
    def mask(self):
        mask_np = np.ones((1, 3 * self.k, 3 * self.k, self.c), dtype=np.float32)
        mask_np[:, self.k: 2 * self.k, self.k: 2 * self.k, :] = 0
        return tf.constant(mask_np, dtype=tf.float32, name='mask')

    @property
    def filter_size(self):
        if self.k == 1:
            return 3
        else:
            return min(self.k + 1, 9)

    @property
    def filter_num(self):
        return min(16 * self.k * self.c, 256)

    @property
    def conv_num(self):
        from math import log
        return int(log(self.k, 2))

    def __init__(self, x, y, k, c):
        self.k = k
        self.c = c
        self.y = y

        x = preprocess_imagenet(x)
        h = x * self.mask

        with tf.variable_scope('PatchSampler'):
            with tf.variable_scope('layer1'):
                h = tf.contrib.layers.conv2d(h, self.filter_num, self.filter_size,
                                             activation_fn=tf.nn.leaky_relu)
                h = tf.contrib.layers.conv2d(h, self.filter_num, self.filter_size,
                                             activation_fn=tf.nn.leaky_relu)
                h = tf.contrib.layers.avg_pool2d(h, (2, 2))

            with tf.variable_scope('layer2'):
                if self.k != 1:
                    if self.k >= 4:
                        crop_bottom = crop_top = self.k // 4
                    else:
                        crop_top = 0
                        crop_bottom = 1

                    h = h[:, crop_top: -crop_bottom, crop_top: -crop_bottom]

            with tf.variable_scope('layer3'):
                h = tf.contrib.layers.conv2d(h, self.filter_num, self.k,
                                             activation_fn=tf.nn.leaky_relu)
                h = tf.contrib.layers.conv2d(h, self.filter_num, self.k,
                                             activation_fn=tf.nn.leaky_relu)

                for _ in range(self.conv_num):
                    h = tf.contrib.layers.conv2d(h, self.filter_num, self.k,
                                                 activation_fn=tf.nn.leaky_relu)

            with tf.variable_scope('output'):
                logits = tf.contrib.layers.conv2d(h, 256 * self.c, self.k, activation_fn=None)
                self.logits = tf.reshape(logits, [-1, self.k, self.k, self.c, 256])
                self.probs = tf.nn.softmax(self.logits)

    @lazy_property
    def path_ckpt(self):
        name = 'PatchSampler_K{K}_C{C}'.format(K=self.k, C=self.c)
        return 'ckpts/{}/{}'.format(name, name)

    @lazy_property
    def saver(self):
        return tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                         scope='PatchSampler'))

    def save(self, sess, epoch=None):
        os.makedirs(os.path.dirname(self.path_ckpt), exist_ok=True)
        self.saver.save(sess, self.path_ckpt, global_step=epoch)

    def load(self, sess):
        try:
            self.saver.restore(sess, self.path_ckpt)
        except ValueError as e:
            print('[Error] Failed to load checkpoint: %s' % self.name)
