from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from nets import mobilenet_v1
from nets import mobilenet_v1_ml
from nets import resnet_v2
from nets import resnet_v2_ml


slim = tf.contrib.slim

networks_map = {'mobilenet_v1': mobilenet_v1.mobilenet_v1,
                'mobilenet_v1_ml': mobilenet_v1_ml.mobilenet_v1,
                'resnet_v2_50': resnet_v2.resnet_v2_50,
                'resnet_v2_50_ml': resnet_v2_ml.resnet_v2_50,
                }

arg_scopes_map = {'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_ml': mobilenet_v1_ml.mobilenet_v1_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_50_ml': resnet_v2_ml.resnet_arg_scope,
                  }

def get_network_fn(name, num_classes, weight_decay=0.0, is_training=False):
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images, scope=None):
        arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, scope=scope)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
