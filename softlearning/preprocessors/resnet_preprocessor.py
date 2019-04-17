"""ResNet50 model for Keras.
Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.
Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from gym import spaces

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

from softlearning.utils.keras import PicklableKerasModel
from .base_preprocessor import BasePreprocessor


L2_WEIGHT_DECAY = 0
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   batch_norm_axis=None):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2a')(input_tensor)

    if batch_norm_axis is not None:
        x = layers.BatchNormalization(axis=batch_norm_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2a')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2b')(x)
    if batch_norm_axis is not None:
        x = layers.BatchNormalization(axis=batch_norm_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2b')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2c')(x)
    if batch_norm_axis is not None:
        x = layers.BatchNormalization(axis=batch_norm_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.LeakyReLU()(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               batch_norm_axis=None):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the second conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2a')(input_tensor)
    if batch_norm_axis is not None:
        x = layers.BatchNormalization(axis=batch_norm_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2a')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2b')(x)
    if batch_norm_axis is not None:
        x = layers.BatchNormalization(axis=batch_norm_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2b')(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2c')(x)
    if batch_norm_axis is not None:
        x = layers.BatchNormalization(axis=batch_norm_axis,
                                      momentum=BATCH_NORM_DECAY,
                                      epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                             name=conv_name_base + '1')(input_tensor)
    if batch_norm_axis is not None:
        shortcut = layers.BatchNormalization(axis=batch_norm_axis,
                                             momentum=BATCH_NORM_DECAY,
                                             epsilon=BATCH_NORM_EPSILON,
                                             name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.LeakyReLU()(x)
    return x


def resnet6(input_shape,
            output_size,
            batch_norm_axis=None):
    img_input = layers.Input(shape=input_shape, dtype=tf.float32)
    x = img_input

    x = conv_block(
        x, 3, [16, 16, 64], stage=2, block='a',
        batch_norm_axis=batch_norm_axis)
    x = identity_block(
        x, 3, [16, 16, 64], stage=2, block='b',
        batch_norm_axis=batch_norm_axis)
    x = conv_block(
        x, 3, [32, 32, 128], stage=3, block='a',
        batch_norm_axis=batch_norm_axis)
    x = identity_block(
        x, 3, [32, 32, 128], stage=3, block='b',
        batch_norm_axis=batch_norm_axis)

    x = layers.GlobalAveragePooling2D(name='average_pool')(x)
    x = layers.Dense(
         output_size,
         kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
         bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
         name='fully_connected')(x)

    model = models.Model(img_input, x, name='resnet6')
    return model


def resnet6_preprocessor(
        input_shapes,
        image_shape,
        output_size,
        *args,
        name="resnet6_preprocessor",
        **kwargs):
    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    concatenated_input = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(inputs)

    image_size = np.prod(image_shape)
    images_flat, input_raw = tf.keras.layers.Lambda(
        lambda x: [x[..., :image_size], x[..., image_size:]]
    )(concatenated_input)

    images = tf.keras.layers.Reshape(image_shape)(images_flat)
    preprocessed_images = resnet6(
        input_shape=image_shape,
        output_size=output_size - input_raw.shape[-1],
        *args,
        **kwargs,
    )(images)
    output = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )([preprocessed_images, input_raw])

    preprocessor = PicklableKerasModel(inputs, output, name=name)

    assert preprocessor.output.shape.as_list()[-1] == output_size

    return preprocessor


class Resnet6Preprocessor(BasePreprocessor):
    def __init__(self, observation_space, output_size, *args, **kwargs):
        super(Resnet6Preprocessor, self).__init__(
            observation_space, output_size)

        assert isinstance(observation_space, spaces.Box)
        input_shapes = (observation_space.shape, )

        self._resnet6 = resnet6_preprocessor(
            input_shapes=input_shapes,
            output_size=output_size,
            *args,
            **kwargs)

    def transform(self, observation):
        transformed = self._resnet6(observation)
        return transformed
