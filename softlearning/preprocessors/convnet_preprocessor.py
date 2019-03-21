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


def convnet(input_shape,
            output_size,
            conv_filters=(32, 64, 128),
            conv_kernel_sizes=(3, 3, 3),
            conv_strides=(2, 2, 2),
            batch_norm_axis=None,
            *args,
            **kwargs):
    img_input = layers.Input(shape=input_shape, dtype=tf.float32)
    x = img_input

    for (conv_filter, conv_kernel_size, conv_stride) in zip(
            conv_filters, conv_kernel_sizes, conv_strides):
        x = layers.Conv2D(
            filters=conv_filter,
            kernel_size=conv_kernel_size,
            strides=conv_stride,
            padding="SAME",
            activation='linear',
            *args,
            **kwargs
        )(x)

        if batch_norm_axis is not None:
            x = layers.BatchNormalization(
                axis=batch_norm_axis,
                momentum=BATCH_NORM_DECAY,
                epsilon=BATCH_NORM_EPSILON
            )(x)

        x = layers.LeakyReLU()(x)

    x = layers.GlobalAveragePooling2D(name='average_pool')(x)
    x = layers.Dense(
         output_size,
         kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
         bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
         name='fully_connected')(x)

    model = models.Model(img_input, x, name='convnet')
    return model


def convnet_preprocessor(
        input_shapes,
        image_shape,
        output_size,
        name="convnet_preprocessor",
        make_picklable=True,
        *args,
        **kwargs):
    inputs = [
        layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    concatenated_input = layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(inputs)

    image_size = np.prod(image_shape)
    images_flat, input_raw = layers.Lambda(
        lambda x: [x[..., :image_size], x[..., image_size:]]
    )(concatenated_input)

    images = layers.Reshape(image_shape)(images_flat)
    preprocessed_images = convnet(
        input_shape=image_shape,
        output_size=output_size - input_raw.shape[-1],
        *args,
        **kwargs,
    )(images)
    output = layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )([preprocessed_images, input_raw])

    preprocessor = PicklableKerasModel(inputs, output, name=name)

    assert preprocessor.output.shape.as_list()[-1] == output_size

    return preprocessor


class ConvnetPreprocessor(BasePreprocessor):
    def __init__(self, observation_space, output_size, *args, **kwargs):
        super(ConvnetPreprocessor, self).__init__(
            observation_space, output_size)

        assert isinstance(observation_space, spaces.Box)
        input_shapes = (observation_space.shape, )

        self._convnet = convnet_preprocessor(
            input_shapes=input_shapes,
            output_size=output_size,
            *args,
            **kwargs)

    def transform(self, observation):
        transformed = self._convnet(observation)
        return transformed
