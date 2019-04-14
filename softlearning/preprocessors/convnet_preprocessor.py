import numpy as np
import tensorflow as tf
from gym import spaces

from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import regularizers

from softlearning.utils.keras import PicklableKerasModel
from .base_preprocessor import BasePreprocessor


def convnet(input_shape,
            output_size,
            conv_filters=(32, 64, 128),
            conv_kernel_sizes=(3, 3, 3),
            conv_strides=(2, 2, 2),
            use_global_average_pool=False,
            normalization_type=None,
            downsampling_type='pool',
            *args,
            **kwargs):
    assert downsampling_type in ('pool', 'conv'), downsampling_type

    img_input = layers.Input(shape=input_shape, dtype=tf.float32)
    x = img_input

    for (conv_filter, conv_kernel_size, conv_stride) in zip(
            conv_filters, conv_kernel_sizes, conv_strides):
        x = layers.Conv2D(
            filters=conv_filter,
            kernel_size=conv_kernel_size,
            strides=(conv_stride if downsampling_type == 'conv' else 1),
            padding="SAME",
            activation='linear',
            *args,
            **kwargs
        )(x)

        if normalization_type == 'batch':
            x = layers.BatchNormalization()(x)
        elif normalization_type == 'layer':
            raise NotImplementedError(normalization_type)
        elif normalization_type == 'weight':
            raise NotImplementedError(normalization_type)
        else:
            assert normalization_type is None, normalization_type

        x = layers.LeakyReLU()(x)

        if downsampling_type == 'pool' and conv_stride > 1:
            x = getattr(tf.keras.layers, 'AvgPool2D')(
                pool_size=conv_stride, strides=conv_stride
            )(x)

    if use_global_average_pool:
        x = layers.GlobalAveragePooling2D(name='average_pool')(x)
    else:
        x = tf.keras.layers.Flatten()(x)

    model = models.Model(img_input, x, name='convnet')
    model.summary()
    return model


def invert_convnet(convnet):
    input_shape = convnet.output.shape.as_list()[-1] // 2
    input_layer = tf.keras.layers.Input(shape=(input_shape, ))

    x = input_layer

    assert isinstance(convnet.layers[-1], tf.keras.layers.Dense)
    assert isinstance(
        convnet.layers[-2], tf.keras.layers.GlobalAveragePooling2D)

    x = layers.Reshape((1, 1, x.shape.as_list()[-1]))(x)
    x = layers.Conv2DTranspose(
        filters=convnet.layers[-2].input.shape.as_list()[-1],
        kernel_size=convnet.layers[-2].input.shape.as_list()[1:-1]
    )(x)

    for layer in convnet.layers[-3:0:-1]:
        if isinstance(layer, layers.Conv2D):
            assert x.shape.as_list() == layer.output.shape.as_list()
            layer_config = layer.get_config()
            layer_config['filters'] = layer.input.shape.as_list()[-1]
            x = layers.Conv2DTranspose.from_config(layer_config)(x)
            assert x.shape.as_list() == layer.input.shape.as_list()
        elif isinstance(layer, layers.LeakyReLU):
            x = type(layer).from_config(layer.get_config())(x)
        else:
            raise NotImplementedError(layer)

    model = models.Model(input_layer, x, name='inverted_convnet')
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
