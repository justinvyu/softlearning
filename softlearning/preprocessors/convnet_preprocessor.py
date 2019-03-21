import tensorflow as tf
from gym import spaces

from softlearning.models.feedforward import feedforward_model
from softlearning.utils.keras import PicklableKerasModel

from .base_preprocessor import BasePreprocessor


def convnet_preprocessor(
        input_shapes,
        image_shape,
        output_size,
        conv_filters=(32, 32),
        conv_kernel_sizes=(3, 3),
        conv_strides=(2, 2),
        pool_type='MaxPool2D',
        pool_sizes=(0, 0),
        pool_strides=(1, 1),
        dense_hidden_layer_sizes=(64, 64),
        data_format='channels_last',
        name="convnet_preprocessor",
        use_batch_norm=False,
        use_layer_norm=False,
        make_picklable=True,
        *args,
        **kwargs):
    if data_format == 'channels_last':
        H, W, C = image_shape
    elif data_format == 'channels_first':
        C, H, W = image_shape

    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    concatenated_input = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(inputs)

    images_flat, input_raw = tf.keras.layers.Lambda(
        lambda x: [x[..., :H * W * C], x[..., H * W * C:]]
    )(concatenated_input)

    images = tf.keras.layers.Reshape(image_shape)(images_flat)

    assert not (use_batch_norm and use_layer_norm)

    out = images
    for (conv_filter, conv_kernel_size, conv_stride,
         pool_size, pool_stride) in zip(
             conv_filters, conv_kernel_sizes, conv_strides,
             pool_sizes, pool_strides):
        out = tf.keras.layers.Conv2D(
            filters=conv_filter,
            kernel_size=conv_kernel_size,
            strides=conv_stride,
            padding="SAME",
            activation='linear',
            *args,
            **kwargs
        )(out)

        if use_batch_norm:
            out = tf.keras.layers.BatchNormalization(axis=-1)(out)
        elif use_layer_norm:
            out = tf.keras.layers.BatchNormalization(axis=0)(out)

        out = tf.keras.layers.LeakyReLU()(out)

        if pool_size > 0:
            out = getattr(tf.keras.layers, pool_type)(
                pool_size=pool_size, strides=pool_stride
            )(out)

    flattened = tf.keras.layers.Flatten()(out)
    concatenated_output = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )([flattened, input_raw])

    output = (
        feedforward_model(
            input_shapes=(concatenated_output.shape[1:].as_list(), ),
            output_size=output_size,
            hidden_layer_sizes=dense_hidden_layer_sizes,
            activation='relu',
            output_activation='linear',
            *args,
            **kwargs
        )([concatenated_output])
        if dense_hidden_layer_sizes
        else concatenated_output)

    model = PicklableKerasModel(inputs, output, name=name)

    return model


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
