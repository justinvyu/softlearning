from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from gym import spaces

from softlearning.utils.keras import PicklableKerasModel
from .base_preprocessor import BasePreprocessor


def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))


def sampling(inputs):
    z_mean, z_log_var = inputs
    batch_size = tf.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.random_normal(shape=(batch_size, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def make_encoder(input_shape, latent_size, base_depth, beta=1.0):
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation='relu')

    input_layer = tf.keras.layers.Input(
        shape=input_shape, name='encoder_input')
    out = input_layer

    out = conv(filters=base_depth, kernel_size=5, strides=1)(out)
    out = conv(filters=base_depth, kernel_size=5, strides=2)(out)
    out = conv(filters=2 * base_depth, kernel_size=5, strides=1)(out)
    out = conv(filters=2 * base_depth, kernel_size=5, strides=2)(out)
    out = conv(filters=4 * latent_size, kernel_size=8, padding="VALID")(out)

    out = tf.keras.layers.Flatten()(out)
    shift_and_log_scale_diag = tf.keras.layers.Dense(
        2 * latent_size, activation=None
    )(out)

    shift, log_scale_diag = tf.keras.layers.Lambda(
        lambda shift_and_log_scale_diag: tf.split(
            shift_and_log_scale_diag,
            num_or_size_splits=2,
            axis=-1)
    )(shift_and_log_scale_diag)

    latents = tf.keras.layers.Lambda(
        sampling, output_shape=(latent_size,), name='z'
    )([shift, log_scale_diag])

    encoder = tf.keras.Model(
        input_layer, [shift, log_scale_diag, latents], name='encoder')

    return encoder


def make_decoder(latent_size, output_shape, base_depth):
    deconv = functools.partial(
        tf.keras.layers.Conv2DTranspose, padding="SAME", activation='relu')
    conv = functools.partial(
        tf.keras.layers.Conv2D, padding="SAME", activation='relu')

    input_layer = tf.keras.layers.Input(
        shape=(latent_size, ), name='decoder_input')
    # Collapse the sample and batch dimension and convert to rank-4 tensor for
    # use with a convolutional decoder network.
    codes = tf.keras.layers.Reshape((1, 1, latent_size))(input_layer)
    out = codes

    out = deconv(filters=2 * base_depth, kernel_size=8, padding="VALID")(out)
    out = deconv(filters=2 * base_depth, kernel_size=5, strides=1)(out)
    out = deconv(filters=2 * base_depth, kernel_size=5, strides=2)(out)
    out = deconv(filters=base_depth, kernel_size=5, strides=1)(out)
    out = deconv(filters=base_depth, kernel_size=5, strides=2)(out)
    out = deconv(filters=base_depth, kernel_size=5, strides=1)(out)

    out = conv(filters=output_shape[-1], kernel_size=5, activation=None)(out)

    decoder = tf.keras.Model(input_layer, out, name='decoder')

    return decoder


def make_latent_prior(latent_size):
    prior = tfp.distributions.Independent(
        tfp.distributions.Normal(loc=tf.zeros(latent_size), scale=1),
        reinterpreted_batch_ndims=1)
    return prior


def create_beta_vae(image_shape, output_size, base_depth, beta=1.0):
    encoder = make_encoder(image_shape, output_size, base_depth, beta=beta)
    decoder = make_decoder(output_size, image_shape, base_depth)

    outputs = decoder(encoder(encoder.inputs)[2])
    vae = tf.keras.Model(encoder.inputs, outputs, name='vae')

    return vae


def create_vae_preprocessor(
        input_shapes,
        image_shape,
        output_size,
        vae,
        name="beta_vae_preprocessor"):

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

    encoded_images = vae.get_layer('encoder')(images)[0]

    output = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )([encoded_images, input_raw])

    model = PicklableKerasModel(inputs, output, name=name)

    return model


class VAEPreprocessor(BasePreprocessor):
    def __init__(self,
                 observation_space,
                 output_size,
                 image_shape,
                 *args,
                 **kwargs):
        super(VAEPreprocessor, self).__init__(observation_space, output_size)
        self.image_shape = image_shape

        assert isinstance(observation_space, spaces.Box)
        input_shapes = (observation_space.shape, )

        self.vae = create_beta_vae(
            image_shape,
            output_size,
            base_depth=32,
            beta=1.0)
        self.preprocessor = create_vae_preprocessor(
            input_shapes,
            image_shape,
            output_size,
            self.vae,
        )

    def transform(self, observation):
        transformed = self.preprocessor(observation)
        return transformed

    @property
    def trainable_variables(self):
        return self.vae.trainable_variables
