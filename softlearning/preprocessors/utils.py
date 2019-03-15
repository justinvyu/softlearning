from copy import deepcopy


def get_convnet_preprocessor(observation_space,
                             name='convnet_preprocessor',
                             **kwargs):
    from .convnet_preprocessor import ConvnetPreprocessor

    num_conv_layers = kwargs.pop('num_conv_layers')
    num_filters_per_layer = kwargs.pop('num_filters_per_layer')
    pool_size = kwargs.pop('pool_size')

    kwargs.update({
        'conv_filters': (num_filters_per_layer, ) * num_conv_layers,
        'conv_kernel_sizes': ((3, 3), ) * num_conv_layers,
        'pool_sizes': ((pool_size, pool_size), ) * num_conv_layers,
        'pool_strides': (pool_size, ) * num_conv_layers,
    })

    preprocessor = ConvnetPreprocessor(
        observation_space=observation_space, name=name, **kwargs)

    return preprocessor


def get_feedforward_preprocessor(observation_space,
                                 name='feedforward_preprocessor',
                                 **kwargs):
    from .feedforward_preprocessor import FeedforwardPreprocessor
    preprocessor = FeedforwardPreprocessor(
        observation_space=observation_space, name=name, **kwargs)

    return preprocessor


def get_vae_preprocessor(observation_space,
                         name='vae_preprocessor',
                         **kwargs):
    from .vae_preprocessor import VAEPreprocessor
    preprocessor = VAEPreprocessor(
        observation_space=observation_space,
        name=name,
        **kwargs)

    return preprocessor


PREPROCESSOR_FUNCTIONS = {
    'ConvnetPreprocessor': get_convnet_preprocessor,
    'FeedforwardPreprocessor': get_feedforward_preprocessor,
    'VAEPreprocessor': get_vae_preprocessor,
    None: lambda *args, **kwargs: None
}


def get_preprocessor_from_params(env, preprocessor_params, *args, **kwargs):
    if preprocessor_params is None:
        return None

    preprocessor_type = preprocessor_params.get('type', None)
    preprocessor_kwargs = deepcopy(preprocessor_params.get('kwargs', {}))

    if preprocessor_type is None:
        return None

    preprocessor = PREPROCESSOR_FUNCTIONS[
        preprocessor_type](
            env.observation_space,
            *args,
            **preprocessor_kwargs,
            **kwargs)

    return preprocessor


def get_preprocessor_from_variant(variant, env, *args, **kwargs):
    preprocessor_params = variant['preprocessor_params']
    return get_preprocessor_from_params(
        env, preprocessor_params, *args, **kwargs)
