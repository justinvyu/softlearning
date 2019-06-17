from ray import tune
import numpy as np

from softlearning.misc.utils import get_git_rev, deep_update

M = 256
REPARAMETERIZE = True

NUM_COUPLING_LAYERS = 2

GAUSSIAN_POLICY_PARAMS_BASE = {
    'type': 'GaussianPolicy',
    'kwargs': {
        'hidden_layer_sizes': (M, M),
        'squash': True,
    }
}

GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN = {}

POLICY_PARAMS_BASE = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_BASE,
}

POLICY_PARAMS_BASE.update({
    'gaussian': POLICY_PARAMS_BASE['GaussianPolicy'],
})

POLICY_PARAMS_FOR_DOMAIN = {
    'GaussianPolicy': GAUSSIAN_POLICY_PARAMS_FOR_DOMAIN,
}

POLICY_PARAMS_FOR_DOMAIN.update({
    'gaussian': POLICY_PARAMS_FOR_DOMAIN['GaussianPolicy'],
})

DEFAULT_MAX_PATH_LENGTH = 200
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'DClaw3': 200,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 3 ,
        'eval_deterministic': False,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}

#TODO Avi Most of the algorithm params for classifier-style methods
#are shared. Rewrite this part to reuse the params
ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'n_epochs': 200,
        }
    },
    'SACClassifier': {
        'type': 'SACClassifier',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 10000,
            'classifier_optim_name': 'adam',
            'reward_type': 'logits',
            'n_epochs': 200,
            'mixup_alpha': 0.0,
        }
    },
    'RAQ': {
        'type': 'RAQ',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            #'target_entropy': tune.grid_search([-10, -7, -5, -2, 0, 5]),
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 10,
            'classifier_optim_name': 'adam',
            'reward_type': 'logits',
            'active_query_frequency': 1,
            'n_epochs': 200,
            'mixup_alpha': 1.0,
            'image_only': True,
        }
    },
    'VICE': {
        'type': 'VICE',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 1, # tune.grid_search([1, 5, 10]),
            'classifier_optim_name': 'adam',
            'n_epochs': 500,
            'mixup_alpha': 1.0,
            'video_save_frequency': 1,
            'image_only': True,
        }
    },
    'VICEGAN': {
        'type': 'VICEGAN',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': tune.grid_search([10]),
            'classifier_optim_name': 'adam',
            'n_epochs': 500,
            'mixup_alpha': 0.0,
            'video_save_frequency': 1,
        }
    },
    'VICERAQ': {
        'type': 'VICERAQ',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'store_extra_policy_info': False,
            'action_prior': 'uniform',
            'classifier_lr': 1e-4,
            'classifier_batch_size': 128,
            'n_initial_exploration_steps': int(1e3),
            'n_classifier_train_steps': 10,
            'classifier_optim_name': 'adam',
            'active_query_frequency': 1,
            'n_epochs': 500,
            'mixup_alpha': 0.0,
            'video_save_frequency': 1,
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'td_target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                }[spec.get('config', spec)['domain']],
            ))
        }
    }
}

class NegativeLogLossFn(object):
    def __init__(self, eps, offset=0.0):
        self._eps = eps
        self._offset = offset

    def __call__(self, object_target_distance):
        return - np.log(object_target_distance + self._eps) + self._offset

    def __str__(self):
        return (
            f'NegativeLogLossFn(eps={self._eps:e},offset={self._offset:.3f})')

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._eps == other._eps and self._offset == other._offset

        return super(NegativeLogLossFn, self).__eq__(other)

ENVIRONMENT_PARAMS = {
    'DClaw3': {
        'ScrewV0-v0': {  # 6 DoF
            'isHARDARE': False,
        },
        'ScrewV2-v0': {
            'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (0, 0),
            'reset_free': False,
        },
        'ImageScrewV2-v0': {
            'is_hardware': False,
            'image_shape': (32, 32, 3),
            'reset_free': False,
            # 'goal_in_state': True,
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
        },
        'InfoScrewV2-v0': {
            # 'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'is_hardware': False,
            'image_shape': (32, 32, 3),
            'reset_free': False,
            # 'goal_in_state': True,
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (0, 0),
            # 'add_non_encoded_observations': True,
        }
    },
    'HardwareDClaw3': {
        'ScrewV2-v0': {
            'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (-np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
            'frame_skip': 30,
        },
        'ImageScrewV2-v0': {
            'image_shape': (32, 32, 3),
            # 'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'pose_difference_cost_coeff': 1e-1,
            'joint_velocity_cost_coeff': 1e-1,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (0, 0),
            'hw_w_sim_imgs': False,
            'save_eval_paths': True,
        },
    },
    'Pusher2d': {
        'ImageDefault-v0': {
            'image_shape': (32, 32, 3),
            'goal': (0, -1.35),
        }
    },
    'InvisibleArm': {
        'ImageFreeFloatManipulation': {
            'image_shape': (32, 32, 3),
            'viewer_params': {
                "azimuth": 90,
                "elevation": -32, # -27.7,
                "distance": 0.30,
                "lookat": np.array([-2.48756381e-18, -2.48756381e-18, 7.32824139e-01])
            },
            'rotation_only': True,
            'fixed_arm': True,
            'fixed_claw': False,
            'initial_x_range': (0., 0.),
            'initial_y_range': (0., 0.),
            'target_x_range': (0., 0.),
            'target_y_range': (0., 0.),
            'initial_z_rotation_range': (0., 0.),
            'target_z_rotation_range': (np.pi, np.pi),
        }
    }
}

DEFAULT_NUM_EPOCHS = 200
NUM_CHECKPOINTS = 10


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    # algorithm_params = deep_update(
    #     ALGORITHM_PARAMS_BASE,
    #     ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    # )
    # algorithm_params = deep_update(
    #     algorithm_params,
    #     ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
    # )
    algorithm_params = ALGORITHM_PARAMS_BASE
    algorithm_params = deep_update(
            algorithm_params,
            ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {})
        )

    variant_spec = {
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        'env_params': {
            'domain': domain,
            'task': task,
            'universe': universe,
            'kwargs': ENVIRONMENT_PARAMS.get(domain, {}).get(task, {}),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': 1e6,
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency':  DEFAULT_NUM_EPOCHS // NUM_CHECKPOINTS,
            'checkpoint_replay_pool': False,
        },
    }

    if task == 'InfoScrewV2-v0':
        variant_spec['replay_pool_params']['kwargs']['include_images'] = True
    return variant_spec


def get_variant_spec_classifier(universe,
                                domain,
                                task,
                                policy,
                                algorithm,
                                n_goal_examples,
                                *args,
                                **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    classifier_layer_size = L = 256
    variant_spec['classifier_params'] = {
        'type': 'feedforward_classifier',
        'kwargs': {
            'hidden_layer_sizes': (L,L),
            }
        }

    variant_spec['data_params'] = {
        'n_goal_examples': n_goal_examples,
        'n_goal_examples_validation_max': 100,
    }

    if algorithm in ['RAQ', 'VICERAQ']:

        if 'ScrewV2' in task:
            is_goal_key = 'is_goal'
        else:
            raise NotImplementedError('Success metric not defined for task')

        variant_spec.update({

            'sampler_params': {
                'type': 'ActiveSampler',
                'kwargs': {
                    'is_goal_key': is_goal_key,
                    'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH),
                    'min_pool_size': MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH),
                    'batch_size': 256,
                }
            },
            'replay_pool_params': {
                'type': 'ActiveReplayPool',
                'kwargs': {
                    'max_size': 1e6,
                }
            },

            })

    return variant_spec

def get_variant_spec(args):
    universe, domain = args.universe, args.domain
    task, algorithm = args.task, args.algorithm
    active_query_frequency = args.active_query_frequency
    n_epochs = args.n_epochs

    if args.algorithm in ['SACClassifier', 'RAQ', 'VICE', 'VICEGAN', 'VICERAQ']:
        variant_spec = get_variant_spec_classifier(
            universe, domain, task, args.policy, args.algorithm,
            args.n_goal_examples)
    else:
        variant_spec = get_variant_spec_base(
            universe, domain, task, args.policy, args.algorithm)

    if args.algorithm in ['RAQ', 'VICERAQ']:
        variant_spec['algorithm_params']['kwargs']['active_query_frequency'] = \
            active_query_frequency

    variant_spec['algorithm_params']['kwargs']['n_epochs'] = \
            n_epochs

    if 'Image' in task:
        # preprocessor_params = {
        #     'type': 'convnet_preprocessor',
        #     'kwargs': {
        #         'image_shape': variant_spec['env_params']['image_shape'],
        #         # 'image_shape': (48, 48, 3),
        #         'output_size': M,
        #         'conv_filters': (8, 8),
        #         'conv_kernel_sizes': ((5, 5), (5, 5)),
        #         'pool_type': 'MaxPool2D',
        #         'pool_sizes': ((2, 2), (2, 2)),
        #         'pool_strides': (2, 2),
        #         'dense_hidden_layer_sizes': (),
        #     },
        # }
        image_shape = variant_spec['env_params']['kwargs']['image_shape']
        preprocessor_params = tune.grid_search([
        {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': image_shape,
                'output_size': None,
                'conv_filters': (base_size, ) * num_layers,
                'conv_kernel_sizes': (conv_kernel_size, ) * num_layers,
                'conv_strides': (conv_strides, ) * num_layers,
                'normalization_type': normalization_type,
                'downsampling_type': downsampling_type,
                'use_global_average_pool': use_global_average_pool,
            },
        }
        for base_size in (64, )
        for conv_kernel_size in (3, )
        for conv_strides in (2, )
        for normalization_type in (None, )
        for num_layers in (3, )
        for use_global_average_pool in (False, )
        for downsampling_type in ('conv', )
        if (image_shape[0] / (conv_strides ** num_layers)) >= 1
    ])

        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        n_epochs = variant_spec['algorithm_params']['kwargs']['n_epochs']
        variant_spec['replay_pool_params']['kwargs']['max_size'] = int(n_epochs*1000)

        if args.algorithm in ['SACClassifier', 'RAQ', 'VICE', 'VICEGAN', 'VICERAQ']:
            variant_spec['classifier_params']['kwargs']['preprocessor_params'] = (
                preprocessor_params.copy())

    # elif 'Image' in task:
    #     raise NotImplementedError('Add convnet preprocessor for this image input')

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
