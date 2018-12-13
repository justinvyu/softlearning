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

DEFAULT_MAX_PATH_LENGTH = 1000
MAX_PATH_LENGTH_PER_DOMAIN = {
    'Point2DEnv': 50,
    'DClaw3': 200,
    'ImageDClaw3': 100,
    'HardwareDClaw3': 100,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 1,
        'n_initial_exploration_steps': int(1e3),
        'reparameterize': REPARAMETERIZE,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'lr': 3e-4,
        'discount': 0.99,
        'target_update_interval': 1,
        'tau': 0.005,
        'target_entropy': 'auto',
        'reward_scale': 1.0,
        'store_extra_policy_info': False,
        'action_prior': 'uniform',
        'save_full_state': False,
    }
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e2),
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Walker': int(3e3),
    'Ant': int(3e3),
    'Humanoid': int(1e4),
    'Pusher2d': int(1e3),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(200),
    'Reacher': int(200),
    'DClaw3': int(100),
    'ImageDClaw3': int(400),
    'HardwareDClaw3': int(100),
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': (
                    MAX_PATH_LENGTH_PER_DOMAIN.get(
                        domain, DEFAULT_MAX_PATH_LENGTH
                    ) * 10),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
    }
}


class NegativeLogLossFn(object):
    def __init__(self, eps):
        self._eps = eps

    def __call__(self, object_target_distance):
        return (
            - np.log(object_target_distance + self._eps)
            + np.log(np.pi + self._eps))

    def __str__(self):
        return str(f'eps={self._eps:e}')

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._eps == other._eps

        return super(NegativeLogLossFn, self).__eq__(other)


ENV_PARAMS = {
    'Swimmer': {  # 2 DoF
    },
    'Hopper': {  # 3 DoF
    },
    'HalfCheetah': {  # 6 DoF
    },
    'Walker': {  # 6 DoF
    },
    'Ant': {  # 8 DoF
        'CustomDefault': {
            'survive_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Humanoid': {  # 17 DoF
        'CustomDefault': {
            'survive_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Pusher2d': {  # 3 DoF
        'Default': {
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 1.0,
            'goal': (0, -1),
        },
        'DefaultReach': {
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'ImageDefault': {
            'image_shape': (32, 32, 3),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 3.0,
        },
        'ImageReach': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'BlindReach': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        }
    },
    'DClaw3': {
        'ScrewV2': {
            'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'pose_difference_cost_coeff': 0.0,
            'joint_velocity_cost_coeff': 0.0,
            'joint_acceleration_cost_coeff': tune.grid_search([0]),
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
        }
    },
    'ImageDClaw3': {
        'Screw': {
            'image_shape': (32, 32, 3),
            'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'pose_difference_cost_coeff': 0.0,
            'joint_velocity_cost_coeff': 0.0,
            'joint_acceleration_cost_coeff': tune.grid_search([0]),
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
        }
    },
    'HardwareDClaw3': {
        'ScrewV2': {
            'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'pose_difference_cost_coeff': 1e-1,
            'joint_velocity_cost_coeff': 1e-1,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-1.98, -1.98 + 2 * np.pi),
        },
        'ImageScrewV2': {
            'image_shape': (32, 32, 3),
            'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'pose_difference_cost_coeff': 1e-1,
            'joint_velocity_cost_coeff': 1e-1,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-1.98, -1.98 + 2 * np.pi),
        },
    },
    'Point2DEnv': {
        'Default': {
            'observation_keys': ('observation', ),
        },
        'Wall': {
            'observation_keys': ('observation', ),
        },
    }
}

NUM_CHECKPOINTS = 5


def get_variant_spec(universe, domain, task, policy):
    variant_spec = {
        'domain': domain,
        'task': task,
        'universe': universe,
        'git_sha': get_git_rev(),

        'env_params': ENV_PARAMS.get(domain, {}).get(task, {}),
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
        'algorithm_params': deep_update(
            ALGORITHM_PARAMS_BASE,
            ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
        ),
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': 5e5,
            }
        },
        'sampler_params': {
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': 1000,
                # MAX_PATH_LENGTH_PER_DOMAIN.get(
                #     domain, DEFAULT_MAX_PATH_LENGTH),
                'batch_size': 256,
            }
        },
        'run_params': {
            'seed': lambda spec: np.random.randint(0, 10000),
            'checkpoint_at_end': True,
            'checkpoint_frequency': lambda spec: (
                25000 // (spec.get('config', spec)
                          ['algorithm_params']
                          ['kwargs']
                          ['epoch_length'])
            ),
            # NUM_EPOCHS_PER_DOMAIN.get(
            #     domain, DEFAULT_NUM_EPOCHS) // NUM_CHECKPOINTS
        },
    }

    return variant_spec


def get_variant_spec_image(universe, domain, task, policy, *args, **kwargs):
    variant_spec = get_variant_spec(
        universe, domain, task, policy, *args, **kwargs)

    if 'image' in task.lower() or 'image' in domain.lower():
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'image_shape': variant_spec['env_params']['image_shape'],
                'output_size': M,
                'conv_filters': (4, 4),
                'conv_kernel_sizes': ((3, 3), (3, 3)),
                'pool_type': 'MaxPool2D',
                'pool_sizes': ((2, 2), (2, 2)),
                'pool_strides': (2, 2),
                'dense_hidden_layer_sizes': (),
            },
        }
        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, M)
        variant_spec['policy_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())

        variant_spec['Q_params']['kwargs']['preprocessor_params'] = (
            preprocessor_params.copy())
        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (M, M)

    return variant_spec
