from copy import deepcopy

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
        'observation_keys': None,
        'observation_preprocessors_params': {
            'observations': None,
        }
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
    'DClaw3': 250,
    'HardwareDClaw3': 100,
    'Pendulum': 200,
    'Pusher2d': 200,
    'InvisibleArm': 250,
}

ALGORITHM_PARAMS_BASE = {
    'type': 'SAC',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': tune.grid_search([1]),
        'eval_render_kwargs': {},
        'eval_n_episodes': 1, # num of eval rollouts
        'eval_deterministic': False,
        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,
    }
}


ALGORITHM_PARAMS_ADDITIONAL = {
    'SAC': {
        'type': 'SAC',
        'kwargs': {
            'reparameterize': REPARAMETERIZE,
            'lr': 3e-4,
            'target_update_interval': 1,
            'tau': 5e-3,
            'target_entropy': 'auto',
            'action_prior': 'uniform',
            'n_initial_exploration_steps': int(1e3),
            'her_iters': tune.grid_search([0]),
        }
    },
    'SQL': {
        'type': 'SQL',
        'kwargs': {
            'policy_lr': 3e-4,
            'target_update_interval': 1,
            'n_initial_exploration_steps': int(1e3),
            'reward_scale': tune.sample_from(lambda spec: (
                {
                    'Swimmer': 30,
                    'Hopper': 30,
                    'HalfCheetah': 30,
                    'Walker2d': 10,
                    'Ant': 300,
                    'Humanoid': 100,
                    'Pendulum': 1,
                }.get(
                    spec.get('config', spec)
                    ['environment_params']
                    ['training']
                    ['domain'],
                    1.0
                ),
            )),
        }
    }
}

DEFAULT_NUM_EPOCHS = 200

NUM_EPOCHS_PER_DOMAIN = {
    'Swimmer': int(3e2),
    'Hopper': int(1e3),
    'HalfCheetah': int(3e3),
    'Walker2d': int(3e3),
    'Ant': int(3e3),
    'Humanoid': int(1e4),
    'Pusher2d': int(3e2),
    'HandManipulatePen': int(1e4),
    'HandManipulateEgg': int(1e4),
    'HandManipulateBlock': int(1e4),
    'HandReach': int(1e4),
    'Point2DEnv': int(200),
    'Reacher': int(200),
    'DClaw3': int(200),
    'HardwareDClaw3': int(100),
    'InvisibleArm': int(1e3),
    'Pendulum': 10,
    'Sawyer': int(1e4),
    'ball_in_cup': int(2e4),
    'cheetah': int(2e4),
    'finger': int(2e4),
}

ALGORITHM_PARAMS_PER_DOMAIN = {
    **{
        domain: {
            'kwargs': {
                'n_epochs': NUM_EPOCHS_PER_DOMAIN.get(
                    domain, DEFAULT_NUM_EPOCHS),
                'n_initial_exploration_steps': tune.sample_from(lambda spec: (
                    10 * spec.get('config', spec)
                    ['sampler_params']
                    ['kwargs']
                    ['max_path_length']
                )),
            }
        } for domain in NUM_EPOCHS_PER_DOMAIN
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


class LinearLossFn(object):
    def __call__(self, object_target_distance):
        return -object_target_distance

    def __str__(self):
        return str(f'LinearLossFn()')

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self._eps == other._eps

        return super(LinearLossFn, self).__eq__(other)


ENVIRONMENT_PARAMS = {
    'Swimmer': {  # 2 DoF
    },
    'Hopper': {  # 3 DoF
    },
    'HalfCheetah': {  # 6 DoF
    },
    'Walker2d': {  # 6 DoF
    },
    'Ant': {  # 8 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Humanoid': {  # 17 DoF
        'Parameterizable-v3': {
            'healthy_reward': 0.0,
            'healthy_z_range': (-np.inf, np.inf),
            'exclude_current_positions_from_observation': False,
        }
    },
    'Pusher2d': {  # 3 DoF
        'Default-v0': {
            'eef_to_object_distance_cost_coeff': 1.0,
            'goal_to_object_distance_cost_coeff': 1.0,
            'ctrl_cost_coeff': 0.0,
            'goal': (0, -1),
            'puck_initial_x_range': (0, 1),
            'puck_initial_y_range': (-1, -0.5),
            'goal_x_range': (-1, 0),
            'goal_y_range': (-1, 1),
            'num_goals': 2,
            'swap_goal_upon_completion': True,
            'reset_free': True,
            # 'pixel_wrapper_kwargs': {
            #     # 'observation_key': 'pixels',
            #     # 'pixels_only': True,
            #     'render_kwargs': {
            #         'width': 32,
            #         'height': 32,
            #         'camera_id': -1,
            #     },
            # },
        },
        'DefaultReach-v0': {
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'ImageDefault-v0': {
            'image_shape': (32, 32, 3),
            'arm_object_distance_cost_coeff': 0.0,
            'goal_object_distance_cost_coeff': 3.0,
        },
        'ImageReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        },
        'BlindReach-v0': {
            'image_shape': (32, 32, 3),
            'arm_goal_distance_cost_coeff': 1.0,
            'arm_object_distance_cost_coeff': 0.0,
        }
    },
    'Point2DEnv': {
        'Default-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
        'Wall-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
    },
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
            'target_initial_position_range': (-np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (-np.pi, np.pi),
            'reset_free': True,
        },
        'ImageScrewV2-v0': {
            'object_target_distance_reward_fn': NegativeLogLossFn(1e-6),
            'is_hardware': False,
            'image_shape': (32, 32, 3),
            'reset_free': False,
            # 'goal_in_state': True,
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (-np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (0, 0),
            'state_reward': True,
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
            'object_target_distance_reward_fn': NegativeLogLossFn(0),
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
            'object_target_distance_reward_fn': NegativeLogLossFn(0),
            'pose_difference_cost_coeff': 0,
            'joint_velocity_cost_coeff': 0,
            'joint_acceleration_cost_coeff': 0,
            'target_initial_velocity_range': (0, 0),
            'target_initial_position_range': (np.pi, np.pi),
            'object_initial_velocity_range': (0, 0),
            'object_initial_position_range': (0, 0),
            'hw_w_sim_imgs': False,
            'save_eval_paths': True,
        },
    },
    'Point2DEnv': {
        'Default-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
        'Wall-v0': {
            'observation_keys': ('observation', 'desired_goal'),
        },
    },
    'Sawyer': {
        task_name: {
            'has_renderer': False,
            'has_offscreen_renderer': False,
            'use_camera_obs': False,
            'reward_shaping': tune.grid_search([True, False]),
        }
        for task_name in (
                'Lift',
                'NutAssembly',
                'NutAssemblyRound',
                'NutAssemblySingle',
                'NutAssemblySquare',
                'PickPlace',
                'PickPlaceBread',
                'PickPlaceCan',
                'PickPlaceCereal',
                'PickPlaceMilk',
                'PickPlaceSingle',
                'Stack',
        )
    },
    'InvisibleArm': {
        'FreeFloatManipulation': {
            'has_renderer': False,
            'has_offscreen_renderer': True,
            'use_camera_obs': False,
            'camera_name': 'agentview',
            'use_object_obs': True,
            'position_reward_weight': 10.0,
            'orientation_reward_weight': 1.0,
            'control_freq': 10,
            'objects_type': 'screw',
            'observation_keys': (
                # 'joint_pos',
                # 'joint_vel',
                'gripper_qpos',
                # 'gripper_qvel',
                'eef_pos',
                'eef_quat',
                # 'robot-state',
                # 'custom-cube_position',
                # 'custom-cube_quaternion',
                # 'custom-cube_to_eef_pos',
                # 'custom-cube_to_eef_quat',
                # 'custom-cube-visual_position',
                # 'custom-cube-visual_quaternion',
                'screw_position',
                'screw_quaternion',
                # 'screw_to_eef_pos',
                # 'screw_to_eef_quat',
                'screw-visual_position',
                'screw-visual_quaternion',
            ),
            'target_x_range': [-0.1, 0.1],
            'target_y_range': [-0.1, 0.1],
            'target_z_rotation_range': [-np.pi, np.pi],
            'num_goals': tune.grid_search([1,2,4,8])
        }
    },
    'ball_in_cup': {
        'catch': {
            'pixel_wrapper_kwargs': {
                'observation_key': 'pixels',
                'pixels_only': True,
                'render_kwargs': {
                    'width': 84,
                    'height': 84,
                    'camera_id': 0,
                },
            },
        },
    },
    'cheetah': {
        'run': {
            'pixel_wrapper_kwargs': {
                'observation_key': 'pixels',
                'pixels_only': True,
                'render_kwargs': {
                    'width': 84,
                    'height': 84,
                    'camera_id': 0,
                },
            },
        },
    },
    'finger': {
        'spin': {
            'pixel_wrapper_kwargs': {
                'observation_key': 'pixels',
                'pixels_only': True,
                'render_kwargs': {
                    'width': 84,
                    'height': 84,
                    'camera_id': 0,
                },
            },
        },
    },

}

NUM_CHECKPOINTS = 10
SAMPLER_PARAMS_PER_DOMAIN = {
    'DClaw3': {
        'type': 'SimpleSampler',
    },
    'HardwareDClaw3': {
        'type': 'RemoteSampler',
    }
}


def get_variant_spec_base(universe, domain, task, policy, algorithm):
    algorithm_params = deep_update(
        ALGORITHM_PARAMS_BASE,
        ALGORITHM_PARAMS_ADDITIONAL.get(algorithm, {}),
        ALGORITHM_PARAMS_PER_DOMAIN.get(domain, {})
    )
    if task == 'InfoScrewV2-v0':
        from softlearning.misc.utils import PROJECT_PATH
        algorithm_params['kwargs']['goal_classifier_params_direc'] = os.path.join(
            PROJECT_PATH,
            'goal_classifier/screw_imgs/train_scope/params.ckpt')
    variant_spec = {
        'git_sha': get_git_rev(__file__),

        'environment_params': {
            'training': {
                'domain': domain,
                'task': task,
                'universe': universe,
                'kwargs': (
                    ENVIRONMENT_PARAMS.get(domain, {}).get(task, {})),
            },
            'evaluation': tune.sample_from(lambda spec: (
                spec.get('config', spec)
                ['environment_params']
                ['training']
            )),
        },
        'policy_params': deep_update(
            POLICY_PARAMS_BASE[policy],
            POLICY_PARAMS_FOR_DOMAIN[policy].get(domain, {})
        ),
        'exploration_policy_params': {
            'type': 'UniformPolicy',
            'kwargs': {
                'observation_keys': tune.sample_from(lambda spec: (
                    spec.get('config', spec)
                    ['policy_params']
                    ['kwargs']
                    .get('observation_keys')
                ))
            },
        },
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'observation_preprocessors_params': {
                    'observations': None,
                }
            }
        },
        'algorithm_params': algorithm_params,
        'replay_pool_params': {
            'type': 'SimpleReplayPool',
            'kwargs': {
                'max_size': tune.sample_from(lambda spec: (
                    {
                        'SimpleReplayPool': int(5e5),
                        'TrajectoryReplayPool': int(1e4),
                    }.get(
                        spec.get('config', spec)
                        ['replay_pool_params']
                        ['type'],
                        int(1e6))
                )),
            }
        },
        'sampler_params': deep_update({
            'type': 'SimpleSampler',
            'kwargs': {
                'max_path_length': MAX_PATH_LENGTH_PER_DOMAIN.get(
                    domain, DEFAULT_MAX_PATH_LENGTH),
                'min_pool_size': 1000,
                'batch_size': 256,
            }
        }, SAMPLER_PARAMS_PER_DOMAIN.get(domain, {})),
        'run_params': {
            'seed': tune.sample_from(
                lambda spec: np.random.randint(0, 10000)),
            'checkpoint_at_end': True,
            'checkpoint_frequency': tune.sample_from(lambda spec: (
                25000 // (spec.get('config', spec)
                          ['algorithm_params']
                          ['kwargs']
                          ['epoch_length'])
            )),
        },
    }

    if task == 'InfoScrewV2-v0':
        variant_spec['replay_pool_params']['kwargs']['include_images'] = True
    if task == 'ImageScrewV2-v0' and ENVIRONMENT_PARAMS['DClaw3']['ImageScrewV2-v0']['state_reward']:
        variant_spec['replay_pool_params']['kwargs']['super_observation_space_shape'] = (9+9+2+1+2,)
    if domain == 'HardwareDClaw3':
        variant_spec['sampler_params']['type'] == 'RemoteSampler'
        variant_spec['algorithm_params']['kwargs']['max_train_repeat_per_timestep'] = 1
    return variant_spec


def is_image_env(domain, task, variant_spec):
    return ('image' in task.lower()
            or 'image' in domain.lower()
            or 'pixel_wrapper_kwargs' in (
                variant_spec['environment_params']['training']['kwargs']))


def get_variant_spec_image(universe,
                           domain,
                           task,
                           policy,
                           algorithm,
                           *args,
                           **kwargs):
    variant_spec = get_variant_spec_base(
        universe, domain, task, policy, algorithm, *args, **kwargs)

    if is_image_env(domain, task, variant_spec):
        preprocessor_params = {
            'type': 'convnet_preprocessor',
            'kwargs': {
                'conv_filters': (64, ) * 3,
                'conv_kernel_sizes': (3, ) * 3,
                'conv_strides': (2, ) * 3,
                'normalization_type': 'layer',
                'downsampling_type': 'conv',
            },
        }

        variant_spec['policy_params']['kwargs']['hidden_layer_sizes'] = (M, M)
        variant_spec['policy_params']['kwargs']['observation_preprocessors_params'] = {
            'pixels': deepcopy(preprocessor_params)
        }

        # for key in ('hidden_layer_sizes', 'observation_preprocessors_params'):
        variant_spec['Q_params']['kwargs']['hidden_layer_sizes'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)['policy_params']['kwargs']['hidden_layer_sizes']
            )))
        )
        variant_spec['Q_params']['kwargs']['observation_preprocessors_params'] = (
            tune.sample_from(lambda spec: (deepcopy(
                spec.get('config', spec)['policy_params']['kwargs']['observation_preprocessors_params']
            )))
        )

    return variant_spec


def get_variant_spec(args):
    universe, domain, task = args.universe, args.domain, args.task

    variant_spec = get_variant_spec_image(
        universe, domain, task, args.policy, args.algorithm)

    if args.checkpoint_replay_pool is not None:
        variant_spec['run_params']['checkpoint_replay_pool'] = (
            args.checkpoint_replay_pool)

    return variant_spec
