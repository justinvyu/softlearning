from collections import defaultdict

import numpy as np
from gym.spaces import Box, Dict, Discrete

from .flexible_replay_pool import FlexibleReplayPool


def normalize_observation_fields(observation_space, name='observations'):
    if isinstance(observation_space, Dict):
        fields = [
            normalize_observation_fields(child_observation_space, name)
            for name, child_observation_space
            in observation_space.spaces.items()
        ]
        fields = {
            'observations.{}'.format(name): value
            for field in fields
            for name, value in field.items()
        }
    elif isinstance(observation_space, (Box, Discrete)):
        fields = {
            name: {
                'shape': observation_space.shape,
                'dtype': observation_space.dtype,
            }
        }
    else:
        raise NotImplementedError(
            "Observation space of type '{}' not supported."
            "".format(type(observation_space)))

    return fields


class SimpleReplayPool(FlexibleReplayPool):
    def __init__(self, observation_space, action_space, include_images=False,
        super_observation_space_shape=None, *args, **kwargs):
        self._observation_space = observation_space
        self._action_space = action_space
        self._include_images = include_images
        self._super_observation_space_shape = super_observation_space_shape

        observation_fields = normalize_observation_fields(observation_space)
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have
        # to worry about termination conditions.
        observation_fields.update({
            'next_' + key: value
            for key, value in observation_fields.items()
        })

        fields = {
            **observation_fields,
            **{
                'actions': {
                    'shape': self._action_space.shape,
                    'dtype': 'float32'
                },
                'rewards': {
                    'shape': (1, ),
                    'dtype': 'float32'
                },
                # self.terminals[i] = a terminal was received at time i
                'terminals': {
                    'shape': (1, ),
                    'dtype': 'bool'
                },
            }
        }

        if self._include_images:
            fields = {
                **fields,
                'images': {
                    'shape':(32,32,3),
                    'dtype': 'float32'
                },
            }

        if self._super_observation_space_shape is not None:
            fields = {
                **fields,
                'super_observations': {
                    'shape': super_observation_space_shape,
                    'dtype': 'float32'
                },
            }


        super(SimpleReplayPool, self).__init__(
            *args, fields_attrs=fields, **kwargs)

    def add_samples(self, samples):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).add_samples(samples)

        dict_observations = defaultdict(list)
        for observation in samples['observations']:
            for key, value in observation.items():
                dict_observations[key].append(value)

        dict_next_observations = defaultdict(list)
        for next_observation in samples['next_observations']:
            for key, value in next_observation.items():
                dict_next_observations[key].append(value)

        samples.update(
           **{
               f'observations.{observation_key}': np.array(values)
               for observation_key, values in dict_observations.items()
           },
           **{
               f'next_observations.{observation_key}': np.array(values)
               for observation_key, values in dict_next_observations.items()
           },
        )

        del samples['observations']
        del samples['next_observations']

        if self._include_images:
            samples['images'] = np.array([
                info['image']
                for info in samples['infos']
                ])

        if self._super_observation_space_shape is not None:
            samples['super_observations'] = np.array([
                info['super_observation']
                for info in samples['infos']
                ])

        return super(SimpleReplayPool, self).add_samples(samples)

    def batch_by_indices(self,
                         indices,
                         field_name_filter=None,
                         observation_keys=None):
        if not isinstance(self._observation_space, Dict):
            return super(SimpleReplayPool, self).batch_by_indices(
                indices, field_name_filter=field_name_filter)

        batch = {
            field_name: self.fields[field_name][indices]
            for field_name in self.field_names
        }

        if observation_keys is None:
            observation_keys = tuple(self._observation_space.spaces.keys())

        observations = np.concatenate([
            batch['observations.{}'.format(key)]
            for key in observation_keys
        ], axis=-1)

        next_observations = np.concatenate([
            batch['next_observations.{}'.format(key)]
            for key in observation_keys
        ], axis=-1)

        batch['observations'] = observations
        batch['next_observations'] = next_observations

        if field_name_filter is not None:
            filtered_fields = self.filter_fields(
                batch.keys(), field_name_filter)
            batch = {
                field_name: batch[field_name]
                for field_name in filtered_fields
            }

        return batch

    def terminate_episode(self):
        pass
