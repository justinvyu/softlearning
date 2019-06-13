import os.path as osp

import numpy as np
from gym.envs.mujoco.mujoco_env import MujocoEnv

from serializable import Serializable

from softlearning.misc.utils import PROJECT_PATH
from softlearning.environments.helpers import random_point_in_circle


class Pusher2dEnv(Serializable, MujocoEnv):
    """Two-dimensional Pusher environment

    Pusher2dEnv is a two-dimensional 3-DoF manipulator. Task is to slide a
    cylinder-shaped object, or a 'puck', to a target coordinates.

    Note: Serializable has to be the first super class for classes extending
    MujocoEnv (or at least occur before MujocoEnv). Otherwise MujocoEnv calls
    Serializable.__init__ (from MujocoEnv.__init__), and the Serializable
    attributes (_Serializable__args and _Serializable__kwargs) will get
    overwritten.
    """
    MODEL_PATH = osp.abspath(
        osp.join(PROJECT_PATH, 'models', 'pusher_2d.xml'))

    JOINT_INDS = list(range(0, 3))
    PUCK_INDS = list(range(3, 5))
    TARGET_INDS = list(range(5, 7))

    # TODO.before_release Fix target visualization (right now the target is
    # always drawn in (-1, 0), regardless of the actual goal.

    def __init__(self,
                 goal=(0, -1),
                 eef_to_object_distance_cost_coeff=0,
                 goal_to_object_distance_cost_coeff=1.0,
                 ctrl_cost_coeff=0.1,
                 puck_initial_x_range=(0, 1),
                 puck_initial_y_range=(-1, -0.5),
                 goal_x_range=(-1, 0),
                 goal_y_range=(-1, 1),
                 num_goals=-1,
                 swap_goal_upon_completion=True,
                 reset_free=True):
        """
        goal (`list`): List of two elements denoting the x and y coordinates of
            the goal location. Either of the coordinate can also be a string
            'any' to make the reward not to depend on the corresponding
            coordinate.
        arm_distance_coeff ('float'): Coefficient for the arm-to-object distance
            cost.
        goal_distance_coeff ('float'): Coefficient for the object-to-goal
            distance cost.
        """
        self._Serializable__initialize(locals())

        # self._goal_mask = [coordinate != 'any' for coordinate in goal]
        # self._goal = np.array(goal)[self._goal_mask].astype(np.float32)

        self._eef_to_object_distance_cost_coeff = eef_to_object_distance_cost_coeff
        self._goal_to_object_distance_cost_coeff = goal_to_object_distance_cost_coeff
        self._ctrl_cost_coeff = ctrl_cost_coeff

        self._puck_initial_x_range = puck_initial_x_range
        self._puck_initial_y_range = puck_initial_y_range
        self._goal_x_range = goal_x_range
        self._goal_y_range = goal_y_range
        self._num_goals = num_goals
        self._swap_goal_upon_completion = swap_goal_upon_completion

        if self._num_goals > 0:
            self._goals = np.random.uniform(
                low=(goal_x_range[0], goal_y_range[0]),
                high=(goal_x_range[1], goal_y_range[1]),
                size=num_goals
            )

            if self._swap_goal_upon_completion:
                self._current_goal_index = 0
        else:
            self._swap_goal_upon_completion = False

        self._reset_free = reset_free
        # self._last_qpos = np.zeros(self.model.nq)

        MujocoEnv.__init__(self, model_path=self.MODEL_PATH, frame_skip=5)

        self.model.stat.extent = 10
        self._last_qpos = np.concatenate([
            self.sim.data.qpos.flat[self.JOINT_INDS],
            [np.mean(puck_initial_x_range), np.mean(puck_initial_y_range)],
            self.init_qpos.squeeze()[self.TARGET_INDS],
        ])

    # def _pre_action(self, action):
    #     """Action preprocessor."""
    #     action = np.clip(action, -1, 1)
    #     ctrl_range = self.sim.model.actuator_ctrlrange
    #     bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
    #     weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
    #     applied_action = bias + weight * action
    #     return applied_action

    def step(self, action):
        # action = self._pre_action(action)
        reward, info = self.compute_reward(self._get_obs(), action)

        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        done = False

        self._last_qpos = np.concatenate([
            self.sim.data.qpos.flat[self.JOINT_INDS],
            self.get_body_com("object")[:2],
            self.init_qpos.squeeze()[self.TARGET_INDS],
        ])

        return observation, reward, done, info

    def compute_reward(self, observations, actions):
        is_batch = True
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]
            is_batch = False

        eef_pos = observations[:, 9:11]
        obj_pos = observations[:, 11:13]
        goal_pos = observations[:, 13:]
        # obj_pos_masked = obj_pos[:, self._goal_mask]

        # goal_object_distances = np.linalg.norm(
        #     self._goal[None] - obj_pos_masked, axis=1)
        goal_to_object_distances = np.linalg.norm(
            goal_pos - obj_pos, ord=2, axis=1)

        eef_to_object_distances = np.linalg.norm(
            eef_pos - obj_pos, ord=2, axis=1)

        ctrl_costs = np.sum(actions**2, axis=1)

        rewards = - (
            + self._eef_to_object_distance_cost_coeff * eef_to_object_distances
            + self._goal_to_object_distance_cost_coeff * goal_to_object_distances
            + self._ctrl_cost_coeff * ctrl_costs)

        if not is_batch:
            rewards = rewards.squeeze()
            eef_to_object_distances = eef_to_object_distances.squeeze()
            goal_to_object_distances = goal_to_object_distances.squeeze()

        return rewards, {
            'eef_to_object_distance': eef_to_object_distances,
            'goal_to_object_distance': goal_to_object_distances
        }

    def viewer_setup(self):

        self.viewer.cam.trackbodyid = 0
        cam_pos = np.array([0, 0, 0, 4, -45, 0])
        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        # self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1


    def reset_model(self):
        if self._reset_free:
            qpos = self._last_qpos
            qvel = self.init_qvel.copy().squeeze()
        else:
            qpos = np.zeros(self.model.nq)
            ctrl_range = self.sim.model.actuator_ctrlrange

            # qpos[self.JOINT_INDS] = np.random.uniform(
            #     low=ctrl_range[:, 0], high=ctrl_range[:, 1])
            qpos[self.JOINT_INDS] = 0

            # qpos = np.random.uniform(
            #     low=-np.pi, high=np.pi, size=self.model.nq
            # ) + self.init_qpos.squeeze()
            # qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

            # puck_position = np.random.uniform(
            #     low=[0.3, -1.0], high=[1.0, -0.4]),
            qpos[self.PUCK_INDS] = np.random.uniform(
                low=(self._puck_initial_x_range[0],
                     self._puck_initial_y_range[0]),
                high=(self._puck_initial_x_range[1],
                      self._puck_initial_y_range[1])
                )
            qvel = self.init_qvel.copy().squeeze()
            qvel[self.PUCK_INDS] = 0
            qvel[self.TARGET_INDS] = 0

        if self._num_goals > 0:
            if self._swap_goal_upon_completion:
                puck_position = self.get_body_com("object")[:2]
                goal_position = self.get_body_com("goal")[:2]
                if np.linalg.norm(puck_position - goal_position) < 0.01:
                    other_goal_indices = [i for i in range(self._num_goals)
                                          if i != self._current_goal_index]
                    self._current_goal_index = np.random.choice(
                        other_goal_indices)

                qpos[self.TARGET_INDS] = self._goals[self._current_goal_index]
            else:
                qpos[self.TARGET_INDS] = np.random.choice(self._goals)
        else:
            qpos[self.TARGET_INDS] = np.random.uniform(
                low=(self._goal_x_range[0],
                     self._goal_y_range[0]),
                high=(self._goal_x_range[1],
                      self._goal_y_range[1])
                )

        # TODO: remnants from rllab -> gym conversion
        # qacc = np.zeros(self.sim.data.qacc.shape[0])
        # ctrl = np.zeros(self.sim.data.ctrl.shape[0])
        # full_state = np.concatenate((qpos, qvel, qacc, ctrl))

        self.set_state(np.array(qpos), np.array(qvel))

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            np.sin(self.sim.data.qpos.flat[self.JOINT_INDS]),
            np.cos(self.sim.data.qpos.flat[self.JOINT_INDS]),
            self.sim.data.qvel.flat[self.JOINT_INDS],
            self.get_body_com("distal_4")[:2],
            self.get_body_com("object")[:2],
            self.get_body_com("goal")[:2],
        ]).reshape(-1)


class ForkReacherEnv(Pusher2dEnv):
    def __init__(self,
                 arm_goal_distance_cost_coeff=1.0,
                 arm_object_distance_cost_coeff=0.0,
                 *args,
                 **kwargs):
        self._Serializable__initialize(locals())

        self._arm_goal_distance_cost_coeff = arm_goal_distance_cost_coeff
        self._arm_object_distance_cost_coeff = arm_object_distance_cost_coeff

        super(ForkReacherEnv, self).__init__(*args, **kwargs)

    def compute_reward(self, observations, actions):
        is_batch = True
        if observations.ndim == 1:
            observations = observations[None]
            actions = actions[None]
            is_batch = False
        else:
            raise NotImplementedError('Might be broken.')

        arm_pos = observations[:, -8:-6]
        object_pos = observations[:, -5:-3]
        goal_pos = observations[:, -2:]

        arm_goal_dists = np.linalg.norm(arm_pos - goal_pos, ord=2, axis=1)
        arm_object_dists = np.linalg.norm(arm_pos - object_pos, ord=2, axis=1)
        ctrl_costs = np.linalg.norm(actions, ord=2, axis=1)

        rewards = (
            - self._arm_goal_distance_cost_coeff * arm_goal_dists
            - self._arm_object_distance_cost_coeff * arm_object_dists
            - self._ctrl_cost_coeff * ctrl_costs)

        if not is_batch:
            rewards = rewards.squeeze()
            arm_goal_dists = arm_goal_dists.squeeze()
            arm_object_dists = arm_object_dists.squeeze()

        return rewards, {
            'arm_goal_distance': arm_goal_dists,
            'arm_object_distance': arm_object_dists,
            'control_cost': ctrl_costs,
        }

    def reset_model(self, qpos=None, qvel=None):
        if qpos is None:
            # qpos = np.random.uniform(
            #     low=-0.1, high=0.1, size=self.model.nq
            # ) + self.init_qpos.squeeze()

            # qpos[self.JOINT_INDS[0]] = np.random.uniform(-np.pi, np.pi)
            # qpos[self.JOINT_INDS[1]] = np.random.uniform(
            #     -np.pi/2, np.pi/2) + np.pi/4
            # qpos[self.JOINT_INDS[2]] = np.random.uniform(
            #     -np.pi/2, np.pi/2) + np.pi/2

            target_position = np.array(random_point_in_circle(
                angle_range=(0, 2*np.pi), radius=(0.6, 1.2)))
            target_position[1] += 1.0

            qpos[self.TARGET_INDS] = target_position
            # qpos[self.TARGET_INDS] = [1.0, 2.0]
            # qpos[self.TARGET_INDS] = self.init_qpos.squeeze()[self.TARGET_INDS]

            puck_position = np.random.uniform([-1.0], [1.0], size=[2])
            puck_position = (
                np.sign(puck_position)
                * np.maximum(np.abs(puck_position), 1/2))
            puck_position[np.where(puck_position == 0)] = 1.0
            # puck_position[1] += 1.0
            # puck_position = np.random.uniform(
            #     low=[0.3, -1.0], high=[1.0, -0.4]),

            qpos[self.PUCK_INDS] = puck_position

        if qvel is None:
            qvel = self.init_qvel.copy().squeeze()
            qvel[self.PUCK_INDS] = 0
            qvel[self.TARGET_INDS] = 0

        # TODO: remnants from rllab -> gym conversion
        # qacc = np.zeros(self.sim.data.qacc.shape[0])
        # ctrl = np.zeros(self.sim.data.ctrl.shape[0])
        # full_state = np.concatenate((qpos, qvel, qacc, ctrl))

        # super(Pusher2dEnv, self).reset(full_state)

        self.set_state(np.array(qpos), np.array(qvel))

        return self._get_obs()

    def _get_obs(self):
        super_observation = super(ForkReacherEnv, self)._get_obs()
        observation = np.concatenate([
            super_observation, self.get_body_com('goal')[:2]
        ])
        return observation
