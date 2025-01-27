import os
import uuid
from collections import OrderedDict
from numbers import Number

import skimage
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from flatten_dict import flatten

from softlearning.models.utils import flatten_input_structure
from .rl_algorithm import RLAlgorithm


tfd = tfp.distributions


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            plotter=None,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,
            her_iters=0,
            goal_classifier_params_directory=None,
            save_full_state=False,
            save_eval_paths=False,
            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize

        self._her_iters = her_iters
        self._base_env = training_environment.unwrapped

        self._save_full_state = save_full_state
        self._save_eval_paths = save_eval_paths
        self._goal_classifier_params_directory = goal_classifier_params_directory
        self._build()

    def _build(self):
        super(SAC, self)._build()
        if self._goal_classifier_params_directory:
            self._load_goal_classifier(self._goal_classifier_params_directory)
        else:
            self._goal_classifier = None

        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _load_goal_classifier(self, goal_classifier_params_directory):
        import sys
        from goal_classifier.conv import CNN
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

        # print_tensors_in_checkpoint_file(goal_classifier_params_directory, all_tensors=False, tensor_name='')
        self._goal_classifier = CNN(goal_cond=True)
        variables = self._goal_classifier.get_variables()
        cnn_vars = [v for v in tf.trainable_variables() if v.name.split('/')[0] == 'goal_classifier']
        saver = tf.train.Saver(cnn_vars)
        saver.restore(self._session, goal_classifier_params_directory)

    def _classify_as_goals(self, images, goals):
        # NOTE: we can choose any goals we want.
        # Things to try:
        # - random goal
        # - single goal
        feed_dict = {self._goal_classifier.images: images,
            self._goal_classifier.goals: goals}
        # lid_pos = observations[:, -2]
        goal_probs = self._session.run(self._goal_classifier.pred_probs, feed_dict=feed_dict)[:, 1].reshape((-1, 1))
        return goal_probs

    def _get_Q_target(self):
        policy_inputs = flatten_input_structure({
            name: self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        })
        next_actions = self._policy.actions(policy_inputs)
        next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].observation_keys
        }
        next_Q_inputs = flatten_input_structure(
            {**next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self._alpha * next_log_pis

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * self._placeholders['rewards'],
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return tf.stop_gradient(Q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = self._get_Q_target()
        assert Q_target.shape.as_list() == [None, 1]

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': self._placeholders['actions']})
        Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        policy_inputs = flatten_input_structure({
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        })
        actions = self._policy.actions(policy_inputs)
        log_pis = self._policy.log_pis(policy_inputs, actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)
        min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
            ('alpha', self._alpha)
        ))

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""
        feed_dict = self._get_feed_dict(iteration, batch)
        self._session.run(self._training_ops, feed_dict)
        if self._her_iters:
            # Q: Is it better to build a large batch and take one grad step, or
            # resample many mini batches and take many grad steps?
            new_batches = {}
            for _ in range(self._her_iters):
                new_batch = self._get_goal_resamp_batch(batch)
                new_feed_dict = self._get_feed_dict(iteration, new_batch)
                self._session.run(self._training_ops, new_feed_dict)

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }
        if self._goal_classifier:
            if 'images' in batch.keys():
                images = batch['images']
                goal_sin = batch['observations'][:,-2].reshape((-1, 1))
                goal_cos = batch['observations'][:,-1].reshape((-1, 1))
                goals = np.arctan2(goal_sin, goal_cos)
            else:
                images = batch['observations'][:, :32*32*3].reshape((-1, 32, 32, 3))
            feed_dict[self._placeholders['rewards']] = self._classify_as_goals(images, goals)
        else:
            feed_dict[self._placeholders['rewards']] = batch['rewards']

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        return feed_dict

    def _get_goal_resamp_batch(self, batch):
        new_goal = self._base_env.sample_goal()
        old_goal = self._base_env.get_goal()
        batch_obs = batch['observations']
        batch_act = batch['actions']
        batch_next_obs = batch['next_observations']

        new_batch_obs = self._base_env.relabel_obs_w_goal(batch_obs, new_goal)
        new_batch_next_obs = self._base_env.relabel_obs_w_goal(batch_next_obs, new_goal)

        if self._base_env.use_super_state_reward():
            batch_super_obs = batch['super_observations']
            new_batch_super_obs = super(self._base_env).relabel_obs_w_goal(batch_super_obs, new_goal)
            new_batch_rew = np.expand_dims(self._base_env.compute_rewards(new_batch_super_obs, batch_act)[0], 1)
        else:
            new_batch_rew = np.expand_dims(self._base_env.compute_rewards(new_batch_obs, batch_act)[0], 1)

        new_batch = {
            'rewards': new_batch_rew,
            'observations': new_batch_obs,
            'actions': batch['actions'],
            'next_observations': new_batch_next_obs,
            'terminals': batch['terminals'],
        }
        # (TODO) Implement relabeling of terminal flags
        return new_batch

    def _init_diagnostics_ops(self):
        self._diagnostics_ops = {
            **{
                f'{key}-{metric_name}': metric_fn(values)
                for key, values in (
                        ('Q_values', self._Q_values),
                        ('Q_losses', self._Q_losses),
                        ('policy_losses', self._policy_losses))
                for metric_name, metric_fn in (
                        ('mean', tf.reduce_mean),
                        ('std', lambda x: tfp.stats.stddev(
                            x, sample_axis=None)))
            },
            'alpha': self._alpha,
            'global_step': self.global_step,
        }

        return self._diagnostics_ops

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        # TODO(hartikainen): We need to unwrap self._diagnostics_ops from its
        # tensorflow `_DictWrapper`.
        diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(flatten_input_structure({
                name: batch['observations'][name]
                for name in self._policy.observation_keys
            })).items()
        ]))

        if self._goal_classifier:
            diagnostics.update({'goal_classifier/avg_reward': np.mean(feed_dict[self._rewards_ph])})

        if self._save_eval_paths:
            import pickle
            file_name = f'eval_paths_{iteration // self.epoch_length}.pkl'
            with open(os.path.join(os.getcwd(), file_name)) as f:
                pickle.dump(evaluation_paths, f)

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        return saveables
