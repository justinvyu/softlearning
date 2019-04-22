import os
import uuid
from collections import OrderedDict
from numbers import Number

import skimage
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.training import training_util

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
            store_extra_policy_info=False,

            save_full_state=False,
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
        self._store_extra_policy_info = store_extra_policy_info

        self._save_full_state = save_full_state

        observation_shape = self._training_environment.active_observation_shape
        action_shape = self._training_environment.action_space.shape

        assert len(observation_shape) == 1, observation_shape
        self._observation_shape = observation_shape
        assert len(action_shape) == 1, action_shape
        self._action_shape = action_shape

        self._build()

    def _build(self):
        self._training_ops = {}

        self._init_global_step()
        self._init_placeholders()
        self._init_actor_update()
        self._init_critic_update()
        self._init_preprocessor_update()
        self._init_diagnostics_ops()

    def train(self, *args, **kwargs):
        """Initiate training of the SAC instance."""
        return self._train(*args, **kwargs)

    def _init_global_step(self):
        self.global_step = training_util.get_or_create_global_step()
        self._training_ops.update({
            'increment_global_step': training_util._increment_global_step(1)
        })

    def _init_placeholders(self):
        """Create input placeholders for the SAC algorithm.

        Creates `tf.placeholder`s for:
            - observation
            - next observation
            - action
            - reward
            - terminals
        """
        self._iteration_ph = tf.placeholder(
            tf.int64, shape=None, name='iteration')

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='observation',
        )

        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._observation_shape),
            name='next_observation',
        )

        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, *self._action_shape),
            name='actions',
        )

        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='rewards',
        )

        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, 1),
            name='terminals',
        )

        if self._store_extra_policy_info:
            self._log_pis_ph = tf.placeholder(
                tf.float32,
                shape=(None, 1),
                name='log_pis',
            )
            self._raw_actions_ph = tf.placeholder(
                tf.float32,
                shape=(None, *self._action_shape),
                name='raw_actions',
            )

    def _get_Q_target(self):
        next_actions = self._policy.actions([self._next_observations_ph])
        next_log_pis = self._policy.log_pis(
            [self._next_observations_ph], next_actions)

        next_Qs_values = tuple(
            Q([self._next_observations_ph, next_actions])
            for Q in self._Q_targets)

        min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_value = min_next_Q - self._alpha * next_log_pis

        Q_target = td_target(
            reward=self._reward_scale * self._rewards_ph,
            discount=self._discount,
            next_value=(1 - self._terminals_ph) * next_value)

        return Q_target

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = tf.stop_gradient(self._get_Q_target())

        assert Q_target.shape.as_list() == [None, 1]

        Q_values = self._Q_values = tuple(
            Q([self._observations_ph, self._actions_ph])
            for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.train.AdamOptimizer(
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

        actions = self._policy.actions([self._observations_ph])
        log_pis = self._policy.log_pis([self._observations_ph], actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.train.AdamOptimizer(
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

        Q_log_targets = tuple(
            Q([self._observations_ph, actions])
            for Q in self._Qs)
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

        self._policy_optimizer = tf.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

    def _init_preprocessor_update(self):
        if self._policy._preprocessor.__class__.__name__ == 'VAEPreprocessor':
            assert self._Qs[0]._preprocessor is self._Qs[1]._preprocessor
            preprocessors = (
                ('policy', self._policy._preprocessor),
                ('Q', self._Qs[0]._preprocessor),
            )

            loss_type = 'mean_squared_error'
            self.vae_reconstruction_losses = {}
            self.vae_kl_losses = {}
            self.vae_losses = {}

            for (preprocessor_name, preprocessor) in preprocessors:
                vae = preprocessor.vae
                encoder = vae.get_layer('encoder')
                image_shape = preprocessor.image_shape

                # normalized_images = (
                #     (self._observations_ph[:, :np.prod(image_shape)] + 1.0)
                #     / 2.0)
                images = self._observations_ph[:, :np.prod(image_shape)]

                loss_inputs = tf.reshape(images, (-1, *image_shape))
                loss_outputs = vae(loss_inputs)

                z_mean, z_log_var = encoder(loss_inputs)[:2]

                if loss_type == 'binary_crossentropy':
                    reconstruction_loss = tf.keras.losses.binary_crossentropy(
                        loss_inputs, loss_outputs)
                elif loss_type == 'mean_squared_error':
                    reconstruction_loss = tf.keras.losses.mean_squared_error(
                        loss_inputs, loss_outputs)
                else:
                    raise NotImplementedError(loss_type)

                reconstruction_loss = tf.keras.losses.binary_crossentropy(
                    loss_inputs, loss_outputs)

                reconstruction_loss = tf.reshape(reconstruction_loss, (-1,))
                reconstruction_loss *= np.prod(image_shape)
                reconstruction_loss = self.vae_reconstruction_losses[preprocessor_name] = (
                    tf.reduce_mean(reconstruction_loss))
                kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                kl_loss = tf.reduce_sum(kl_loss, axis=-1)
                kl_loss *= -0.5
                kl_loss = self.vae_kl_losses[preprocessor_name] = tf.reduce_mean(kl_loss)

                vae_loss = self.vae_losses[preprocessor_name] = vae.loss_weight * (
                    reconstruction_loss + vae.beta * kl_loss)

                vae_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self._policy_lr,
                    name=f"{preprocessor_name}_vae_optimizer")

                vae_train_op = vae_optimizer.minimize(
                    loss=vae_loss,
                    var_list=vae.trainable_variables)

                self._training_ops.update({
                    f'{preprocessor_name}_vae_train_op': vae_train_op
                })

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

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct TensorFlow feed_dict from sample batch."""

        feed_dict = {
            self._observations_ph: batch['observations'],
            self._actions_ph: batch['actions'],
            self._next_observations_ph: batch['next_observations'],
            self._rewards_ph: batch['rewards'],
            self._terminals_ph: batch['terminals'],
        }

        if self._store_extra_policy_info:
            feed_dict[self._log_pis_ph] = batch['log_pis']
            feed_dict[self._raw_actions_ph] = batch['raw_actions']

        if iteration is not None:
            feed_dict[self._iteration_ph] = iteration

        return feed_dict

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

        if self._policy._preprocessor.__class__.__name__ == 'VAEPreprocessor':
            self._diagnostics_ops.update(self.vae_reconstruction_losses)
            self._diagnostics_ops.update(self.vae_kl_losses)
            self._diagnostics_ops.update(self.vae_losses)

        return self._diagnostics_ops

    def _vae_diagnostics(self,
                         iteration,
                         batch,
                         training_paths,
                         evaluation_paths):
        image_scale = 32 // self._training_environment.unwrapped.image_shape[0]

        assert self._Qs[0]._preprocessor is self._Qs[1]._preprocessor
        preprocessors = (
            ('policy', self._policy._preprocessor),
            ('Q', self._Qs[0]._preprocessor),
        )

        image_dir = os.path.join(os.getcwd(), 'image-samples', 'vae')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)

        for (preprocessor_name, preprocessor) in preprocessors:
            vae = preprocessor.vae
            encoder = vae.get_layer('encoder')
            decoder = vae.get_layer('decoder')

            image_shape = preprocessor.image_shape
            image_size = np.prod(image_shape)
            encoded_shape = encoder.output[-1].shape[1:].as_list()

            num_images = 4
            # Generate never-before-seen images.
            z = np.random.normal(
                loc=np.zeros(encoded_shape),
                scale=1.0,
                size=(num_images, *encoded_shape))
            xtilde = decoder.predict(z)

            # Examine reconstruction of random images from pool.
            random_observations = self._pool.random_batch(
                num_images)['observations']
            x = random_observations[:, :image_size].reshape((-1, *image_shape))
            xhat = vae.predict(x)

            large_x = (
                x.repeat(image_scale, axis=-3).repeat(image_scale, axis=-2))
            large_xhat = (
                xhat.repeat(image_scale, axis=-3).repeat(image_scale, axis=-2))
            large_xtilde = (
                xtilde
                .repeat(image_scale, axis=-3)
                .repeat(image_scale, axis=-2))

            column_large_x = large_x.reshape((-1, *large_x.shape[-2:]))
            column_large_xhat = large_xhat.reshape(
                (-1, *large_xhat.shape[-2:]))
            column_large_xtilde = large_xtilde.reshape(
                (-1, *large_xtilde.shape[-2:]))
            grid_x = np.concatenate((
                column_large_x, column_large_xhat, column_large_xtilde,
            ), axis=-2)
            unnormalized_x = ((grid_x + 1.0) * 255.0 / 2.0).astype(np.uint8)
            skimage.io.imsave(
                os.path.join(
                    image_dir, f'{iteration}-{preprocessor_name}.png'),
                unnormalized_x)

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Records mean and standard deviation of Q-function and state
        value function, and TD-loss (mean squared Bellman error)
        for the sample batch.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        diagnostics = self._session.run(self._diagnostics_ops, feed_dict)

        policy_diagnostics = self._policy.get_diagnostics(
            batch['observations'])
        diagnostics.update({
            f'policy/{key}': value
            for key, value in policy_diagnostics.items()
        })

        if (iteration % 1000 == 0
            and self._policy._preprocessor.__class__.__name__ == 'VAEPreprocessor'):
            self._vae_diagnostics(
                iteration, batch, training_paths, evaluation_paths)

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
