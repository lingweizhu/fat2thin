import numpy as np
from core.agent import base
from core.network.network_architectures import FCNetwork
from core.policy.student import Student

import os
import torch
from torch.nn.utils import clip_grad_norm_

class XQL(base.ActorCritic):
    def __init__(self, cfg):
        super(XQL, self).__init__(cfg)

        self.temperature = cfg.tau
        self.expectile = cfg.expectile # 0.8
        self.sample_random_times = 1
        self.noise = True
        self.vanilla = True
        self.noise_std = 0.1
        self.log_loss = True
        self.loss_temp = 1.0

        self.value_net = FCNetwork(cfg.device, np.prod(cfg.state_dim), [cfg.hidden_units]*2, 1)
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), cfg.q_lr)

    # def update_v(critic: Model, argsvalue: Model, batch: Batch,
    #              expectile: float, loss_temp: float, double: bool, vanilla: bool, key: PRNGKey,
    #              args) -> Tuple[Model, InfoDict]:
    def update_v(self, data):
        states, actions = data['obs'], data['act']

        # rng1, rng2 = jax.random.split(key)
        if self.sample_random_times > 0:
            # add random actions to smooth loss computation (use 1/2(rho + Unif))
            times = self.sample_random_times
            # random_action = jax.random.uniform(
            #     rng1, shape=(times * actions.shape[0],
            #                  actions.shape[1]),
            #     minval=-1.0, maxval=1.0)
            random_action = torch.rand(times * actions.shape[0], actions.shape[1]) * 2.0 - 1.0
            obs = torch.concatenate([states for _ in range(self.sample_random_times + 1)], axis=0)
            acts = torch.concatenate([actions, random_action], axis=0)
        else:
            obs = states
            acts = actions

        if self.noise:
            std = self.noise_std
            # noise = jax.random.normal(rng2, shape=(acts.shape[0], acts.shape[1]))
            noise = torch.normal(mean=torch.zeros(acts.shape), std=torch.ones(acts.shape))
            noise = torch.clip(noise * std, -0.5, 0.5)
            acts = (acts + noise)
            acts = torch.clip(acts, -1, 1)

        # q1, q2 = critic(obs, acts)
        # if double:
        #     q = jnp.minimum(q1, q2)
        # else:
        #     q = q1
        q, _, _ = self.get_q_value_target(obs, acts)

        # v = value.apply({'params': value_params}, obs)
        v = self.value_net(obs)

        if self.vanilla:
            value_loss = self.expectile_loss(q - v, self.expectile).mean()
        else:
            if self.log_loss:
                value_loss = self.gumbel_log_loss(q - v, alpha=self.loss_temp).mean()
            else:
                value_loss = self.gumbel_rescale_loss(q - v, alpha=self.loss_temp).mean()

        # print("v", q.size(), v.size(), self.gumbel_log_loss(q - v, alpha=self.loss_temp).size(),
        #       std, noise.size(), acts.size(), obs.size())
        return value_loss

    # def gumbel_rescale_loss(self, diff, alpha, args=None):
    #     """ Gumbel loss J: E[e^x - x - 1]. For stability to outliers, we scale the gradients with the max value over a batch
    #     and optionally clip the exponent. This has the effect of training with an adaptive lr.
    #     """
    #     z = diff / alpha
    #     if args.max_clip is not None:
    #         z = torch.minimum(z, args.max_clip)  # clip max value
    #     max_z = torch.maximum(z, axis=0)
    #     max_z = torch.where(max_z < -1.0, -1.0, max_z)
    #     # max_z = jax.lax.stop_gradient(max_z)  # Detach the gradients
    #     max_z
    #     loss = jnp.exp(z - max_z) - z * jnp.exp(-max_z) - jnp.exp(-max_z)  # scale by e^max_z
    #     return loss

    def gumbel_log_loss(self, diff, alpha=1.0):
        """ Gumbel loss J: E[e^x - x - 1]. We can calculate the log of Gumbel loss for stability, i.e. Log(J + 1)
        log_gumbel_loss: log((e^x - x - 1).mean() + 1)
        """
        diff = diff
        x = diff / alpha
        grad = self.grad_gumbel(x, alpha)
        # use analytic gradients to improve stability
        # loss = jax.lax.stop_gradient(grad) * x
        loss = grad.detach() * x
        # print("gumbel log", grad.size(), diff.size(), x.size())
        return loss

    def grad_gumbel(self, x, alpha, clip_max=7):
        """Calculate grads of log gumbel_loss: (e^x - 1)/[(e^x - x - 1).mean() + 1]
        We add e^-a to both numerator and denominator to get: (e^(x-a) - e^(-a))/[(e^(x-a) - xe^(-a)).mean()]
        """
        # clip inputs to grad in [-10, 10] to improve stability (gradient clipping)
        x = torch.clip(x, -np.inf, clip_max)  # jnp.clip(x, a_min=-10, a_max=10)

        # calculate an offset `a` to prevent overflow issues
        x_max = torch.max(x, axis=0)[0]
        # choose `a` as max(x_max, -1) as its possible for x_max to be very small and we want the offset to be reasonable
        x_max = torch.where(x_max < -1, -1, x_max)
        # keep track of original x
        x_orig = x
        # offsetted x
        x1 = x - x_max

        grad = (torch.exp(x1) - torch.exp(-x_max)) / \
               (torch.mean(torch.exp(x1) - x_orig * torch.exp(-x_max), axis=0, keepdims=True))
        # print("gumbel", x1.size(), x_max, x_orig.size())
        return grad

    def expectile_loss(self, diff, expectile=0.8):
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        # print("expectile", weight.size(), diff.size())
        return weight * (diff ** 2)

    # def update_q(critic: Model, target_value: Model, batch: Batch,
    #              discount: float, double: bool, key: PRNGKey, loss_temp: float, args) -> Tuple[
    def update_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']

        # next_v = target_value(batch.next_observations)
        with torch.no_grad():
            next_v = self.value_net(next_states)
            v = self.value_net(states)

        target_q = rewards + self.gamma * (1. - dones) * next_v

        # q1, q2 = critic.apply({'params': critic_params}, batch.observations, acts)
        # v = target_value(batch.observations)
        _, q1, q2 = self.get_q_value(states, actions, with_grad=True)


        def mse_loss(q, q_target, *args):
            x = q - q_target
            loss = self.huber_loss(x, delta=20.0)  # x**2
            return loss.mean()

        critic_loss = mse_loss

        loss1 = critic_loss(q1, target_q, v, self.loss_temp)
        loss2 = critic_loss(q2, target_q, v, self.loss_temp)
        critic_loss = (loss1 + loss2).mean()
        # print("q", next_v.size(), v.size(), q1.size(), q2.size(), target_q.size())

        # if args.grad_pen:
        #     lambda_ = args.lambda_gp
        #     q1_grad, q2_grad = grad_norm(critic, critic_params, batch.observations, acts)
        #     loss_dict['q1_grad'] = q1_grad.mean()
        #     loss_dict['q2_grad'] = q2_grad.mean()
        #
        #     if double:
        #         gp_loss = (q1_grad + q2_grad).mean()
        #     else:
        #         gp_loss = q1_grad.mean()
        #
        #     critic_loss += lambda_ * gp_loss

        # loss_dict.update({
        #     'q1': q1.mean(),
        #     'q2': q2.mean()
        # })
        #
        # new_critic, info = critic.apply_gradient(critic_loss_fn)

        return critic_loss

    def huber_loss(self, x, delta: float = 1.):
        """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
        See "Robust Estimation of a Location Parameter" by Huber.
        (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
        Args:
        x: a vector of arbitrary shape.
        delta: the bounds for the huber loss transformation, defaults at 1.
        Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
        Returns:
        a vector of same shape of `x`.
        """
        # 0.5 * x^2                  if |x| <= d
        # 0.5 * d^2 + d * (|x| - d)  if |x| > d
        abs_x = torch.abs(x)
        quadratic = torch.clip(abs_x, 0, delta)
        # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
        linear = abs_x - quadratic
        # print("huber", abs_x.size(), quadratic.size())
        return 0.5 * quadratic ** 2 + delta * linear

    # def update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model,
    #            batch: Batch, temperature: float, double: bool) -> Tuple[Model, InfoDict]:
    def update_actor(self, data):
        states, actions = data['obs'], data['act']
        # v = value(batch.observations)
        with torch.no_grad():
            v = self.value_net(states)

        # q1, q2 = critic(batch.observations, batch.actions)
        # if double:
        #     q = jnp.minimum(q1, q2)
        # else:
        #     q = q1
        q, _, _ = self.get_q_value_target(states, actions)

        exp_a = torch.exp((q - v) * self.temperature)
        # exp_a = torch.minimum(exp_a, 100.0)
        exp_a = torch.clip(exp_a, 0., 100.0)

        # dist = actor.apply({'params': actor_params},
        #                    batch.observations,
        #                    training=True,
        #                    rngs={'dropout': key})
        # log_probs = dist.log_prob(batch.actions)
        log_probs = self.ac.pi.log_prob(states, actions)
        actor_loss = -(exp_a * log_probs).mean()

        # print(exp_a.size(), log_probs.size(), q.size(), v.size())
        return actor_loss

    def update(self, data):
        self.value_optimizer.zero_grad()
        loss_vs = self.update_v(data)
        loss_vs.backward()
        self.value_optimizer.step()
        
        loss_q = self.update_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

        loss_pi = self.update_actor(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        return


    def save(self, timestamp=''):
        parameters_dir = self.parameters_dir
        params = {
            "actor_net": self.ac.pi.state_dict(),
            "critic_net": self.ac.q1q2.state_dict(),
            "value_net": self.value_net.state_dict()
        }
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        torch.save(params, path)

    def load(self, parameters_dir, timestamp=''):
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        model = torch.load(path)
        self.ac.pi.load_state_dict(model["actor_net"])
        self.ac.q1q2.load_state_dict(model["critic_net"])
        self.value_net.load_state_dict(model["value_net"])

