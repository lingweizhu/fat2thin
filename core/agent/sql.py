import numpy as np
from core.agent import base
from core.network.network_architectures import FCNetwork
from core.policy.student import Student

import os
import torch
from torch.nn.utils import clip_grad_norm_

class SQL(base.ActorCritic):
    def __init__(self, cfg):
        super(SQL, self).__init__(cfg)
        self.alpha = cfg.tau # 2.0

        self.value_net = FCNetwork(cfg.device, np.prod(cfg.state_dim), [cfg.hidden_units]*2, 1)
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), cfg.q_lr)

    def update_v(self, data):
    # def update_v(critic: Model, value: Model, batch: Batch,
    #              alpha: float, alg: str) -> Tuple[Model, InfoDict]:
        # q1, q2 = critic(batch.observations, batch.actions)
        # q = jnp.minimum(q1, q2)

        states, actions = data['obs'], data['act']
        q, _, _ = self.get_q_value_target(states, actions)

        # v = value.apply({'params': value_params}, batch.observations)
        v = self.value_net(states)

        sp_term = (q - v) / (2 * self.alpha) + 1.0
        sp_weight = torch.where(sp_term > 0, 1., 0.)
        value_loss = (sp_weight * (sp_term ** 2) + v / self.alpha).mean()
        # print("v", sp_weight.size(), sp_term.size(), v.size())
        return value_loss

    # def update_q(critic: Model, value: Model,
    #              batch: Batch, discount: float) -> Tuple[Model, InfoDict]:
    def update_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        # next_v = value(batch.next_observations)
        # target_q = batch.rewards + discount * batch.masks * next_v
        with torch.no_grad():
            next_v = self.value_net(next_states)
            q_target = rewards + (self.gamma * (1 - dones) * next_v)

        _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
        critic1_loss = (0.5* (q_target - q1) ** 2).mean()
        critic2_loss = (0.5* (q_target - q2) ** 2).mean()
        loss = (critic1_loss + critic2_loss) * 0.5
        # print("q", next_v.size(), q1.size(), q2.size(), q_target.size())

        # def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        #     q1, q2 = critic.apply({'params': critic_params}, batch.observations,
        #                           batch.actions)
        #     critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        #     return critic_loss, {
        #         'critic_loss': critic_loss,
        #         'q1': q1.mean()
        #     }

        # new_critic, info = critic.apply_gradient(critic_loss_fn)

        return loss

    # def update_actor(key: PRNGKey, actor: Model, critic: Model, value: Model,
    #                  batch: Batch, alpha: float, alg: str) -> Tuple[Model, InfoDict]:
    def update_actor(self, data):
        states, actions = data['obs'], data['act']
        # v = value(batch.observations)
        with torch.no_grad():
            v = self.value_net(states)
        # q1, q2 = critic(batch.observations, batch.actions)
        # q = jnp.minimum(q1, q2)
        q, _, _ = self.get_q_value_target(states, actions)

        weight = q - v
        # weight = torch.maximum(weight, 0)
        weight = torch.clip(weight, 0, 100.)

        log_probs = self.ac.pi.log_prob(states, actions)
        loss = -(weight * log_probs).mean()
        # print("pi", weight.size(), log_probs.size())
        # def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        #     dist = actor.apply({'params': actor_params},
        #                        batch.observations,
        #                        training=True,
        #                        rngs={'dropout': key})
        #     log_probs = dist.log_prob(batch.actions)
        #     actor_loss = -(weight * log_probs).mean()
        #     return actor_loss, {'actor_loss': actor_loss}
        # new_actor, info = actor.apply_gradient(actor_loss_fn)

        return loss

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
