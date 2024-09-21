import numpy as np
from core.agent import base
from core.network.network_architectures import FCNetwork
from core.policy.student import Student

import os
import torch

from torch import nn
import math
import torch.distributions as td
import torch.nn.functional as F

class VAE(nn.Module):
    # Vanilla Variational Auto-Encoder

    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim=750, dropout=0.0):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = 'cpu'

    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        u = self.decode(state, z)
        return u, mean, std

    def elbo_loss(self, state, action, beta, num_samples=1):
        """
        Note: elbo_loss one is proportional to elbo_estimator
        i.e. there exist a>0 and b, elbo_loss = a * (-elbo_estimator) + b
        """
        mean, std = self.encode(state, action)

        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s)

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        u = self.decode(state, z)
        recon_loss = ((u - action) ** 2).mean(dim=(1, 2))

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        vae_loss = recon_loss + beta * KL_loss

        return vae_loss.view((-1, 1))

    def iwae_loss(self, state, action, beta, num_samples=10):
        ll = self.importance_sampling_estimator(state, action, beta, num_samples)
        return -ll

    def elbo_estimator(self, state, action, beta, num_samples=1):
        mean, std = self.encode(state, action)

        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_s + std_s * torch.randn_like(std_s)

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = math.sqrt(beta / 4)

        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).sum(-1)
        elbo = log_pxz.sum(-1).mean(-1) - KL_loss
        return elbo

    def importance_sampling_estimator(self, state, action, beta, num_samples=500):
        # * num_samples correspond to num of samples L in the paper
        # * note that for exact value for \hat \log \pi_\beta in the paper, we also need **an expection over L samples**
        mean, std = self.encode(state, action)

        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x D]
        z = mean_enc + std_enc * torch.randn_like(std_enc)  # [B x S x D]

        state = state.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        action = action.repeat(num_samples, 1, 1).permute(1, 0, 2)  # [B x S x C]
        mean_dec = self.decode(state, z)
        std_dec = math.sqrt(beta / 4)

        # Find q(z|x)
        log_qzx = td.Normal(loc=mean_enc, scale=std_enc).log_prob(z)
        # Find p(z)
        mu_prior = torch.zeros_like(z).to(self.device)
        std_prior = torch.ones_like(z).to(self.device)
        log_pz = td.Normal(loc=mu_prior, scale=std_prior).log_prob(z)
        # Find p(x|z)
        std_dec = torch.ones_like(mean_dec).to(self.device) * std_dec
        log_pxz = td.Normal(loc=mean_dec, scale=std_dec).log_prob(action)

        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        ll = w.logsumexp(dim=-1) - math.log(num_samples)
        return ll

    def encode(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], -1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        return mean, std

    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        if self.max_action is not None:
            return self.max_action * torch.tanh(self.d3(a))
        else:
            return self.d3(a)




class SPOT(base.ActorCritic):
    def __init__(self, cfg):
        super(SPOT, self).__init__(cfg)

        self.num_samples = 1
        self.beta = 0.5 #cfg.tau
        self.lambd = cfg.tau

        self.vae = VAE(cfg.state_dim, cfg.action_dim, cfg.action_dim*2, 1., hidden_dim=cfg.hidden_units)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3, weight_decay=0.)
        self.n_iter = 1e5
        self.vae_train(int(self.n_iter))

    def vae_train(self, n_iter):
        for i in range(n_iter):
            data = self.get_data()
            train_states, train_actions = data['obs'], data['act']

            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(train_states, train_actions)

            recon_loss = F.mse_loss(recon, train_actions)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + self.beta * KL_loss
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            if i % 10000 == 0:
                self.logger.info("VAE training iteration {}, loss: {:.6f}".format(i, vae_loss.item()))

    def compute_loss_pi(self, data):
        states, actions = data['obs'], data['act']
        Q, _, _ = self.get_q_value_target(states, actions)
        neg_log_beta = self.vae.elbo_loss(states, actions, self.beta, self.num_samples)
        actor_loss = - Q.mean() / Q.abs().mean().detach() + self.lambd * neg_log_beta.mean()
        # print("pi", Q.size(), neg_log_beta.size())
        return actor_loss

    def compute_loss_q(self, data):
        states, actions, rewards, next_states, dones = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        with torch.no_grad():
            next_action, _ = self.ac.pi.sample(next_states)
        next_v, _, _ = self.get_q_value_target(next_states, next_action)
        q_target = rewards + (self.gamma * (1 - dones) * next_v)

        _, q1, q2 = self.get_q_value(states, actions, with_grad=True)
        critic1_loss = (0.5* (q_target - q1) ** 2).mean()
        critic2_loss = (0.5* (q_target - q2) ** 2).mean()
        loss_q = (critic1_loss + critic2_loss) * 0.5
        # print("q", q1.shape, q2.shape, q_target.shape, rewards.shape, dones.shape, actions.shape)
        return loss_q
        
    def update(self, data):

        loss_q = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

        loss_pi = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        return


    def save(self, timestamp=''):
        parameters_dir = self.parameters_dir
        params = {
            "actor_net": self.ac.pi.state_dict(),
            "critic_net": self.ac.q1q2.state_dict(),
            "vae": self.vae.state_dict()
        }
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        torch.save(params, path)

    def load(self, parameters_dir, timestamp=''):
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        model = torch.load(path)
        self.ac.pi.load_state_dict(model["actor_net"])
        self.ac.q1q2.load_state_dict(model["critic_net"])
        self.vae.load_state_dict(model["vae"])

