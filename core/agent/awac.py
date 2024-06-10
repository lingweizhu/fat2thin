import os
import torch
import torch.nn.functional as F
from core.agent import base
from core.utils import torch_utils
from core.policy.student import Student

class AWAC(base.ActorCritic):

    def __init__(self, cfg):
        super(AWAC, self).__init__(cfg)
        self.lambda_ = cfg.tau

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        lambda_ = self.lambda_

        o = data['obs']
        pi, _ = self.ac.pi.rsample(o)

        # Learned policy
        v_pi, _, _ = self.get_q_value(o, pi, with_grad=False)

        # Behavior policy
        a = data['act']
        q_old_actions, _, _ = self.get_q_value(o, a, with_grad=False)

        adv_pi = q_old_actions - v_pi
        beh_logpp = self.ac.pi.log_prob(o, a)

        weights = F.softmax(adv_pi / lambda_, dim=0)
        loss_pi = (-beh_logpp * len(weights) * weights.detach()).mean()
        # print("pi", beh_logpp.size(), weights.size(), adv_pi.size())
        return loss_pi

    def compute_loss_q(self, data):
        o, a, r, op, d = data['obs'], data['act'], data['reward'], data['obs2'], data['done']
        _, q1, q2 = self.get_q_value(o, a, with_grad=True)
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, _ = self.ac.pi.sample(op)
        q_pi_targ, _, _ = self.get_q_value_target(op, a2)
        backup = r + self.gamma * (1 - d) * (q_pi_targ)

        # MSE loss against Bellman backup
        loss_q1 = torch.nn.functional.mse_loss(q1, backup)#((q1 - backup) ** 2).mean()
        loss_q2 = torch.nn.functional.mse_loss(q2, backup)#((q2 - backup) ** 2).mean()
        loss_q = (loss_q1 + loss_q2) * 0.5
        # print("q", q1.size(), q2.size(), backup.size(), a2.size(), op.size(), d.size(), r.size())
        return loss_q

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()

    def save(self, timestamp=''):
        parameters_dir = self.parameters_dir
        params = {
            "actor_net": self.ac.pi.state_dict(),
            "critic_net": self.ac.q1q2.state_dict(),
        }
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        torch.save(params, path)


    def load(self, parameters_dir, timestamp=''):
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        model = torch.load(path)
        self.ac.pi.load_state_dict(model["actor_net"])
        self.ac.q1q2.load_state_dict(model["critic_net"])

class AWACqG(AWAC):
    def __init__(self, cfg):
        super(AWACqG, self).__init__(cfg)
        self.beh_pi = Student(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2,
                              cfg.action_min, cfg.action_max, df=cfg.distribution_param)
        self.beh_pi_optimizer = torch.optim.Adam(list(self.beh_pi.parameters()), cfg.pi_lr)

    def compute_loss_beh_pi(self, data):
        states, actions = data['obs'], data['act']
        beh_log_probs = self.beh_pi.log_prob(states, actions)
        beh_loss = -beh_log_probs.mean()
        return beh_loss, beh_log_probs

    def compute_loss_pi(self, data):
        states = data['obs']
        actions, log_pi = self.ac.pi.rsample(states)
        with torch.no_grad():
            log_pi_beta = self.beh_pi.log_prob(states, actions)
            # v = self.value_net(states)
        min_Q, _, _ = self.get_q_value_target(states, actions)
        # adv = (min_Q - v) / self.lambda_
        adv = min_Q / self.lambda_
        nonzeros = torch.where(log_pi_beta > -6.)[0]
        pi_loss = (log_pi - log_pi_beta - adv)[nonzeros].sum()/len(log_pi)
        return pi_loss, ""

    def update(self, data):
        loss_beh_pi, _ = self.compute_loss_beh_pi(data)
        self.beh_pi_optimizer.zero_grad()
        loss_beh_pi.backward()
        self.beh_pi_optimizer.step()

        # self.value_optimizer.zero_grad()
        # loss_vs, v_info, logp_info = self.compute_loss_value(data)
        # loss_vs.backward()
        # self.value_optimizer.step()

        loss_q, qinfo = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        loss_pi, _ = self.compute_loss_pi(data)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()
        return

    def save(self, timestamp=''):
        parameters_dir = self.parameters_dir
        params = {
            "actor_net": self.ac.pi.state_dict(),
            "critic_net": self.ac.q1q2.state_dict(),
            # "value_net": self.value_net.state_dict(),
            "behavior_net": self.beh_pi.state_dict()
        }
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        torch.save(params, path)


    def load(self, parameters_dir, timestamp=''):
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        model = torch.load(path)
        self.ac.pi.load_state_dict(model["actor_net"])
        self.ac.q1q2.load_state_dict(model["critic_net"])
        self.beh_pi.load_state_dict(model["behavior_net"])
