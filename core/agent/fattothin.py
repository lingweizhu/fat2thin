import os
import numpy as np
import torch
from core.agent import base
from core.policy.qGaussian import qHeavyTailedGaussian
from core.policy.student import Student
from core.network.network_architectures import FCNetwork
from core.utils import torch_utils


class FatToThin(base.ActorCritic):
    def __init__(self, cfg):
        super(FatToThin, self).__init__(cfg)
        self.discrete_action = cfg.discrete_control
        self.action_dim = cfg.action_dim
        self.state_dim = cfg.state_dim
        self.alpha = cfg.tau

        """
        -------- actor policy ------- 
        *GAC style 
        E_{a ~ pi_{actor}} [-ln pi_{actor}(a|s) - ln pi_{proposal}(a|s)]

        SPOT style
        E_{a ~ pi_{proposal}} [-Q{actor}(a|s) - ln pi_{proposal}(a|s)]
        """
        if self.cfg.actor_loss == "MaxLikelihood":
            self.calculate_actor_loss = self.actor_maxlikelihood
        elif self.cfg.actor_loss == "CopyMu-Pi":
            self.calculate_actor_loss = self.actor_copy_pi
        # elif self.cfg.actor_loss == "CopyMu-Proposal":
        #     self.calculate_actor_loss = self.actor_copy_proposal
        elif self.cfg.actor_loss == "KL":
            self.calculate_actor_loss = self.actor_kl
        # elif self.cfg.actor_loss == "MSE":
        #     self.calculate_actor_loss = self.actor_mse
        elif self.cfg.actor_loss == "GAC":
            self.calculate_actor_loss = self.actor_gac
        elif self.cfg.actor_loss == "SPOT":
            self.calculate_actor_loss = self.actor_spot
        elif self.cfg.actor_loss == "TAWAC":
            self.calculate_actor_loss = self.actor_tawac # TODO: Need to check the math

        if cfg.proposal_distribution == "HTqGaussian":
            self.proposal = qHeavyTailedGaussian(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units]*2, cfg.action_min, cfg.action_max, entropic_index=cfg.distribution_param)
            if cfg.distribution == "qGaussian":
                self.proposal.load_state_dict(self.ac.pi.state_dict())
        elif cfg.proposal_distribution == "Student":
            self.proposal = Student(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units] * 2, cfg.action_min, cfg.action_max, df=cfg.distribution_param)
        else:
            raise NotImplementedError

        self.proposal_optimizer = torch.optim.Adam(list(self.proposal.parameters()), cfg.pi_lr)
        self.rho = cfg.rho
        self.n_action_proposals = 10
        self.entropic_index = 0
        # match the initialization of both policies
        self.exp_threshold = 10000

        self.value_net = FCNetwork(cfg.device, np.prod(cfg.state_dim), [cfg.hidden_units]*2, 1)
        self.value_optimizer = torch.optim.Adam(list(self.value_net.parameters()), cfg.q_lr)

    def _exp_q(self, inputs, q):
        if self.entropic_index == 1:
            return torch.exp(inputs)
        else:
            return torch.maximum(torch.FloatTensor([0.]), 1 + (1 - q) * inputs) ** (1/(1-q))

    def update_value(self, data):
        """L_{\phi}, learn z for state value, v = tau log z"""
        states = data['obs']
        v_phi = self.value_net(states)
        with torch.no_grad():
            actions, _ = self.ac.pi.sample(states)
            min_Q, _, _ = self.get_q_value_target(states, actions)
        target = min_Q
        value_loss = (0.5 * (v_phi - target) ** 2).mean()
        # print("v", v_phi.size(), target.size(), min_Q.size(), actions.size())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        return

    def update_critic(self, data):
        state_batch, action_batch, reward_batch, next_state_batch, dones = (
            data['obs'], data['act'], data['reward'], data['obs2'], data['done'])

        # next_state_action, _ = self.ac.pi.sample(next_state_batch) # TODO: try to use proposal
        next_state_action, _ = self.proposal.sample(next_state_batch)

        next_q, _, _ = self.get_q_value_target(next_state_batch, next_state_action)
        target_q_value = reward_batch + self.gamma * (1 - dones) * next_q

        minq, q1, q2 = self.get_q_value(state_batch, action_batch, with_grad=True)

        critic1_loss = (0.5 * (target_q_value - q1) ** 2).mean()
        critic2_loss = (0.5 * (target_q_value - q2) ** 2).mean()
        q_loss = (critic1_loss + critic2_loss) * 0.5

        # Update the critic
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        return

    def _get_best_actions(self, state_batch, stacked_s_batch_full, sample_actions):
        batch_size = state_batch.shape[0]
        top_action = int(self.rho * self.n_action_proposals)

        q_values, _, _ = self.get_q_value(stacked_s_batch_full, sample_actions, with_grad=False)
        q_values = q_values.reshape(batch_size, self.n_action_proposals, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)
        best_ind = sorted_q[:, :top_action]
        best_ind = best_ind.repeat_interleave(self.action_dim, -1)
        sample_actions = sample_actions.reshape(batch_size, self.n_action_proposals,
                                                self.action_dim)
        best_actions = torch.gather(sample_actions, 1, best_ind)
        stacked_s_batch = state_batch.repeat_interleave(top_action, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dim))
        return stacked_s_batch_full, stacked_s_batch, best_actions

    def actor_maxlikelihood(self, state_batch, action_batch):
        action_samples, _ = self.ac.pi.rsample(state_batch)
        proposal_logprob = self.proposal.log_prob(state_batch, action_samples)
        actor_loss = -proposal_logprob.mean()
        # print(proposal_logprob.shape)
        return actor_loss

    def actor_kl(self, state_batch, action_batch): # TODO: this may need a GAC style update!!!!
        action_samples, logp = self.ac.pi.rsample(state_batch)
        with torch.no_grad():
            proposal_logprob = self.proposal.log_prob(state_batch, action_samples)
        actor_loss = (logp - proposal_logprob).mean()
        # print(logp.shape, proposal_logprob.shape, (logp - proposal_logprob).shape)

        # print()
        # print(action_samples.mean(axis=0))
        # temp, tlogp = self.proposal.sample(state_batch)
        # print(temp.mean(axis=0))
        # print(logp.mean(), proposal_logprob.mean(), tlogp.mean())
        return actor_loss

    def actor_copy_pi(self, state_batch, action_batch):
        self.ac.pi.mean_net.load_state_dict(self.proposal.mean_net.state_dict())
        action_samples, _ = self.ac.pi.rsample(state_batch)
        proposal_logprob = self.proposal.log_prob(state_batch, action_samples)
        actor_loss = -proposal_logprob.mean()
        return actor_loss

    def actor_copy_proposal(self, state_batch, action_batch):
        self.ac.pi.mean_net.load_state_dict(self.proposal.mean_net.state_dict())
        action_samples, _ = self.proposal.sample(state_batch)
        logprob = self.ac.pi.log_prob(state_batch, action_samples)
        actor_loss = -logprob.mean()
        return actor_loss

    def actor_mse(self, state_batch, action_batch):
        action_samples, _ = self.ac.pi.rsample(state_batch)

        stacked_s_batch_full = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
        proposal_samples, _ = self.proposal.sample(stacked_s_batch_full)
        stacked_s_batch_full, stacked_s_batch, best_actions = self._get_best_actions(state_batch, stacked_s_batch_full, proposal_samples)
        best_actions = best_actions.reshape(len(state_batch), int(self.rho*self.n_action_proposals), self.action_dim)
        best_actions = best_actions.mean(dim=1)
        actor_loss = torch.nn.functional.mse_loss(action_samples, best_actions)
        # print(best_actions.shape, action_samples.shape)
        # print(actor_loss)
        return actor_loss

    def actor_gac(self, state_batch, action_batch):
        stacked_s_batch_full = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
        action_samples, _ = self.ac.pi.sample(stacked_s_batch_full)
        stacked_s_batch_full, stacked_s_batch, best_actions = self._get_best_actions(state_batch, stacked_s_batch_full, action_samples)
        with torch.no_grad():
            proposal_logprob = self.proposal.log_prob(stacked_s_batch, best_actions)
        logp = self.ac.pi.log_prob(stacked_s_batch, best_actions)
        actor_loss = (-logp - proposal_logprob * self.alpha).mean()
        # print(logp.mean(), proposal_logprob.mean())
        # print(logp.shape, proposal_logprob.shape, (logp - proposal_logprob).shape)
        return actor_loss

    def actor_spot(self, state_batch, action_batch):
        action_samples, _ = self.ac.pi.rsample(state_batch)
        min_Q, _, _ = self.get_q_value(state_batch, action_samples, with_grad=True)
        with torch.no_grad():
            proposal_logprob = self.proposal.log_prob(state_batch, action_samples)
            baseline = self.value_net(state_batch)
        actor_loss = (-min_Q/baseline - proposal_logprob*self.alpha).mean()
        # print((min_Q/baseline).mean(), (proposal_logprob*self.alpha).mean())
        # print(min_Q.shape, baseline.shape, proposal_logprob.shape, (min_Q/baseline + (proposal_logprob * self.alpha)).shape)

        # stacked_s_batch_full = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
        # action_samples, _ = self.ac.pi.rsample(stacked_s_batch_full)
        # stacked_s_batch_full, stacked_s_batch, best_actions = self._get_best_actions(state_batch, stacked_s_batch_full, action_samples)
        # min_Q, _, _ = self.get_q_value(stacked_s_batch, best_actions, with_grad=True)
        # with torch.no_grad():
        #     proposal_logprob = self.proposal.log_prob(stacked_s_batch, best_actions)
        # actor_loss = -(min_Q + (proposal_logprob * self.alpha)).mean()
        return actor_loss

    def actor_tawac(self, state_batch, action_batch):
        action_samples, _ = self.ac.pi.rsample(state_batch)
        min_Q, _, _ = self.get_q_value(state_batch, action_batch, with_grad=True)
        with torch.no_grad():
            baseline = self.value_net(state_batch)
            log_probs = self.proposal.log_prob(state_batch, action_samples)
        x = (min_Q - baseline) / self.alpha
        tsallis_policy = self._exp_q(x, q=self.entropic_index)
        clipped = torch.clip(tsallis_policy, self.eps, self.exp_threshold)
        actor_loss = -(clipped * log_probs).mean()
        # print(min_Q.shape, baseline.shape, x.shape, tsallis_policy.shape, clipped.shape, log_probs.shape, (clipped * log_probs).shape)
        return actor_loss

    def update_proposal(self, data):
        state_batch, action_batch = data['obs'], data['act']
        log_probs = self.proposal.log_prob(state_batch, action_batch)
        min_Q, q1, q2 = self.get_q_value(state_batch, action_batch, with_grad=False)
        # baseline_dim = 1 if min_Q.shape[1] > 1 else 0
        # if self.entropic_index >= 1:
        #     baseline = min_Q.max(dim=1, keepdim=True)[0]
        # # elif self.entropic_index < 1:
        # else:
        #     # use mean to filter out half of bad losses
        #     baseline = min_Q.mean(dim=baseline_dim, keepdim=True)[0]
        with torch.no_grad():
            baseline = self.value_net(state_batch)
        x = (min_Q - baseline) / self.alpha
        tsallis_policy = self._exp_q(x, q=self.entropic_index)
        clipped = torch.clip(tsallis_policy, self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        self.proposal_optimizer.zero_grad()
        pi_loss.backward()
        self.proposal_optimizer.step()

    def update_actor(self, data):
        state_batch, action_batch = data['obs'], data['act']
        actor_loss = self.calculate_actor_loss(state_batch, action_batch)
        self.pi_optimizer.zero_grad()
        actor_loss.backward()
        self.pi_optimizer.step()

        # self.ac.pi.load_state_dict(self.proposal.state_dict())

    def update(self, data):
        self.update_value(data)
        self.update_critic(data)
        self.update_proposal(data)
        self.update_actor(data)
        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()
        return

    def save(self, timestamp=''):
        parameters_dir = self.parameters_dir
        params = {
            "actor_net": self.ac.pi.state_dict(),
            "critic_net": self.ac.q1q2.state_dict(),
            "proposal_net": self.proposal.state_dict(),
        }
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        torch.save(params, path)

    def load(self, parameters_dir, timestamp=''):
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        model = torch.load(path)
        self.ac.pi.load_state_dict(model["actor_net"])
        self.ac.q1q2.load_state_dict(model["critic_net"])
        self.proposal.load_state_dict(model["proposal_net"])


    def log_file(self, elapsed_time=-1, test=True):
        mean, median, min_, max_ = self.log_return(self.ep_returns_queue_train, "TRAIN", elapsed_time)
        if test:
            self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True)
            self.populate_latest = True
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "TEST", elapsed_time)
            try:
                normalized = np.array([self.eval_env.env.unwrapped.get_normalized_score(ret_) for ret_ in self.ep_returns_queue_test])
                mean, median, min_, max_ = self.log_return(normalized, "Normalized", elapsed_time)
            except:
                pass

            self.populate_states, self.populate_actions, self.populate_true_qs = self.populate_returns(log_traj=True, sampler=self.proposal)
            mean, median, min_, max_ = self.log_return(self.ep_returns_queue_test, "Proposal", elapsed_time)
            try:
                normalized = np.array(
                    [self.eval_env.env.unwrapped.get_normalized_score(ret_) for ret_ in
                     self.ep_returns_queue_test])
                mean, median, min_, max_ = self.log_return(normalized, "Proposal", elapsed_time)
            except:
                pass
        return mean, median, min_, max_
