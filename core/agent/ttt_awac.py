import os
import torch
from core.agent import base
import torch.nn.functional as F

class TsallisAwacTklLoss(base.ActorCritic):
    """
    GreedyAC style implementation of MPO based on Martha's suggestion
    """
    def __init__(self, cfg):
        super(TsallisAwacTklLoss, self).__init__(cfg)
        self.action_dim = cfg.action_dim
        self.state_dim = cfg.state_dim
        self.gamma = cfg.gamma
        self.alpha = cfg.tau
        self.rho = cfg.rho
        self.entropic_index = 1 #1. / cfg.tsallis_q
        self.loss_entropic_index = 0.5 #cfg.tsallis_q#TODO: a different number
        self.n_action_proposals = 10
        self.ratio_threshold = 0.2 #ratio_threshold
        self.fdiv_name = "jensen_shannon"

    def _exp_q(self, inputs, q):
        if self.entropic_index == 1:
            return torch.exp(inputs)
        else:
            return torch.maximum(torch.FloatTensor([0.]), 1 + (1 - q) * inputs) ** (1/(1-q))

    def _log_q(self, inputs, q):
        if q == 1:
            return torch.log(inputs)
        else:
            return (inputs ** (1-q) - 1) / (1 - q)
    

    def update_critic(self, data):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            (data['obs'], data['act'], data['reward'], data['obs2'], 1 - data['done'])
        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        next_state_action, _ = self.ac.pi.sample(next_state_batch)
        next_q, _, _ = self.get_q_value_target(next_state_batch, next_state_action)
        with torch.no_grad():
            # next_q = self.critic.target_net(next_state_batch, next_state_action)
            log_base_policy = self.ac.pi.log_prob(next_state_batch, action_batch)
            _, logprobs = self.ac.pi.sample(next_state_batch)
            policy_ratio = logprobs.exp() / (log_base_policy.exp() + 1e-8)
            policy_ratio = torch.clamp(policy_ratio, 1 - self.ratio_threshold, 1 + self.ratio_threshold)
            fdiv = self.fdiv(policy_ratio, self.fdiv_name, num_terms=7)
            """
            we are minimizing distance to a TKL policy
            fdiv = TKL = -pi * ln_q mu/pi
            r - TKL = r + ln_q mu/pi
            """
            target_q_value = reward_batch + mask_batch * self.gamma * (next_q - self.alpha * fdiv)
            # target_q_value = reward_batch + mask_batch * self.gamma * next_q   
        _, q1, q2 = self.get_q_value(state_batch, action_batch, with_grad=True)
        # Calculate the loss on the critic
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        # q_loss = F.mse_loss(target_q_value, q_value)
        critic1_loss = (0.5 * (target_q_value - q1) ** 2).mean()
        critic2_loss = (0.5 * (target_q_value - q2) ** 2).mean()
        q_loss = (critic1_loss + critic2_loss) * 0.5
        # Update the critic
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        # print(reward_batch.size(), mask_batch.size(), next_q.size(), fdiv.size(), target_q_value.size(), q1.size(), q2.size())
        return

    def update_actor(self, data):

        state_batch, old_action_batch, reward_batch, next_state_batch, mask_batch = \
            (data['obs'], data['act'], data['reward'], data['obs2'], 1 - data['done'])
        # action_batch, _, _, = self.actor.policy.sample(state_batch, self.n_action_proposals)
        stacked_s_batch = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
        action_batch, _ = self.ac.pi.sample(stacked_s_batch)

        samples = int(self.rho * self.n_action_proposals)
        q_values, _, _ = self.get_q_value(stacked_s_batch, action_batch, with_grad=False)
        q_values = q_values.reshape(self.batch_size, self.n_action_proposals, 1)
        sorted_q = torch.argsort(q_values, dim=1, descending=True)

        best_ind = sorted_q[:, :samples]
        best_ind = best_ind.repeat_interleave(self.action_dim, -1)
        action_batch = action_batch.reshape(self.batch_size, self.n_action_proposals, self.action_dim)
        best_actions = torch.gather(action_batch, 1, best_ind)
        # Reshape samples for calculating the loss

        stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
        stacked_old_a_batch = old_action_batch.repeat_interleave(samples, dim=0)
        best_actions = torch.reshape(best_actions, (-1, self.action_dim))

        # with torch.no_grad():
        #     best_q_values = self.critic.value_net(stacked_s_batch, best_actions)
        #     log_base_policy = self.actor.policy.log_prob(stacked_s_batch, stacked_old_a_batch)
        best_q_values, _, _ = self.get_q_value(stacked_s_batch, best_actions, with_grad=False)
        with torch.no_grad():
            log_base_policy = self.ac.pi.log_prob(stacked_s_batch, stacked_old_a_batch)

        logprobs = self.ac.pi.log_prob(stacked_s_batch, best_actions)

        """
        when q values negative, always choose q > 1
        q - q.max, choose q > 1
        q - q.min, choose q < 1
        when q - q.mean, both
        """
        baseline_dim = 1 if best_q_values.shape[1] > 1 else 0
        if self.entropic_index >= 1:
            baseline = best_q_values.max(dim=1, keepdim=True)[0]
        # elif self.entropic_index < 1:
        else:
            """
            use mean to filter out half of bad losses
            """
            baseline = best_q_values.mean(dim=baseline_dim, keepdim=True)[0]
            # baseline = best_q_values.min(dim=baseline_dim, keepdim=True)[0]

        """
        Tsallis KL as loss function
        TKL(a|b) = E_a [-ln_q b/a] 
        """
        policy_ratio = logprobs.exp() / (log_base_policy.exp() + 1e-8)
        policy_ratio = torch.clamp(policy_ratio, 1 - self.ratio_threshold, 1 + self.ratio_threshold)
        # logq_ratio = self._log_q(policy_ratio, q=self.loss_entropic_index)
        # policy_loss = - torch.mean(exp_q_scale * logq_ratio)
        # fdiv = torch.clamp(self._forwardkl_neglnt(policy_ratio), -1/self.ratio_threshold, 1/self.ratio_threshold)
        exp_q_scale = self._exp_q((best_q_values - baseline) / self.alpha, q=self.entropic_index)
        fdiv = self.fdiv(policy_ratio, self.fdiv_name, num_terms=7)
        policy_loss = -torch.mean(exp_q_scale * fdiv)

        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

    def _logq_change_base(self, ratio, q):
        # ratio = torch.clamp(self._log_q(ratio, self.loss_entropic_index), 1 - self.ratio_threshold, 1 + self.ratio_threshold)
        return ((1 + (1-self.loss_entropic_index)*self._log_q(ratio, self.loss_entropic_index)) ** ((1-q)/(1-self.loss_entropic_index)) - 1) / (1-q)
        # return ((1 + (1-self.loss_entropic_index)*ratio) ** ((1-q)/(1-self.loss_entropic_index)) - 1) / (1-q)


    def fdiv(self, ratio, fname, num_terms=5):

        if num_terms < 2:
            return self._log_q(ratio, self.loss_entropic_index)
        
        # match fname:
        #     case "forwardkl":
        if fname == "forwardkl":
            series = 0.5 * self._logq_change_base(ratio, 2)
            for q in range(3, num_terms):
                series +=  (-1)**q / q * self._logq_change_base(ratio, q)

        elif fname == "backwardkl":
            # case "backwardkl":
            series = 0.5 * self._logq_change_base(ratio, 2)
            for q in range(3, num_terms):
                series +=  (-1)**q * (q-1) / q * self._logq_change_base(ratio, q)

        elif fname == "jeffrey":
            # case "jeffrey":
            series = self._logq_change_base(ratio, 2)
            for q in range(3, num_terms):
                series +=  (-1)**q * self._logq_change_base(ratio, q)

        elif fname == "jensen_shannon":
            # case "jensen_shannon":
            series = 0.
            for q in range(3, num_terms):
                series +=  (-1)**q * (1 - 0.5**(q-2)) / q * self._logq_change_base(ratio, q)
        elif fname == "gan":
            # case "gan":
            series = 0.25 * self._logq_change_base(ratio, 2)
            for q in range(3, num_terms):
                series +=  (-1)**q * (1 - 0.5**(q-1)) / q * self._logq_change_base(ratio, q)
        else:
            # case _:
                raise NotImplementedError
            
        return series

    
    def update(self, data):
        self.update_critic(data)
        self.update_actor(data)
        if self.use_target_network and self.total_steps % self.target_network_update_freq == 0:
            self.sync_target()
        return

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
