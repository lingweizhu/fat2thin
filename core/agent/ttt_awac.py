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
        self.n_action_proposals = 10
        self.ratio_threshold = 0.2
        self.fdiv_name = cfg.fdiv_info[0]
        self.fdiv_term = int(cfg.fdiv_info[1])
        self.beh_pi = self.get_policy_func(cfg.discrete_control, cfg)
        self.beh_pi_optimizer = torch.optim.Adam(list(self.beh_pi.parameters()), cfg.pi_lr)
        self.log_clip = False

        self.entropic_index = 0.0
        self.loss_entropic_index = 0.0
        # self.entropic_index = 0.5
        # self.loss_entropic_index = 0.5
        if self.loss_entropic_index == 0.:
            self.clamp_ratio = self.clamp_ratio_0
        elif self.loss_entropic_index == 0.5:
            self.clamp_ratio = self.clamp_ratio_1_2
        elif self.loss_entropic_index == 2./3.:
            self.clamp_ratio = self.clamp_ratio_2_3
        elif self.loss_entropic_index == 2.:
            self.clamp_ratio = self.clamp_ratio_2_1
        elif self.loss_entropic_index == 3.:
            self.clamp_ratio = self.clamp_ratio_3_1
        else:
            raise NotImplementedError

        # if self.entropic_index == self.loss_entropic_index:
        #     self.get_baseline = self.get_baseline_max
        #     self.fdiv = self.fdiv_equal_idx
        # else:
        #     self.get_baseline = self.get_baseline_default
        #     raise NotImplementedError
        #     # change return to loss
        #     self.fdiv = self.fdiv_default

    def clamp_ratio_0(self, ratio):
        return torch.clamp(ratio, min=1-self.ratio_threshold, max=1+self.ratio_threshold) - 1.

    def clamp_ratio_1_2(self, ratio):
        ret = 2.0 * (torch.clamp(torch.sqrt(ratio), min=1-self.ratio_threshold, max=1+self.ratio_threshold) - 1.)
        return ret

    def clamp_ratio_2_3(self, ratio):
        return 3.0 * (torch.clamp(ratio**(1/3), min=1-self.ratio_threshold, max=1+self.ratio_threshold) - 1.)

    def clamp_ratio_3_4(self, ratio):
        return 4.0 * (torch.clamp(ratio**(1/4), min=1-self.ratio_threshold, max=1+self.ratio_threshold) - 1.)

    def clamp_ratio_2_1(self, ratio):
        return -1. * (torch.clamp(ratio**(-1), min=1-self.ratio_threshold, max=1+self.ratio_threshold) - 1.)

    def clamp_ratio_3_1(self, ratio):
        return (torch.clamp(ratio**(-2), min=1-self.ratio_threshold, max=1+self.ratio_threshold) - 1.) / (-2)


    def update_beh_pi(self, data):
        """L_{\omega}, learn behavior policy"""
        states, actions = data['obs'], data['act']
        beh_log_probs = self.beh_pi.log_prob(states, actions)
        beh_loss = -beh_log_probs.mean()
        # print("beh", beh_log_probs.size(), beh_loss)
        self.beh_pi_optimizer.zero_grad()
        beh_loss.backward()
        self.beh_pi_optimizer.step()
        return

    def _exp_q(self, inputs, q):
        if self.entropic_index == 1:
            return torch.exp(inputs)
        else:
            return torch.maximum(torch.FloatTensor([0.]), 1 + (1 - q) * inputs) ** (1/(1-q))

    # def _log_q(self, inputs, q):
    #     if q == 1:
    #         return torch.log(inputs)
    #     else:
    #         return (inputs ** (1-q) - 1) / (1 - q)
    

    def update_critic(self, data):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = \
            (data['obs'], data['act'], data['reward'], data['obs2'], 1 - data['done'])
        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        next_action, logprobs = self.ac.pi.sample(next_state_batch)
        next_q, _, _ = self.get_q_value_target(next_state_batch, next_action)
        # with torch.no_grad():
        #     log_base_policy = self.beh_pi.log_prob(next_state_batch, next_action)
        # policy_ratio = logprobs.exp() / (log_base_policy.exp() + 1e-8)
        # # if not self.log_clip:
        # #     policy_ratio = torch.clamp(policy_ratio, 1 - self.ratio_threshold, 1 + self.ratio_threshold)
        # fdiv = self.fdiv(policy_ratio, self.fdiv_name, num_terms=self.fdiv_term)

        """
        we are minimizing distance to a TKL policy
        fdiv = TKL = -pi * ln_q mu/pi
        r - TKL = r + ln_q mu/pi
        """
        # target_q_value = reward_batch + mask_batch * self.gamma * (next_q - self.alpha * fdiv)
        target_q_value = reward_batch + mask_batch * self.gamma * next_q
        _, q1, q2 = self.get_q_value(state_batch, action_batch, with_grad=True)
        # Calculate the loss on the critic
        critic1_loss = (0.5 * (target_q_value - q1) ** 2).mean()
        critic2_loss = (0.5 * (target_q_value - q2) ** 2).mean()
        q_loss = (critic1_loss + critic2_loss) * 0.5
        # print(target_q_value.mean(), policy_ratio.mean(), fdiv.mean())
        # Update the critic
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        # print(torch.where(torch.isnan(next_q)), torch.where(torch.isnan(q1)), torch.where(torch.isnan(q2)))
        # print("critic", q_loss)
        # print("Q", reward_batch.size(), mask_batch.size(), next_q.size(), logprobs.size(), log_base_policy.size(), policy_ratio.size(), fdiv.size(), target_q_value.size(), q1.size(), q2.size())
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

    def get_baseline_max(self, best_q_values):
        return best_q_values.max()

    def get_baseline_default(self, best_q_values):
        """
        when q values negative, always choose q > 1
        q - q.max, choose q > 1
        q - q.min, choose q < 1
        when q - q.mean, both
        """

        baseline_dim = 1 if best_q_values.shape[1] > 1 else 0
        if self.entropic_index >= 1:
            # baseline = best_q_values.max(dim=1, keepdim=True)[0]
            baseline = best_q_values.max()[0]
        # elif self.entropic_index < 1:
        else:
            """
            use mean to filter out half of bad losses
            """
            baseline = best_q_values.mean(dim=baseline_dim, keepdim=True)[0]
            # baseline = best_q_values.min(dim=baseline_dim, keepdim=True)[0]
        return baseline

    def update_actor(self, data):
        state_batch, old_action_batch, reward_batch, next_state_batch, mask_batch = \
            (data['obs'], data['act'], data['reward'], data['obs2'], 1 - data['done'])

        best_q_values, _, _ = self.get_q_value(state_batch, old_action_batch, with_grad=False)
        baseline = self.get_baseline(best_q_values)
        """
        Tsallis KL as loss function
        TKL(a|b) = E_a [-ln_q b/a] 
        """
        logprobs = self.ac.pi.log_prob(state_batch, old_action_batch)
        with torch.no_grad():
            log_base_policy = self.beh_pi.log_prob(state_batch, old_action_batch)
        policy_ratio = logprobs.exp() / (log_base_policy.exp() + 1e-8)
        advantage = (best_q_values - baseline) / self.alpha

        # if not self.log_clip:
        #     policy_ratio = torch.clamp(policy_ratio, 1 - self.ratio_threshold, 1 + self.ratio_threshold)
        fdiv = self.fdiv_default(policy_ratio, advantage, self.fdiv_name, num_terms=self.fdiv_term)
        exp_q_scale = self._exp_q((best_q_values - baseline) / self.alpha, q=self.entropic_index)
        policy_loss = -torch.mean(exp_q_scale * fdiv)

        # policy_loss = self.fdiv(policy_ratio, advantage, self.fdiv_name, num_terms=self.fdiv_term)


        # print(torch.where(torch.isnan(best_q_values)))
        # print(torch.where(torch.isnan(fdiv)))
        # print("actor", policy_ratio.mean(), fdiv.mean(), best_q_values.mean(), policy_loss)
        # print()

        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

    def _logq_change_base(self, ratio, q):
        # if self.log_clip:
        #     ratio = torch.clamp(self._log_q(ratio, self.loss_entropic_index), 1 - self.ratio_threshold, 1 + self.ratio_threshold)
        #     return ((1 + (1-self.loss_entropic_index)*ratio) ** ((1-q)/(1-self.loss_entropic_index)) - 1) / (1-q)
        # else:
        #     return ((1 + (1-self.loss_entropic_index)*self._log_q(ratio, self.loss_entropic_index)) ** ((1-q)/(1-self.loss_entropic_index)) - 1) / (1-q)
        ret = ((1 + (1 - self.loss_entropic_index) * self.clamp_ratio(ratio)) ** (
                            (1 - q) / (1 - self.loss_entropic_index)) - 1) / (1 - q)
        return ret

    def fdiv_equal_idx(self, ratio, advantage, fname, num_terms=5):
        """
        assume the advantage is (best_q_values - baseline) / self.alpha
        we compute exp_q (advantage) and ln_q ratio in the infinite series
        by computing them each as a vector and then do vector multiplication
        """
        assert self.entropic_index == self.loss_entropic_index, "you can only call this fdiv when entropic_index = loss_entropic_index!"

        if num_terms < 2:
            return - torch.mean(
                self._exp_q(advantage, q=self.entropic_index) * self.clamp_ratio(ratio))

        expq_adv_array = torch.cat(
            [self._exp_q(advantage, q=q)[None, :] for q in range(2, num_terms + 1)], dim=0)

        if fname == "forwardkl":
            lnq_ratio_array = torch.cat(
                [(-1) ** q / q * self._logq_change_base(ratio, q)[None, :] for q in range(2, num_terms + 1)], dim=0)

        elif fname == "backwardkl":
            lnq_ratio_array = torch.cat(
                [(-1) ** q * (q - 1) / q * self._logq_change_base(ratio, q)[None, :] for q in
                 range(2, num_terms + 1)], dim=0)

        elif fname == "jeffrey":
            lnq_ratio_array = torch.cat(
                [(-1) ** q * self._logq_change_base(ratio, q)[None, :] for q in range(2, num_terms + 1)], dim=0)

        elif fname == "jensen_shannon":
            lnq_ratio_array = torch.cat(
                [(-1) ** q * (1 - 0.5 ** (q - 2)) / q * self._logq_change_base(ratio, q)[None, :] for q in
                 range(2, num_terms + 1)], dim=0)

        elif fname == "gan":
            lnq_ratio_array = torch.cat(
                [(-1) ** q * (1 - 0.5 ** (q - 1)) / q * self._logq_change_base(ratio, q)[None, :] for q in
                 range(2, num_terms + 1)], dim=0)

            # print(advantage.size(), self._exp_q(advantage, q=2).size(), ratio.size(), self.clamp_ratio(ratio).size())
            # print(expq_adv_array[0].size())
            # print(expq_adv_array.size())
            # print(self._logq_change_base(ratio, 2).size())
            # print(lnq_ratio_array.size())
            # exit()

        else:
            raise NotImplementedError

        # series = - torch.matmul(expq_adv_array, lnq_ratio_array)
        # print((expq_adv_array * lnq_ratio_array)[:, 0, :])
        # print(torch.sum(expq_adv_array * lnq_ratio_array, dim=0)[0, :])
        series = - torch.sum(expq_adv_array * lnq_ratio_array, dim=0).mean()
        return series

    def fdiv_default(self, ratio, placeholder, fname, num_terms=5):

        if num_terms < 2:
            # if self.log_clip:
            #     return torch.clamp(self._log_q(ratio, self.loss_entropic_index), 1 - self.ratio_threshold, 1 + self.ratio_threshold)
            # else:
            #     return self._log_q(ratio, self.loss_entropic_index)
            return self.clamp_ratio(ratio)

        if fname == "forwardkl":
            series = 0.5 * self._logq_change_base(ratio, 2)
            for q in range(3, num_terms+1):
                series +=  (-1)**q / q * self._logq_change_base(ratio, q)

        elif fname == "backwardkl":
            series = 0.5 * self._logq_change_base(ratio, 2)
            for q in range(3, num_terms+1):
                series +=  (-1)**q * (q-1) / q * self._logq_change_base(ratio, q)

        elif fname == "jeffrey":
            series = self._logq_change_base(ratio, 2)
            for q in range(3, num_terms+1):
                series +=  (-1)**q * self._logq_change_base(ratio, q)
            # print(series.mean())

        elif fname == "jensen_shannon":
            temp = [ratio.mean()]
            series = 0.
            for q in range(3, num_terms+1):
                term = (-1)**q * (1 - 0.5**(q-2)) / q * self._logq_change_base(ratio, q)
                series += term
                temp.append("logp={:.6f}".format(self._logq_change_base(ratio, q).detach().mean()))
                temp.append("term={:.6f}".format(term.detach().mean()))
            # print(temp)
            # print(series.mean())

        elif fname == "gan":
            # series = 0.25 * self._logq_change_base(ratio, 2)
            # for q in range(3, num_terms):
            #     series +=  (-1)**q * (1 - 0.5**(q-1)) / q * self._logq_change_base(ratio, q)
            # # print(series.mean())
            series = 0.
            # temp = [ratio.mean()]
            for q in range(2, num_terms+1):
                term = (-1)**q * (1 - 0.5**(q-1)) / q * self._logq_change_base(ratio, q)
                series += term
                # temp.append("logp={:.6f}".format(self._logq_change_base(ratio, q).detach().mean()))
                # temp.append("term={:.6f}".format(term.detach().mean()))
            # print(temp)
            # print(series.mean())
        else:
            # case _:
                raise NotImplementedError
        # series /= series.detach().max()
        return series

    
    def update(self, data):
        self.update_beh_pi(data)
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
            "behavior_net": self.beh_pi.state_dict(),
        }
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        torch.save(params, path)

    def load(self, parameters_dir, timestamp=''):
        path = os.path.join(parameters_dir, "parameter"+timestamp)
        model = torch.load(path)
        self.ac.pi.load_state_dict(model["actor_net"])
        self.ac.q1q2.load_state_dict(model["critic_net"])
        self.beh_pi.load_state_dict(model["behavior_net"])
