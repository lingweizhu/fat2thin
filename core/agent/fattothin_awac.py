import os
import torch
from core.agent import base
from core.policy.qGaussian import qMultivariateGaussian


class FatToThinQGaussianAWAC(base.ActorCritic):
    # def __init__(self,
    #              discrete_action: bool,
    #              action_dim: int,
    #              state_dim: int,
    #              gamma: float,
    #              batch_size: float,
    #              alpha: float,
    #              device: torch.device,
    #              behavior_policy: torch.nn.Module,
    #              proposal_policy: torch.nn.Module,
    #              critic: torch.nn.Module,
    #              replay_buffer: torch.nn.Module,
    #              rho: float,
    #              n_action_proposals: float,
    #              entropic_index: float,
    #              ) -> None:
    #     super().__init__()
    def __init__(self, cfg):
        super(FatToThinQGaussianAWAC, self).__init__(cfg)
        self.discrete_action = cfg.discrete_control
        self.action_dim = cfg.action_dim
        self.state_dim = cfg.state_dim
        # self.batch_size = batch_size
        self.alpha = cfg.tau
        # self.device = device
        # self.bp = behavior_policy
        # self.proposal = self.get_policy_func(cfg.discrete_control, cfg)
        self.proposal = qMultivariateGaussian(cfg.device, cfg.state_dim, cfg.action_dim, [cfg.hidden_units]*2, cfg.action_min, cfg.action_max, entropic_index=cfg.distribution_param)
        self.proposal_optimizer = torch.optim.Adam(list(self.proposal.parameters()), cfg.pi_lr)
        # self.critic = critic
        # self.buffer = replay_buffer
        self.rho = cfg.rho
        self.n_action_proposals = 10#n_action_proposals
        self.entropic_index = 0 #cfg.tsallis_q
        # match the initialization of both policies
        self.proposal.load_state_dict(self.ac.pi.state_dict())
        self.exp_threshold = 10000

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

    # @torch.no_grad()
    # def act(self, state: Float[np.ndarray, "state_dim"], greedy: bool=False) -> Float[np.ndarray, "action_dim"]:
    #     state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
    #     action, _, action_mean = self.bp.sample(state)
    #     act = action.detach().cpu().numpy()[0]
    #     if greedy:
    #         act = action_mean.detach().cpu().numpy()[0]
    #     if not self.discrete_action:
    #         return act
    #     else:
    #         return int(act[0])

    def update_critic(self, data):
        state_batch, action_batch, reward_batch, next_state_batch, dones = (
            data['obs'], data['act'], data['reward'], data['obs2'], data['done'])

        # When updating Q functions, we don't want to backprop through the
        # policy and target network parameters
        # next_state_action, _, _ = self.bp.policy.sample(next_state_batch)
        next_state_action, _ = self.ac.pi.sample(next_state_batch)
        # with torch.no_grad():
        #     next_q = self.critic.target_net(next_state_batch, next_state_action)
        #     target_q_value = reward_batch + mask_batch * self.gamma * next_q
        next_q, _, _ = self.get_q_value_target(next_state_batch, next_state_action)
        target_q_value = reward_batch + self.gamma * (1 - dones) * next_q

        # q_value = self.critic.value_net(state_batch, action_batch)
        minq, q1, q2 = self.get_q_value(state_batch, action_batch, with_grad=True)

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
    # def _get_best_actions(self, state_batch, action_batch):
    #     action_batch = action_batch.permute(1, 0, 2)
    #     action_batch = action_batch.reshape(self.batch_size * self.n_action_proposals, self.action_dim)
    #     stacked_s_batch_full = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
    #     # Get the values of the sampled actions and find the best rho * n_action_proposals actions
    #     with torch.no_grad():
    #         q_values = self.critic.value_net(stacked_s_batch_full, action_batch)
    #     q_values = q_values.reshape(self.batch_size, self.n_action_proposals, 1)
    #     sorted_q = torch.argsort(q_values, dim=1, descending=True)
    #     best_ind = sorted_q[:, :int(self.rho * self.n_action_proposals)]
    #     best_ind = best_ind.repeat_interleave(self.action_dim, -1)
    #     action_batch = action_batch.reshape(self.batch_size, self.n_action_proposals, self.action_dim)
    #     best_actions = torch.gather(action_batch, 1, best_ind)
    #     # Reshape samples for calculating the loss
    #     samples = int(self.rho * self.n_action_proposals)
    #     stacked_s_batch = state_batch.repeat_interleave(samples, dim=0)
    #     best_actions = torch.reshape(best_actions, (-1, self.action_dim))
    #
    #     return stacked_s_batch_full, stacked_s_batch, best_actions
    

    def update_actor(self, data) -> None:
        # # Sample a batch from memory
        # # state_batch, action_batch, reward_batch, next_state_batch, \
        # #     mask_batch = self.buffer.sample(batch_size=self.batch_size)
        # # if state_batch is None:
        # #     # Too few samples in the buffer to sample
        # #     return
        # state_batch, action_batch = data['obs'], data['act']
        # # stacked_s_batch_full = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
        # # sample_actions, _ = self.proposal.sample(stacked_s_batch_full)
        # # _, stacked_s_batch, best_actions = self._get_best_actions(state_batch, stacked_s_batch_full, sample_actions)
        # stacked_s_batch = state_batch
        # best_actions = action_batch
        #
        # """
        # -------- proposal policy -------
        # proposal loss:
        # E_{pi_{old}} [ -exp_q( Q - V ) * ln pi_{proposal}} ]
        #
        # pi_{old} may be replay buffer or dataset
        #
        # when q values negative, always choose q > 1
        # q - q.max, choose q > 1
        # q - q.min, choose q < 1
        # when q - q.mean, both
        # """
        # # with torch.no_grad():
        # #     best_q_values = self.critic.value_net(stacked_s_batch, best_actions)
        # best_q_values, _, _ = self.get_q_value(stacked_s_batch, best_actions, with_grad=False)
        #
        # # proposal_logprobs = self.pp.policy.log_prob(stacked_s_batch, best_actions)
        # proposal_logprobs = self.proposal.log_prob(stacked_s_batch, best_actions)
        #
        # baseline_dim = 1 if best_q_values.shape[1] > 1 else 0
        # if self.entropic_index >= 1:
        #     baseline = best_q_values.max(dim=1, keepdim=True)[0]
        # # elif self.entropic_index < 1:
        # else:
        #     # use mean to filter out half of bad losses
        #     baseline = best_q_values.mean(dim=baseline_dim, keepdim=True)[0]
        #
        # exp_q_scale = self._exp_q((best_q_values - baseline) / self.alpha, q=self.entropic_index)
        # proposal_loss = - torch.mean(exp_q_scale * proposal_logprobs)
        #
        # # self.pp.optimizer.zero_grad()
        # self.proposal_optimizer.zero_grad()
        # proposal_loss.backward()
        # self.proposal_optimizer.step()

        state_batch, action_batch = data['obs'], data['act']
        log_probs = self.proposal.log_prob(state_batch, action_batch)
        min_Q, q1, q2 = self.get_q_value(state_batch, action_batch, with_grad=False)

        baseline_dim = 1 if min_Q.shape[1] > 1 else 0
        if self.entropic_index >= 1:
            baseline = min_Q.max(dim=1, keepdim=True)[0]
        # elif self.entropic_index < 1:
        else:
            # use mean to filter out half of bad losses
            baseline = min_Q.mean(dim=baseline_dim, keepdim=True)[0]
        x = (min_Q - baseline) / self.alpha
        tsallis_policy = self._exp_q(x, q=self.entropic_index)
        clipped = torch.clip(tsallis_policy, self.eps, self.exp_threshold)
        pi_loss = -(clipped * log_probs).mean()
        self.proposal_optimizer.zero_grad()
        pi_loss.backward()
        self.proposal_optimizer.step()

        """
        -------- actor policy ------- 
        *GAC style 
        E_{a ~ pi_{actor}} [-ln pi_{actor}(a|s) - ln pi_{proposal}(a|s)]

        SPOT style
        E_{a ~ pi_{proposal}} [-Q{actor}(a|s) - ln pi_{proposal}(a|s)]
        """
        # action_batch, _, _, = self.bp.policy.sample(state_batch, self.n_action_proposals)
        # stacked_s_batch_full, stacked_s_batch, best_actions = self._get_best_actions(state_batch, action_batch)
        stacked_s_batch_full = state_batch.repeat_interleave(self.n_action_proposals, dim=0)
        with torch.no_grad():
            action_samples, _ = self.ac.pi.sample(stacked_s_batch_full)
        stacked_s_batch_full, stacked_s_batch, best_actions = self._get_best_actions(state_batch, stacked_s_batch_full, action_samples)

        # stacked_s_batch_full = stacked_s_batch_full.reshape(-1, self.state_dim)
        # action_samples = action_samples.reshape(-1, self.action_dim)
        # proposal_logprob = self.pp.policy.log_prob(stacked_s_batch_full, action_batch)
        with torch.no_grad():
            proposal_logprob = self.proposal.log_prob(stacked_s_batch, best_actions)
        # n_samples = int(self.rho * self.n_action_proposals)
        # # proposal_logprob = proposal_logprob.reshape(self.batch_size, n_samples, 1)
        # # proposal_logprob = -proposal_logprob[:, :, 0]
        # proposal_logprob = proposal_logprob.reshape(self.batch_size, n_samples)


        # actor_loss = self.bp.policy.log_prob(stacked_s_batch, best_actions)
        actor_loss = self.ac.pi.log_prob(stacked_s_batch, best_actions)
        # # actor_loss = actor_loss.reshape(self.batch_size, n_samples, 1)
        # actor_loss = actor_loss.reshape(self.batch_size, n_samples)
        # actor_loss = actor_loss.mean(axis=1)

        actor_loss = actor_loss + (proposal_logprob * self.alpha)
        actor_loss = -actor_loss.mean()
        self.pi_optimizer.zero_grad()
        actor_loss.backward()
        self.pi_optimizer.step()



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
