import copy

import numpy as np
from scipy.stats import random_correlation
from core.utils.torch_utils import random_seed


class SimBase:
    def __init__(self, seed=np.random.randint(int(1e5))):
        # Use gamma = 0.9
        random_seed(seed)
        self.rng = np.random.RandomState(seed)
        self.state_dim = None
        self.action_dim = None

    def reset(self):
        raise NotImplementedError

    def step(self, a):
        raise NotImplementedError

    def get_dataset(self, timeout=24, num_traj=50):
        dataset = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "terminations": []
        }
        for traj in range(num_traj):
            st = self.reset()
            for t in range(timeout):
                a = self.rng.uniform(low=-1, high=1, size=self.action_dim)
                stp1, rwd, termin, _ = self.step([a])
                dataset["states"].append(copy.deepcopy(st))
                dataset["actions"].append(copy.deepcopy(a))
                dataset["rewards"].append(copy.deepcopy(rwd))
                dataset["next_states"].append(copy.deepcopy(stp1))
                dataset["terminations"].append(copy.deepcopy(termin))
                st = stp1
        return dataset


class SimEnv3(SimBase):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(SimEnv3, self).__init__(seed)
        self.corr = None
        self.last_mu = None
        self.state_dim = 8
        self.action_dim = 1

    def reset(self):
        self.last_mu = self.rng.random(self.state_dim) * 2 - 1
        # self.last_mu = np.zeros(self.state_dim)
        # self.cov = random_correlation.rvs(np.ones(self.state_dim), random_state=self.rng) / 2.

        rnd = self.rng.random((self.state_dim, self.state_dim))
        self.cor = np.corrcoef(rnd, rowvar=False) / 2.
        state = self.rng.multivariate_normal(self.last_mu, self.cor)
        return np.asarray(state)

    def step(self, a):
        a = a[0][0] * 100. # index and denormalization

        # norminator1 = np.exp(a / 100. + self.last_mu[:4]) - np.exp(- (a/100. + self.last_mu[:4]))
        # denorminator1 = np.exp(a / 100. + self.last_mu[:4]) + np.exp(- (a/100. + self.last_mu[:4]))
        # self.last_mu[:4] = norminator1 / denorminator1
        # norminator2 = np.exp(-a / 100. + self.last_mu[4:]) - np.exp(- (-a/100. + self.last_mu[4:]))
        # denorminator2 = np.exp(-a / 100. + self.last_mu[4:]) + np.exp(- (-a/100. + self.last_mu[4:]))
        # self.last_mu[4:] = norminator2 / denorminator2

        # state = self.rng.multivariate_normal(self.last_mu, self.cor)
        self.last_mu, state = self.get_state(self.last_mu, self.cor, a)

        # reward = - np.exp(state[0]/2. + state[4]/2) * ((a/100.)**2) + \
        #     2. * (state[1] + state[2] + state[5] + state[6] + 0.5) * a/100. + \
        #     state[3] + state[7]
        reward = self.get_reward(state, a)
        # reward /= 10.
        done = False
        info = {}
        return np.asarray(state), np.asarray(reward), np.asarray(done), info

    def get_state(self, last_mu, cor, a):
        mu = copy.deepcopy(last_mu)
        norminator1 = np.exp(a / 100. + self.last_mu[:4]) - np.exp(- (a/100. + self.last_mu[:4]))
        denorminator1 = np.exp(a / 100. + self.last_mu[:4]) + np.exp(- (a/100. + self.last_mu[:4]))
        mu[:4] = norminator1 / denorminator1
        norminator2 = np.exp(-a / 100. + self.last_mu[4:]) - np.exp(- (-a/100. + self.last_mu[4:]))
        denorminator2 = np.exp(-a / 100. + self.last_mu[4:]) + np.exp(- (-a/100. + self.last_mu[4:]))
        mu[4:] = norminator2 / denorminator2
        state = self.rng.multivariate_normal(mu, cor)
        return mu, state

    def get_reward(self, state, a):
        reward = - np.exp(state[0]/2. + state[4]/2) * ((a/100.)**2) + \
            2. * (state[1] + state[2] + state[5] + state[6] + 0.5) * a/100. + \
            state[3] + state[7]
        return reward