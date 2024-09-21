import pickle
import time
import copy
import numpy as np

import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl
import gzip
import torch
import matplotlib.pyplot as plt
from core.utils import torch_utils
from mpl_toolkits.mplot3d.axes3d import Axes3D
import core.environment.env_factory as environment

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "Helvetica",
})



def load_testset(env_name, dataset, id, cfg):
    path = None
    if env_name == 'HalfCheetah':
        if dataset == 'expert':
            path = {"env": "halfcheetah-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "halfcheetah-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "halfcheetah-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "halfcheetah-medium-replay-v2"}
    elif env_name == 'Walker2d':
        if dataset == 'expert':
            path = {"env": "walker2d-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "walker2d-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "walker2d-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "walker2d-medium-replay-v2"}
    elif env_name == 'Hopper':
        if dataset == 'expert':
            path = {"env": "hopper-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "hopper-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "hopper-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "hopper-medium-replay-v2"}
    elif env_name == 'Ant':
        if dataset == 'expert':
            path = {"env": "ant-expert-v2"}
        elif dataset == 'medexp':
            path = {"env": "ant-medium-expert-v2"}
        elif dataset == 'medium':
            path = {"env": "ant-medium-v2"}
        elif dataset == 'medrep':
            path = {"env": "ant-medium-replay-v2"}
    
    elif env_name == 'Acrobot':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/acrobot/transitions_50k/train_40k/{}_run.pkl".format(id)}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/acrobot/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    elif env_name == 'LunarLander':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/lunar_lander/transitions_50k/train_500k/{}_run.pkl".format(id)}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/lunar_lander/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    elif env_name == 'MountainCar':
        if dataset == 'expert':
            path = {"pkl": "data/dataset/mountain_car/transitions_50k/train_60k/{}_run.pkl".format(id)}
        elif dataset == 'mixed':
            path = {"pkl": "data/dataset/mountain_car/transitions_50k/train_mixed/{}_run.pkl".format(id)}
    elif env_name == 'SimEnv3':
        path = {"generate": None}

    assert path is not None

    testsets = {}
    for name in path:
        if name == "env":
            env = gym.make(path['env'])
            try:
                data = env.get_dataset()
            except:
                env = env.unwrapped
                data = env.get_dataset()
            testsets[name] = {
                'states': data['observations'],
                'actions': data['actions'],
                'rewards': data['rewards'],
                'next_states': data['next_observations'],
                'terminations': data['terminals'],
            }
        elif name == "pkl":
            pth = path[name]
            with open(pth.format(id), 'rb') as f:
                testsets[name] = pickle.load(f)
        elif name == "generate":
            env = environment.EnvFactory.create_env_fn(cfg)()
            testsets[name] = env.get_dataset()
        else:
            raise NotImplementedError
        return testsets
    else:
        return {}

def run_steps(agent, max_steps, log_interval, eval_pth):
    t0 = time.time()
    evaluations = []
    agent.populate_returns(initialize=True)
    while True:
        if log_interval and not agent.total_steps % log_interval:
            # agent.save(timestamp="_{}".format(agent.total_steps))
            mean, median, min_, max_ = agent.log_file(elapsed_time=log_interval / (time.time() - t0), test=True)
            evaluations.append(mean)
            t0 = time.time()
        if max_steps and agent.total_steps >= max_steps:
            agent.save(timestamp="_{}".format(agent.total_steps))
            break
        agent.step()
    # agent.save(timestamp="_{}".format(agent.total_steps))
    np.save(eval_pth+"/evaluations.npy", np.array(evaluations))

def policy_evolution(cfg, agent, num_samples=500):
    from plot.plot_v0 import formal_distribution_name
    # rng = np.random.RandomState(1024)
    # rnd_idx = rng.choice(agent.offline_data['env']['states'].shape[0], size=10)
    # rnd_states = agent.offline_data['env']['states'][rnd_idx]
    # states = torch_utils.tensor(agent.state_normalizer(rnd_states), agent.device)

    init_state = agent.env.reset().reshape((1, -1))
    state = torch_utils.tensor(agent.state_normalizer(init_state), agent.device)
    state = state.repeat_interleave(num_samples, dim=0)
    with torch.no_grad():
        on_policy, _ = agent.ac.pi.sample(state)
        on_policy = on_policy.cpu().detach().numpy()
    for i in range(0, agent.action_dim):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300, subplot_kw={'projection': '3d'})
        zeros0 = np.zeros((num_samples, i))
        zeros1 = np.zeros((num_samples, cfg.action_dim - 1 - i))
        # zeros0 = on_policy[:, :i]
        # zeros1 = on_policy[:, i+1:]
        xs = np.linspace(-1, 1, num=num_samples).reshape((num_samples, 1))
        actions = np.concatenate([zeros0, xs, zeros1], axis=1)
        actions = torch_utils.tensor(actions, agent.device)
        # time_color = ["#C2E4EF", "#98CAE1", "#6EA6CD", "#4A7BB7", "#364B9A"]
        time_color = {
            "HTqGaussian": ["#729bba", "#5192c4", "#2d81c2", "#1069ad", "#014f8a"],
            "qGaussian": ["#FB9A29", "#EC7014", "#CC4C02", "#993404", "#662506"],
            "Gaussian": ["#bf8888", "#b85f5f", "#b53c3c", "#a81e1e", "#8c0303"],
        }

        xticks = [-1, -0.5, 0, 0.5, 1]
        dfs = []
        plot_xs = []
        plot_ys = []
        plot_gys = []
        # timestamps = list(range(0, 250000, 50000))
        timestamps = list(range(0, 500, 100))

        for idx, timestamp in enumerate(timestamps):
            agent.load(cfg.load_network, "_{}".format(int(timestamp)))
            # with torch.no_grad():
            #     a, _ = agent.ac.pi.sample(rnd_state, deterministic=False)
            # a = torch_utils.to_np(a)
            with torch.no_grad():
                dist, mean, shape, dfx = agent.ac.pi.distribution(state, dim=i)
                density = torch.exp(dist.log_prob(actions[:, i:i+1])).detach().cpu().numpy()
            # ax.plot(xs.flatten(), ys, zs=idx, zdir='y', color=time_color[idx], alpha=0.5)
            plot_xs.append(xs.flatten())
            plot_ys.append(density.flatten())

            assert mean.shape[1] == 1
            normal = torch.distributions.Normal(mean, shape)
            normal_density = torch.exp(normal.log_prob(actions[:, i:i + 1])).detach().cpu().numpy()
            # ax.plot(xs.flatten(), normal_density.flatten(), zs=idx, zdir='y', color=time_color[idx], alpha=0.4, linestyle='--')
            plot_gys.append(normal_density.flatten())

        plot_ys = np.asarray(plot_ys)
        plot_gys = np.asarray(plot_gys)
        # ysmin, ysmax = plot_ys.min(), plot_ys.max()
        # gysmin, gysmax = plot_gys.min(), plot_gys.max()
        for idx, timestamp in enumerate(timestamps):
            xs, ys, gys = plot_xs[idx], plot_ys[idx], plot_gys[idx]
            # # ys = (ys - min(ysmin, gysmin)) / (max(ysmax, gysmax) - min(ysmin, gysmin))
            # # gys = (gys - min(ysmin, gysmin)) / (max(ysmax, gysmax) - min(ysmin, gysmin))
            # ys = ys / ys.max()
            # gys = gys / gys.max()

            largers = np.argpartition(gys, -1)[-1:]
            gys[largers] = np.nan

            ax.plot(xs, ys, zs=idx, zdir='y', color=time_color[cfg.distribution][idx], alpha=0.8)
            ax.plot(xs, gys, zs=idx, zdir='y', color=time_color["Gaussian"][idx], alpha=0.8, linestyle='--')
        ax.plot([], [], c=time_color[cfg.distribution][-1], linestyle='-', label=formal_distribution_name[cfg.distribution], alpha=0.8)
        ax.plot([], [], c=time_color["Gaussian"][-1], linestyle='--', label="Gaussian", alpha=0.8)


        trail = "st" if i == 0 else "nd" if i == 1 else "rd" if i == 2 else "th"

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=0)
        ax.set_zlim(0, 1.)
        ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticklabels(['0', '', '0.5', '', '1'], rotation=0, ha='center')
        ax.set_yticks(np.arange(len(timestamps)))
        ax.set_yticklabels(["{}".format(t//100) for t in timestamps], rotation=-20, ha='center', va='center')
        ax.set_ylabel(r'Steps (x$10^2$)')
        ax.set_zlabel(r'Density')
        ax.set_xlabel(r'Action')

        ax.set_title("{} Policy Evolution\n{}{} Dimension".format(formal_distribution_name[cfg.distribution], i + 1, trail))
        if cfg.distribution == "qGaussian":
            ax.annotate('', xytext=(0.81, -0.01), xy=(1.065, 0.25), xycoords='axes fraction',
                        arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
        elif cfg.distribution == "HTqGaussian":
            # ax.view_init(elev=0, azim=-40, roll=0)
            # ax.annotate('', xytext=(0.65, 0.135), xy=(1.02, 0.155), xycoords='axes fraction',
            #             arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
            # ax.view_init(elev=20, azim=-20, roll=0)
            # ax.annotate('', xytext=(0.51, -0.01), xy=(0.9, 0.05), xycoords='axes fraction',
            #             arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
            ax.view_init(elev=20, azim=-20, roll=0)
            ax.annotate('', xytext=(0.51, -0.01), xy=(0.85, 0.05), xycoords='axes fraction',
                        arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
        plt.legend(loc='lower left', bbox_to_anchor=(-0.3, 0.83), prop={'size': 10}, ncol=1, frameon=False)
        plt.tight_layout()
        plt.savefig(cfg.exp_path+"/{}_vis_dim{}.png".format(cfg.distribution, i), dpi=300)
        # plt.show()

def policy_evolution_multipolicy(cfg, agent_objs, time_color, num_samples=500, alpha=0.8):
    from plot.utils import formal_distribution_name, formal_dataset_name
    if cfg.env_name == "SimEnv3" and cfg.log_interval == cfg.max_steps:
        simenv3_final_policy_samples(cfg, agent_objs, time_color, num_samples, alpha)
        return

    init_state = agent_objs[0].env.reset().reshape((1, -1))
    state = torch_utils.tensor(agent_objs[0].state_normalizer(init_state), agent_objs[0].device)
    state = state.repeat_interleave(num_samples, dim=0)

    for i in range(0, agent_objs[0].action_dim):
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4), dpi=300, subplot_kw={'projection': '3d'})
        zeros0 = np.zeros((num_samples, i))
        zeros1 = np.zeros((num_samples, cfg.action_dim - 1 - i))
        xs = np.linspace(-1, 1, num=num_samples).reshape((num_samples, 1))
        actions = np.concatenate([zeros0, xs, zeros1], axis=1)
        actions = torch_utils.tensor(actions, agent_objs[0].device)

        xticks = [-1, -0.5, 0, 0.5, 1]
        plot_ys = {}
        for agent in agent_objs:
            plot_ys[agent.cfg.distribution] = []
        timestamps = list(range(0, 500, 100))

        for idx, timestamp in enumerate(timestamps):
            for agent in agent_objs:
                agent.load(agent.cfg.load_network, "_{}".format(int(timestamp)))
                with torch.no_grad():
                    dist, mean, shape, dfx = agent.ac.pi.distribution(state, dim=i)
                    density = torch.exp(dist.log_prob(actions[:, i:i+1])).detach().cpu().numpy()
                # print(agent.cfg.agent, agent.cfg.distribution, agent.cfg.load_network, mean.mean(), density.mean())
                plot_ys[agent.cfg.distribution].append(density.flatten())
            # print()
        for idx, timestamp in enumerate(timestamps):
            for agent in agent_objs:
                plot_ys_dist = np.asarray(plot_ys[agent.cfg.distribution])
                ys = np.asarray(plot_ys[agent.cfg.distribution][idx])
                if cfg.density_normalization:
                    ys = (ys - plot_ys_dist.min()) / (plot_ys_dist.max() - plot_ys_dist.min())
                ax.plot(xs.flatten(), ys, zs=idx, zdir='y', color=time_color["{} {}".format(agent.cfg.agent, agent.cfg.distribution)][idx], alpha=alpha, zorder=len(timestamps)-idx)
        for agent in agent_objs:
            ax.plot([], [], color=time_color["{} {}".format(agent.cfg.agent, agent.cfg.distribution)][idx], linestyle='-',
                    label="{} {}".format(agent.cfg.agent, formal_distribution_name[agent.cfg.distribution]), alpha=alpha)

        trail = "st" if i == 0 else "nd" if i == 1 else "rd" if i == 2 else "th"

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=0)
        ax.set_zlim(0, 1.)
        ax.set_zticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_zticklabels(['0', '', '0.5', '', '1'], rotation=0, ha='center')
        ax.set_yticks(np.arange(len(timestamps)))
        ax.set_yticklabels(["{}".format(t//100) for t in timestamps], rotation=-20, ha='center', va='center')
        ax.set_ylabel(r'Steps (x$10^2$)')
        ax.set_zlabel(r'Density')
        ax.set_xlabel(r'Action')
        plt.subplots_adjust(left=-0.1, bottom=None, right=0.9, top=None, wspace=None, hspace=None)
        fig.text(0.17, 0.92, "Policy Evolution on {} {}".format(cfg.env_name, formal_dataset_name[cfg.dataset]), fontsize=12)
        # fig.text(0.4, 0.9, "{}{} Dimension".format(i + 1, trail), fontsize=12)
        ax.view_init(elev=30, azim=-30, roll=0)
        ax.annotate('', xytext=(0.61, 0.09), xy=(0.95, 0.2), xycoords='axes fraction',
                    arrowprops=dict(edgecolor='None', facecolor='grey', alpha=0.3, width=8))
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1.5]))
        ax.grid(False)
        plt.legend(loc='lower left', bbox_to_anchor=(-0, 0.75), prop={'size': 10}, ncol=1, frameon=False)
        plt.tight_layout()
        plt.savefig(cfg.exp_path+"/vis_dim{}.png".format(i), dpi=300)

def simenv3_final_policy_samples(cfg, agent_objs, time_color, num_samples=500, alpha=0.8):
    from plot.utils import formal_distribution_name, formal_dataset_name
    state = agent_objs[0].env.reset()
    for _ in range(10):
        a = np.random.uniform(low=-1, high=1, size=1)
        state, _, done, _ = agent_objs[0].env.step([a])
        assert not done
    state = state.reshape((1, -1))
    state = torch_utils.tensor(agent_objs[0].state_normalizer(state), agent_objs[0].device)
    repeated_state = state.repeat_interleave(num_samples, dim=0)
    xs = np.linspace(-3, 3, num=num_samples).reshape((num_samples, 1))
    # next_s = [agent_objs[0].env.get_state(agent_objs[0].env.last_mu, agent_objs[0].env.cor, a)[1]
    #           for a in xs]
    # rewards = [agent_objs[0].env.get_reward(ns, a)[0] for ns, a in zip(next_s, xs)]
    for i in range(0, agent_objs[0].action_dim):
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
        # ax.plot(xs, rewards, c='black')
        zeros0 = np.zeros((num_samples, i))
        zeros1 = np.zeros((num_samples, cfg.action_dim - 1 - i))
        actions = np.concatenate([zeros0, xs, zeros1], axis=1)
        actions = torch_utils.tensor(actions, agent_objs[0].device)

        plot_ys = {}
        for agent in agent_objs:
            agent.load(agent.cfg.load_network, "_{}".format(int(cfg.log_interval)))
            with torch.no_grad():
                dist, mean, shape, dfx = agent.ac.pi.distribution(state, dim=i)
                density = torch.exp(dist.log_prob(actions[:, i:i+1])).detach().cpu().numpy()
            plot_ys[agent.cfg.distribution] = density.flatten()

        for agent in agent_objs:
            plot_ys_dist = np.asarray(plot_ys[agent.cfg.distribution])
            ys = np.asarray(plot_ys[agent.cfg.distribution])
            # if cfg.density_normalization:
            #     ys = (ys - plot_ys_dist.min()) / (plot_ys_dist.max() - plot_ys_dist.min())
            ax.plot(xs.flatten(), ys, color=time_color["{} {}".format(agent.cfg.agent, agent.cfg.distribution)][-1], alpha=alpha)
        for agent in agent_objs:
            ax.plot([], [], color=time_color["{} {}".format(agent.cfg.agent, agent.cfg.distribution)][-1], linestyle='-',
                    label="{} {}".format(agent.cfg.agent, formal_distribution_name[agent.cfg.distribution]), alpha=alpha)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(r'Density')
        ax.set_xlabel(r'Action')
        plt.subplots_adjust(top=0.8, left=0.2, bottom=0.2)
        fig.text(0.17, 0.92, "Final Policy on {} {}".format(cfg.env_name, formal_dataset_name[cfg.dataset]), fontsize=10)
        plt.legend(loc='lower left', bbox_to_anchor=(0, 0.94), prop={'size': 8}, ncol=1, frameon=False)
        # plt.tight_layout()
        plt.savefig(cfg.exp_path+"/vis_final_dim{}.png".format(i), dpi=300)
