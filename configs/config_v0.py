import copy
from best_setting import BEST_AGENT
from utils import write_job_scripts, add_seed_scripts, add_param_scripts, policy_evolution_scripts

def c20240529():
    sweep = {
        " --pi_lr ": [1e-3, 3e-4, 1e-4],
        " --tau ": [0.01, 0.1, 0.3, 0.5],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --info ": ["test_v0/"],
    }
    target_agents = ["FTT_AWAC"]
    target_envs = ["Hopper"]
    target_datasets = ["expert", "medexp"]
    target_distributions = ["qGaussian"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, num_runs=5, comb_num_base=0, prev_file=0, line_per_file=1)

def c20240607():
    sweep = {
        " --pi_lr ": [3e-4],
        " --tau ": [0.01, 0.1, 0.3, 0.5, 1.0],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["KL"],
        " --proposal_distribution ": ["HTqGaussian"],
        " --info ": ["test_v0/proposal_HTqG_actor_qG_actorloss_KL/"],
    }
    target_agents = ["FTT"]
    target_envs = ["Hopper"]
    target_datasets = ["expert", "medexp"]
    target_distributions = ["qGaussian"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, num_runs=5, comb_num_base=0, prev_file=0, line_per_file=1)

def all():
    sweep = {
        " --pi_lr ": [1e-3, 3e-4, 1e-4],
        " --tau ": [0.01, 0.1, 0.3, 0.5],
        " --q_lr_prob ": [1.],
        " --info ": ["test_v0"],
    }
    target_agents = ["FTT"]
    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medrep", "medium"]
    target_distributions = ["qGaussian"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, num_runs=5, comb_num_base=0, prev_file=0, line_per_file=1)


if __name__ == "__main__":
    c20240529()