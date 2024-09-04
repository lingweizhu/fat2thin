import copy
import itertools
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
    target_agents = ["FTT"]
    target_envs = ["Hopper"]
    target_datasets = ["medexp"]

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [0.5],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["MSE"],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
        " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_MSE/"],
    }
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=0, line_per_file=1)

    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [0.5],
    #     " --q_lr_prob ": [1.],
    #     " --actor_loss ": ["KL"],
    #     " --proposal_distribution ": ["HTqGaussian"],
    #     " --distribution ": ["qGaussian"],
    #     " --info ": ["test_v0/proposal_HTqG_actor_qG_actorloss_KL/"],
    #     # " --distribution ": ["SGaussian"],
    #     # " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_SG_actorloss_KL/"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=6, line_per_file=1)
    #
    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.1],
    #     " --rho ": [0.2],
    #     " --q_lr_prob ": [1.],
    #     " --actor_loss ": ["GAC"],
    #     " --proposal_distribution ": ["HTqGaussian"],
    #     " --distribution ": ["qGaussian"],
    #     " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_GAC/"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=12, line_per_file=1)
    #
    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.1],
    #     " --q_lr_prob ": [1.],
    #     " --actor_loss ": ["SPOT"],
    #     " --proposal_distribution ": ["HTqGaussian"],
    #     " --distribution ": ["qGaussian"],
    #     " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=30, line_per_file=1)
    #
    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.1],
    #     " --q_lr_prob ": [1.],
    #     " --actor_loss ": ["TAWAC"],
    #     " --proposal_distribution ": ["HTqGaussian"],
    #     " --distribution ": ["qGaussian"],
    #     " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_TAWAC/"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=48, line_per_file=1)

def c20240617():
    target_agents = ["FTT"]
    target_envs = ["Hopper"]
    target_datasets = ["medexp"]

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [0.5],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["CopyMu-Proposal"],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
        " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_CopyMu-Proposal/"],
    }
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=0, line_per_file=1)

    sweep[" --actor_loss "] = ["CopyMu-Pi"]
    sweep[" --info "] = ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_CopyMu-Pi/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=6, line_per_file=1)

def c20240710():
    target_agents = ["FTT"]
    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medium", "medrep"]

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [1.0, 0.5, 0.01],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["CopyMu-Pi"],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
        " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_CopyMu-Pi/"],
    }
    prev_file = 0
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)

def c20240725():
    target_agents = ["FTT"]
    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medium", "medrep"]
    prev_file = 0

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [1.0, 0.5, 0.01],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["SPOT"],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
        " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/"],
    }
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    sweep[" --actor_loss "] = ["KL"]
    sweep[" --info "] = ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/"]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)

def c20240821():
    target_agents = ["FTT"]
    target_envs = ["SimEnv3"]
    target_datasets = ["random"]
    prev_file = 0

    sweep = {
        " --pi_lr ": [3e-4, 1e-4, 3e-5],
        " --tau ": [1.0, 0.5, 0.1, 0.01],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --gamma ": [0.9],
        " --timeout ": [24],
        " --actor_loss ": ["KL"],
        " --log_interval ": [10000],
        " --max_steps ": [500000],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
        " --info ": ["test_v0/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/"],
    }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["TAWAC"]
    # sweep[" --distribution "] = ["HTqGaussian", "SGaussian"]
    # sweep[" --info "] = ["test_v0/baseline/"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["IQL"]
    # sweep[" --expectile "] = [0.7, 0.9]
    # sweep[" --tau "] = [0.1, 1./3.]
    # sweep[" --distribution "] = ["HTqGaussian", "SGaussian"]
    # sweep[" --info "] = ["test_v0/baseline/"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    sweep = {
        " --pi_lr ": [3e-4, 1e-4, 3e-5],
        " --tau ": [1.0, 0.5, 0.1, 0.01],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --gamma ": [0.9],
        " --timeout ": [24],
        " --actor_loss ": ["KL"],
        " --log_interval ": [10000],
        " --max_steps ": [500000],
        " --proposal_distribution ": ["SGaussian"],
        " --distribution ": ["qGaussian"],
        " --info ": ["test_v0/test_FTT/proposal_SG_actor_qG_actorloss_KL/"],
    }
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)


def c20240829():
    target_envs = ["SimEnv3"]
    target_datasets = ["random"]
    prev_file = 0

    sweep = {
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --gamma ": [0.9],
        " --timeout ": [24],
        " --actor_loss ": ["KL"],
        " --log_interval ": [100],
        # " --max_steps ": [500000],
        " --max_steps ": [500],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
    }
    sweep[" --info "] = ["policy_evolution/"]

    target_agents = ["FTT"]
    sweep[" --pi_lr "] = [0.0003]
    sweep[" --tau "] = [0.1]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    target_agents = ["FTT"]
    sweep[" --pi_lr "] = [0.0003]
    sweep[" --tau "] = [0.1]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    target_agents = ["IQL"]
    sweep[" --pi_lr "] = [3e-05]
    sweep[" --expectile "] = [0.9]
    sweep[" --tau "] = [1./3.]
    sweep[" --distribution "] = ["SGaussian"]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    target_agents = ["TAWAC"]
    sweep[" --pi_lr "] = [3e-05]
    sweep[" --tau "] = [0.5]
    sweep[" --distribution "] = ["SGaussian"]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    target_agents = ["IQL"]
    sweep[" --pi_lr "] = [0.0003]
    sweep[" --expectile "] = [0.7]
    sweep[" --tau "] = [0.1]
    sweep[" --distribution "] = ["HTqGaussian"]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    target_agents = ["TAWAC"]
    sweep[" --pi_lr "] = [0.0003]
    sweep[" --tau "] = [0.01]
    sweep[" --distribution "] = ["HTqGaussian"]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)


def c20240902():
    sweep = {
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --gamma ": [0.9],
        " --timeout ": [24],
        " --actor_loss ": ["KL"],
        " --log_interval ": [100],
        " --max_steps ": [500],
        " --proposal_distribution ": ["HTqGaussian"],
        " --info ": ["policy_evolution_plot/"],
    }

    target_envs = ['SimEnv3']
    target_datasets = ['random']
    target_agents = ['FTT']
    sweep[" --distributions "] = ['qGaussian HTqGaussian SGaussian']
    sweep[" --agents "] = ['FTT IQL IQL']
    sweep[" --load_network_paths "] = ['data/output/policy_evolution/SimEnv3/random/FTT/qGaussian/0_param/0_run/parameters/ '
                                       'data/output/policy_evolution/SimEnv3/random/IQL/HTqGaussian/0_param/0_run/parameters/ '
                                       'data/output/policy_evolution/SimEnv3/random/IQL/SGaussian/0_param/0_run/parameters/ '
                                       ]
    # sweep[" --distributions "] = ['qGaussian HTqGaussian HTqGaussian SGaussian SGaussian']
    # sweep[" --agents "] = ['FTT IQL TAWAC IQL TAWAC']
    # sweep[" --load_network_paths "] = ['data/output/policy_evolution/SimEnv3/random/FTT/qGaussian/0_param/0_run/parameters/ '
    #                                    'data/output/policy_evolution/SimEnv3/random/IQL/HTqGaussian/0_param/0_run/parameters/ '
    #                                    'data/output/policy_evolution/SimEnv3/random/TAWAC/HTqGaussian/0_param/0_run/parameters/ '
    #                                    'data/output/policy_evolution/SimEnv3/random/IQL/SGaussian/0_param/0_run/parameters/ '
    #                                    'data/output/policy_evolution/SimEnv3/random/TAWAC/SGaussian/0_param/0_run/parameters/ ']
    policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
                             num_runs=1, run_base=0, comb_num_base=0,
                             prev_file=0, line_per_file=1)

def c20240903():
    target_agents = ["IQL", "InAC", "TAWAC", "AWAC", "TD3BC"]
    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medium", "medrep"]

    sweep = {
        " --pi_lr ": [1e-3, 3e-4, 1e-4, 3e-3],
        " --q_lr_prob ": [1.],
        " --tau ": [1./3.],
        " --expectile ": [0.7],
        " --rho ": [0.2],
        " --distribution ": ["SGaussian", "Gaussian"],
        " --info ": ["test_v0/baseline/"],
    }
    prev_file = 0
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)


if __name__ == "__main__":
    c20240903()