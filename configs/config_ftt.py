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
    # target_envs = ["SimEnv3"]
    # target_datasets = ["random"]
    # prev_file = 0
    #
    # sweep = {
    #     " --rho ": [0.2],
    #     " --q_lr_prob ": [1.],
    #     " --gamma ": [0.9],
    #     " --timeout ": [24],
    #     " --actor_loss ": ["KL"],
    #     " --log_interval ": [400],
    #     " --max_steps ": [2000],
    #     # " --max_steps ": [5000],
    #     " --proposal_distribution ": ["HTqGaussian"],
    #     " --distribution ": ["qGaussian"],
    # }
    # sweep[" --info "] = ["policy_evolution/"]
    #
    # target_agents = ["FTT"]
    # sweep[" --pi_lr "] = [0.0003]
    # sweep[" --tau "] = [0.1]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["IQL"]
    # sweep[" --pi_lr "] = [3e-05]
    # sweep[" --expectile "] = [0.9]
    # sweep[" --tau "] = [1./3.]
    # sweep[" --distribution "] = ["SGaussian"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["XQL"]
    # sweep[" --pi_lr "] = [0.0001]
    # sweep[" --expectile "] = [0.8]
    # sweep[" --tau "] = [5.0]
    # sweep[" --distribution "] = ["SGaussian"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["SQL"]
    # sweep[" --pi_lr "] = [0.003]
    # sweep[" --expectile "] = [0.8]
    # sweep[" --tau "] = [5.0]
    # sweep[" --distribution "] = ["SGaussian"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["SPOT"]
    # sweep[" --pi_lr "] = [0.001]
    # sweep[" --expectile "] = [0.8]
    # sweep[" --tau "] = [0.05]
    # sweep[" --distribution "] = ["SGaussian"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=1, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    # target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    # target_datasets = ["medexp", "medium", "medrep"]
    target_envs = ["HalfCheetah"]
    target_datasets = ["medexp"]
    prev_file = 0

    sweep = {
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["KL"],
        " --log_interval ": [1000],
        " --max_steps ": [10000],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
    }
    sweep[" --info "] = ["policy_evolution/"]

    # target_agents = ["FTT"]
    # target_distributions = ["qGaussian"]
    # prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT, ftt_key="HTqG-KL",
    #                              num_runs=1, run_base=0, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["IQL"]
    # sweep[" --distribution "] = ["SGaussian"]
    # target_distributions = ["SGaussian"]
    # prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
    #                              num_runs=1, run_base=0, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["XQL"]
    # target_distributions = ["SGaussian"]
    # prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
    #                              num_runs=1, run_base=0, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # target_agents = ["SQL"]
    # target_distributions = ["SGaussian"]
    # prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
    #                              num_runs=1, run_base=0, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    #
    # # target_agents = ["SPOT"]
    # # target_distributions = ["SGaussian"]
    # # prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
    # #                              num_runs=1, run_base=0, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    target_agents = ["IQL"]
    sweep[" --distribution "] = ["Gaussian"]
    target_distributions = ["Gaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=1, run_base=0, comb_num_base=0, prev_file=prev_file, line_per_file=1)

def c20240902():
    # sweep = {
    #     " --rho ": [0.2],
    #     " --q_lr_prob ": [1.],
    #     " --gamma ": [0.9],
    #     " --timeout ": [24],
    #     " --actor_loss ": ["KL"],
    #     " --log_interval ": [100],
    #     " --max_steps ": [500],
    #     " --proposal_distribution ": ["HTqGaussian"],
    #     " --info ": ["policy_evolution_plot/"],
    # }
    #
    # target_envs = ['SimEnv3']
    # target_datasets = ['random']
    # target_agents = ['FTT']
    #
    # sweep[" --distributions "] = ['qGaussian']
    # sweep[" --agents "] = ['FTT']
    # sweep[" --show_proposal "] = [1]
    # # sweep[" --log_interval "] = [1000]
    # # sweep[" --max_steps "] = [5000]
    # sweep[" --load_network_paths "] = ['data/output/policy_evolution/SimEnv3/random/FTT/qGaussian/0_param/0_run/parameters/ ']
    # policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
    #                          num_runs=1, run_base=0, comb_num_base=0,
    #                          prev_file=0, line_per_file=1)
    #
    # sweep[" --distributions "] = ['qGaussian SGaussian SGaussian SGaussian SGaussian']
    # sweep[" --agents "] = ['FTT IQL XQL SQL SPOT']
    # sweep[" --show_proposal "] = [0]
    # # sweep[" --log_interval "] = [1000]
    # # sweep[" --max_steps "] = [5000]
    # sweep[" --load_network_paths "] = ['data/output/policy_evolution/SimEnv3/random/FTT/qGaussian/0_param/0_run/parameters/ '
    #                                    'data/output/policy_evolution/SimEnv3/random/IQL/SGaussian/0_param/0_run/parameters/ '
    #                                    'data/output/policy_evolution/SimEnv3/random/XQL/SGaussian/0_param/0_run/parameters/ '
    #                                    'data/output/policy_evolution/SimEnv3/random/SQL/SGaussian/0_param/0_run/parameters/ '
    #                                    'data/output/policy_evolution/SimEnv3/random/SPOT/SGaussian/0_param/0_run/parameters/ ']
    # policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
    #                          num_runs=1, run_base=0, comb_num_base=0,
    #                          prev_file=1, line_per_file=1)

    # sweep = {
    #     " --rho ": [0.2],
    #     " --q_lr_prob ": [1.],
    #     " --actor_loss ": ["KL"],
    #     # " --log_interval ": [100],
    #     # " --max_steps ": [1000],
    #     " --proposal_distribution ": ["HTqGaussian"],
    #     " --distribution ": ["qGaussian"],
    #     " --info ": ["policy_evolution_plot/"]
    # }
    # target_envs = ["HalfCheetah"]
    # target_datasets = ["medexp"]
    # target_agents = ["FTT"]
    #
    # sweep[" --distributions "] = ['qGaussian']
    # sweep[" --agents "] = ['FTT']
    # sweep[" --show_proposal "] = [1]
    # sweep[" --log_interval "] = [1000]
    # sweep[" --max_steps "] = [5000]
    # sweep[" --load_network_paths "] = ['data/output/policy_evolution/HalfCheetah/medexp/FTT/qGaussian/0_param/0_run/parameters/ ']
    # policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
    #                          num_runs=1, run_base=0, comb_num_base=0,
    #                          prev_file=2, line_per_file=1)

    sweep = {
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --gamma ": [0.99],
        " --timeout ": [1000],
        " --actor_loss ": ["KL"],
        " --log_interval ": [1000],
        " --max_steps ": [10000],
        " --proposal_distribution ": ["HTqGaussian"],
        " --info ": ["policy_evolution_plot/"],
    }

    target_envs = ['HalfCheetah']
    target_datasets = ['medexp']
    target_agents = ['FTT']

    sweep[" --distributions "] = ['qGaussian Gaussian']
    sweep[" --agents "] = ['FTT IQL']
    sweep[" --show_proposal "] = [0]
    # sweep[" --log_interval "] = [1000]
    # sweep[" --max_steps "] = [5000]
    sweep[" --load_network_paths "] = ['data/output/policy_evolution/HalfCheetah/medexp/FTT/qGaussian/0_param/0_run/parameters/ '
                                       'data/output/policy_evolution/HalfCheetah/medexp/IQL/Gaussian/1_param/0_run/parameters/ ']
    policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
                             num_runs=1, run_base=0, comb_num_base=0,
                             prev_file=1, line_per_file=1)

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
        " --info ": ["test_v1/baseline/"],
    }
    prev_file = 0
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, comb_num_base=0, prev_file=prev_file, line_per_file=1)

def c20240906():
    prev_file = 0
    run_base = 5

    target_agents = ["FTT"]
    target_envs = ["SimEnv3"]
    target_datasets = ["random"]

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
        " --info ": ["test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/"],
    }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    target_distributions = ["qGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="HTqG-KL", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)

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
        " --info ": ["test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/"],
    }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5, run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)
    target_distributions = ["qGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="SG-KL", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)


    target_agents = ["FTT"]
    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medium", "medrep"]

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [1.0, 0.5, 0.01],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["KL"],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
        " --info ": ["test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/"],
    }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5,run_base=run_base,
    #                               comb_num_base=0, prev_file=prev_file, line_per_file=1)
    target_distributions = ["qGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="HTqG-KL", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    sweep[" --actor_loss "] = ["SPOT"]
    sweep[" --info "] = ["test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5,run_base=run_base,
    #                               comb_num_base=0, prev_file=prev_file, line_per_file=1)
    target_distributions = ["qGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="HTqG-SPOT", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    sweep[" --actor_loss "] = ["KL"]
    sweep[" --proposal_distribution "] = ["SGaussian"]
    sweep[" --info "] = ["test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=5,run_base=run_base,
    #                               comb_num_base=0, prev_file=prev_file, line_per_file=1)
    target_distributions = ["qGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="SG-KL", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)

def c20240914():
    # target_agents = ["IQL", "InAC", "TAWAC", "AWAC", "TD3BC"]
    prev_file = 0

    target_envs = ["SimEnv3"]
    target_datasets = ["random"]

    target_agents = ["IQL"]
    sweep = {
        " --pi_lr ": [1e-3, 3e-4, 1e-4, 3e-3],
        " --q_lr_prob ": [1.],
        " --tau ": [1./3, 0.7],
        " --expectile ": [0.7, 0.8, 0.9],
        " --rho ": [0.2],
        " --gamma ": [0.9],
        " --timeout ": [24],
        " --log_interval ": [10000],
        " --max_steps ": [500000],
        " --distribution ": ["SGaussian"],
        " --info ": ["test_v1/baseline/"],
    }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
    #                               num_runs=5, comb_num_base=0, prev_file=prev_file,
    #                               line_per_file=1)
    run_base = 5
    target_distributions = ["SGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="HTqG-SPOT", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    target_agents = ["SQL", "XQL"]
    sweep = {
        " --pi_lr ": [1e-3, 3e-4, 1e-4, 3e-3],
        " --q_lr_prob ": [1.],
        " --tau ": [2., 5.],
        " --expectile ": [0.8],
        " --rho ": [0.2],
        " --gamma ": [0.9],
        " --timeout ": [24],
        " --log_interval ": [10000],
        " --max_steps ": [500000],
        " --distribution ": ["SGaussian"],
        " --info ": ["test_v1/baseline/"],
    }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
    #                               num_runs=5, comb_num_base=0, prev_file=prev_file,
    #                               line_per_file=1)
    run_base = 5
    target_distributions = ["SGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="HTqG-SPOT", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    # target_agents = ["SPOT"]
    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4, 1e-4, 3e-3],
    #     " --q_lr_prob ": [1.],
    #     " --tau ": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    #     " --rho ": [0.2],
    #     " --gamma ": [0.9],
    #     " --timeout ": [24],
    #     " --log_interval ": [10000],
    #     " --max_steps ": [500000],
    #     " --distribution ": ["SGaussian"],
    #     " --info ": ["test_v1/baseline/"],
    # }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
    #                               num_runs=5, comb_num_base=0, prev_file=prev_file,
    #                               line_per_file=1)

    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medium", "medrep"]
    target_agents = ["SQL"]
    sweep = {
        " --pi_lr ": [2e-4],
        " --q_lr_prob ": [1.],
        " --tau ": [2.0, 5.0],
        " --expectile ": [0.8],
        " --rho ": [0.2],
        " --distribution ": ["SGaussian"],
        " --info ": ["test_v1/baseline/"],
    }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
    #                               num_runs=5, comb_num_base=0, prev_file=prev_file,
    #                               line_per_file=1)
    run_base = 5
    target_distributions = ["SGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="HTqG-SPOT", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    target_agents = ["XQL"]
    sweep = {
        " --pi_lr ": [2e-4],
        " --q_lr_prob ": [1.],
        " --tau ": [2.0],
        " --expectile ": [0.8],
        " --rho ": [0.2],
        " --distribution ": ["SGaussian"],
        " --info ": ["test_v1/baseline/"],
    }
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
                                  num_runs=5, comb_num_base=0, prev_file=prev_file,
                                  line_per_file=1)
    run_base = 5
    target_distributions = ["SGaussian"]
    prev_file = add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                                 num_runs=5, ftt_key="HTqG-SPOT", run_base=run_base, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    # target_agents = ["SPOT"]
    # sweep = {
    #     " --pi_lr ": [3e-4],
    #     " --q_lr_prob ": [1.],
    #     " --tau ": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
    #     " --distribution ": ["SGaussian"],
    #     " --info ": ["test_v1/baseline/"],
    # }
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
    #                               num_runs=5, comb_num_base=0, prev_file=prev_file,
    #                               line_per_file=1)

def c20240916():
    sweep = {
        " --info ": ["test_v1_best"],
    }
    target_agents = ["FTT"]
    target_envs = ["SimEnv3"]
    target_datasets = ["random"]
    target_distributions = ["qGaussian"]
    add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                     num_runs=5, run_base=0, comb_num_base=0, prev_file=0, line_per_file=1)
    target_agents = ["IQL", "TAWAC"]
    target_envs = ["SimEnv3"]
    target_datasets = ["random"]
    target_distributions = ["SGaussian"]
    add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                     num_runs=5, run_base=0, comb_num_base=0, prev_file=5, line_per_file=1)

def c20240917():
    sweep = {
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --gamma ": [0.9],
        " --timeout ": [24],
        " --actor_loss ": ["KL"],
        " --log_interval ": [500000],
        " --max_steps ": [500000],
        " --proposal_distribution ": ["HTqGaussian"],
        " --info ": ["final_policy_plot/"],
    }

    target_envs = ['SimEnv3']
    target_datasets = ['random']
    target_agents = ['FTT']
    sweep[" --distributions "] = ['qGaussian SGaussian SGaussian SGaussian']
    sweep[" --agents "] = ['FTT IQL XQL SQL']
    # sweep[" --distributions "] = ['qGaussian SGaussian']
    # sweep[" --agents "] = ['FTT IQL']
    sweep[" --load_network_paths "] = ['data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/SimEnv3/random/FTT/qGaussian/3_param/0_run/parameters/ '
                                       'data/output/test_v1/baseline/SimEnv3/random/IQL/SGaussian/17_param/0_run/parameters/ '
                                       'data/output/test_v1/baseline/SimEnv3/random/XQL/SGaussian/5_param/0_run/parameters/ '
                                       'data/output/test_v1/baseline/SimEnv3/random/SQL/SGaussian/7_param/0_run/parameters/ '
                                       # 'data/output/test_v1/baseline/SimEnv3/random/SPOT/SGaussian/0_param/0_run/parameters/ '
                                       ]
    policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
                             num_runs=1, run_base=0, comb_num_base=0,
                             prev_file=0, line_per_file=1)
    sweep[" --distributions "] = ['qGaussian']
    sweep[" --agents "] = ['FTT']
    sweep[" --show_proposal "] = [1]
    sweep[" --load_network_paths "] = [
        'data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/SimEnv3/random/FTT/qGaussian/3_param/0_run/parameters/ '
        # 'data/output/policy_evolution/SimEnv3/random/FTT/qGaussian/0_param/0_run/parameters/ '
                                       ]
    policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
                             num_runs=1, run_base=0, comb_num_base=0,
                             prev_file=1, line_per_file=1)

    sweep[" --distributions "] = ['qGaussian']
    sweep[" --agents "] = ['FTT']
    sweep[" --show_proposal "] = [1]
    sweep[" --compare_distribution "] = [1]
    sweep[" --load_network_paths "] = [
        'data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/SimEnv3/random/FTT/qGaussian/3_param/0_run/parameters/ '
                                       ]
    policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
                             num_runs=1, run_base=0, comb_num_base=0,
                             prev_file=2, line_per_file=1)

def c20240918():
    sweep = {
        " --info ": ["final_policy"],
        " --proposal_distribution ": ["HTqGaussian"],
    }
    target_envs = ["HalfCheetah", "Hopper", "Walker2d"]
    target_datasets = ["medexp", "medium", "medrep"]
    target_agents = ["FTT"]
    target_distributions = ["qGaussian"]
    add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                     num_runs=3, run_base=0, comb_num_base=0, prev_file=0, line_per_file=1)

    target_agents = ["IQL", "InAC", "TAWAC", "AWAC", "TD3BC"]
    target_distributions = ["SGaussian"]
    add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT,
                     num_runs=3, run_base=0, comb_num_base=0, prev_file=27, line_per_file=1)

def c20240921():
    sweep = {
        " --log_interval ": [1000000],
        " --max_steps ": [1000000],
        " --proposal_distribution ": ["HTqGaussian"],
        " --info ": ["final_policy_plot/"],
    }

    target_envs = ['HalfCheetah']
    target_datasets = ['medexp']
    target_agents = ['FTT']

    sweep[" --distributions "] = ['qGaussian SGaussian SGaussian SGaussian']
    sweep[" --agents "] = ['FTT IQL TAWAC InAC']
    sweep[" --load_network_paths "] = ['data/output/final_policy/HalfCheetah/medexp/FTT/qGaussian/0_param/0_run/parameters/ '
                                       'data/output/final_policy/HalfCheetah/medexp/IQL/SGaussian/1_param/0_run/parameters/ '
                                       'data/output/final_policy/HalfCheetah/medexp/TAWAC/SGaussian/0_param/0_run/parameters/ '
                                       'data/output/final_policy/HalfCheetah/medexp/InAC/SGaussian/0_param/0_run/parameters/ '
                                       # 'data/output/final_policy/HalfCheetah/medexp/AWAC/SGaussian/1_param/0_run/parameters/ '
                                       # 'data/output/final_policy/HalfCheetah/medexp/TD3BC/SGaussian/1_param/0_run/parameters/ '
                                       ]
    policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
                             num_runs=1, run_base=0, comb_num_base=0,
                             prev_file=0, line_per_file=1)

    sweep[" --distributions "] = ['qGaussian']
    sweep[" --agents "] = ['FTT']
    sweep[" --show_proposal "] = [1]
    sweep[" --load_network_paths "] = ['data/output/final_policy/HalfCheetah/medexp/FTT/qGaussian/0_param/0_run/parameters/ ']
    policy_evolution_scripts(sweep, target_agents, target_envs, target_datasets,
                             num_runs=1, run_base=0, comb_num_base=0,
                             prev_file=1, line_per_file=1)

def c20241115():
    target_envs = ["HalfCheetah", "Hopper"]
    target_datasets = ["medexp"]
    prev_file = 0

    sweep = {
        " --exp_type ": ['time'],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["KL"],
        " --log_interval ": [100000],
        " --max_steps ": [100000],
        " --proposal_distribution ": ["HTqGaussian"],
        " --distribution ": ["qGaussian"],
    }
    sweep[" --info "] = ["sampling_time/"]

    target_agents = ["IQL"]
    sweep[" --distribution "] = ["Gaussian"]
    sweep[" --proposal_distribution "] = ["Gaussian"]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
                                  num_runs=1, comb_num_base=0, prev_file=prev_file,
                                  line_per_file=1)

    target_agents = ["IQL"]
    sweep[" --distribution "] = ["HTqGaussian"]
    sweep[" --proposal_distribution "] = ["HTqGaussian"]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
                                  num_runs=1, comb_num_base=0, prev_file=prev_file,
                                  line_per_file=1)

    target_agents = ["IQL"]
    sweep[" --distribution "] = ["qGaussian"]
    sweep[" --proposal_distribution "] = ["qGaussian"]
    prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
                                  num_runs=1, comb_num_base=0, prev_file=prev_file,
                                  line_per_file=1)


if __name__ == "__main__":
    # c20240906() # main results
    # c20240914() # baseline
    # c20240829() # policy evolution
    c20240902() # policy evolution plot
    # c20240916() # simEnv3 save final policy
    # c20240917() # simEnv3 final policy plot
    # c20240918() # D4RL save final policy
    # c20240921() # D4RL final policy plot
    # c20241115() # network sampling time`