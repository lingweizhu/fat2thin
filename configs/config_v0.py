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

def c20240609():
    target_agents = ["TTT"]
    target_envs = ["Hopper"]
    target_datasets = ["medexp"]

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [1.0, 0.5, 0.1],
        " --q_lr_prob ": [1.],
        " --distribution ": ["Student"],
        " --fdiv_info ": ['gan 7'],
        " --info ": ["test_v0/test_TTT/actor_Student/fdiv_7_gan/"],
    }
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=0, line_per_file=1)

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [1.0, 0.5, 0.1],
        " --q_lr_prob ": [1.],
        " --distribution ": ["Student"],
        " --fdiv_info ": ['jensen_shannon 7'],
        " --info ": ["test_v0/test_TTT/actor_Student/fdiv_7_js/"],
    }
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=18, line_per_file=1)

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [1.0, 0.5, 0.1],
        " --q_lr_prob ": [1.],
        " --distribution ": ["Student"],
        " --fdiv_info ": ['jeffrey 7'],
        " --info ": ["test_v0/test_TTT/actor_Student/fdiv_7_jeffrey/"],
    }
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=36, line_per_file=1)

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [1.0, 0.5, 0.1],
        " --q_lr_prob ": [1.],
        " --distribution ": ["Student"],
        " --fdiv_info ": ['backwardkl 7'],
        " --info ": ["test_v0/test_TTT/actor_Student/fdiv_7_bKL/"],
    }
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=54, line_per_file=1)

    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.1],
    #     " --q_lr_prob ": [1.],
    #     " --distribution ": ["Student"],
    #     " --fdiv_info ": ['forwardkl 7'],
    #     " --info ": ["test_v0/test_TTT/actor_Student/fdiv_7_fKL/"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=6, line_per_file=1)
    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.1],
    #     " --q_lr_prob ": [1.],
    #     " --distribution ": ["SGaussian"],
    #     " --fdiv_info ": ['forwardkl 7'],
    #     " --info ": ["test_v0/test_TTT/actor_SGaussian/fdiv_7_fKL/"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=24, line_per_file=1)

    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.1],
    #     " --q_lr_prob ": [1.],
    #     " --distribution ": ["Student"],
    #     " --fdiv_info ": ['forwardkl 1'],
    #     " --info ": ["test_v0/actor_Student/fdiv_1"],
    #     # " --distribution ": ["SGaussian"],
    #     # " --info ": ["test_v0/test_TTT/actor_SGaussian/"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=42, line_per_file=1)
    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.1],
    #     " --q_lr_prob ": [1.],
    #     " --distribution ": ["SGaussian"],
    #     " --fdiv_info ": ['forwardkl 1'],
    #     " --info ": ["test_v0/test_TTT/actor_SGaussian/fdiv_1"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=60, line_per_file=1)

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
    # c20240607()
    c20240609()