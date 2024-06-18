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

def c20240609():
    target_agents = ["TTT"]
    target_envs = ["Hopper"]
    target_datasets = ["medexp"]

    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.1],
    #     " --q_lr_prob ": [1.],
    #     " --distribution ": ["Student"],
    #     " --fdiv_info ": ['gan 7'],
    #     " --info ": ["test_v0/test_TTT/actor_Student/fdiv_7_gan/"],
    # }
    # write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=0, line_per_file=1)

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

def c20240613():
    target_agents = ["TTT"]
    target_envs = ["Ant"]
    target_datasets = ["expert"]

    sweep = {
        " --pi_lr ": [3e-4, 1e-4],
        " --tau ": [1.0],
        " --q_lr_prob ": [1.],
        " --distribution ": ["Student", "HTqGaussian"],
        " --max_steps ": [100000],
        " --log_interval ": [1000]
    }
    sweep[" --fdiv_info "] = ['jensen_shannon 3']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_3_js/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=0, line_per_file=1)
    sweep[" --fdiv_info "] = ['jensen_shannon 4']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_4_js/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=12, line_per_file=1)
    sweep[" --fdiv_info "] = ['jensen_shannon 5']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_5_js/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=24, line_per_file=1)
    sweep[" --fdiv_info "] = ['jensen_shannon 6']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_6_js/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=36, line_per_file=1)

    sweep[" --fdiv_info "] = ['gan 3']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_3_gan/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=48, line_per_file=1)
    sweep[" --fdiv_info "] = ['gan 4']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_4_gan/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=60, line_per_file=1)
    sweep[" --fdiv_info "] = ['gan 5']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_5_gan/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=72, line_per_file=1)
    sweep[" --fdiv_info "] = ['gan 6']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_6_gan/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=84, line_per_file=1)

    sweep[" --fdiv_info "] = ['jeffrey 3']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_3_jeffrey/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=96, line_per_file=1)
    sweep[" --fdiv_info "] = ['jeffrey 4']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_4_jeffrey/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=108, line_per_file=1)
    sweep[" --fdiv_info "] = ['jeffrey 5']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_5_jeffrey/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=120, line_per_file=1)
    sweep[" --fdiv_info "] = ['jeffrey 6']
    sweep[" --info "] = ["test_v0/test_TTT/q_0.0_3.0/fdiv_6_jeffrey/"]
    write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=3, comb_num_base=0, prev_file=132, line_per_file=1)


def c20240615():
    sweep = {
        " --pi_lr ": [1e-3, 3e-4, 1e-4],
        " --q_lr_prob ": [1.],
        " --tau ": [0.1, 1.0],
        " --distribution ": ["Student"],
        " --info ": ["test_v0/"],
    }
    target_agents = ["TTT"]
    target_envs = ["HalfCheetah"]
    target_datasets = ["medexp", "medrep"]

    terms = [2,4,6]
    # distances = ['gan', 'jeffrey', 'js']
    distances = ['jensen_shannon']
    combs = list(itertools.product(terms, distances))
    sweep_num = 12
    num_runs=5
    prev_file=0
    for td in combs:
        t, d = td
        sweep[" --fdiv_info "] = ['{} {}'.format(d, t)]
        sweep[" --info "] = ["test_v0/test_TTT/q_0.0_0.0/fdiv_{}_{}/".format(t, d)]
        prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets, num_runs=num_runs, comb_num_base=0, prev_file=prev_file, line_per_file=1)

    # terms = [1]
    # distances = ['gan']
    # sweep[" --fdiv_info "] = ['gan 1']
    # sweep[" --info "] = ["test_v0/test_TTT/q_0.0_0.0/fdiv_1/"]
    # prev_file = write_job_scripts(sweep, target_agents, target_envs, target_datasets,
    #                               num_runs=num_runs, comb_num_base=0, prev_file=prev_file,
    #                               line_per_file=1)

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

if __name__ == "__main__":
    # c20240607()
    # c20240613()
    c20240615()
    # c20240617()