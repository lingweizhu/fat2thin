import copy
import os, itertools
from utils import write_job_scripts


def read_param_setting(pth, param_key):
    run = os.listdir(pth)[0]
    fp = os.path.join(pth, '{}/log'.format(run))
    with open(fp, 'r') as f:
        lines = f.readlines()
    param_value = {}
    for line in lines:
        if len(param_value) == len(param_key):
            return param_value
        k = line.split('|')[1].strip().split(':')[0].strip()
        k = " --" + k + " "
        if k in param_key:
            param_value[k] = [line.split('|')[1].strip().split(':')[1].strip()]


def experiment_run():

    sweep = {
        " --pi_lr ": [1e-3, 3e-4],
        " --tau ": [1.0, 0.5, 0.01],
        " --rho ": [0.2],
        " --q_lr_prob ": [1.],
        " --actor_loss ": ["KL"],
        " --proposal_distribution ": ["SGaussian"],
        " --distribution ": ["qGaussian"],
        " --info ": ["test_v0/test_FTT/proposal_SG_actor_qG_actorloss_KL/"],
    }
    folders = [
        "../data/output/test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/",
        # "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/",
        # "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/",
    ]
    job_id = 413

    for folder in folders:
        for env in os.listdir(folder):
            fenv = os.path.join(folder, env)
            target_envs = [env]

            for data in os.listdir(fenv):
                target_datasets = [data]

                fdata = os.path.join(fenv, data)
                for agent in os.listdir(fdata):
                    target_agents = [agent]

                    fagent = os.path.join(fdata, agent) + "/qGaussian/"

                    temp = copy.deepcopy(sweep)
                    temp[" --info "] = [folder.split("output/")[1]]
                    keys, values = zip(*temp.items())
                    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

                    params = os.listdir(fagent)
                    if len(params) != len(param_combinations):
                        exist_p = [int(p.split("_")[0]) for p in params]
                        for p in range(len(param_combinations)):
                            if p not in exist_p:
                                missing_setting = copy.deepcopy(param_combinations[p])
                                for k in missing_setting:
                                    missing_setting[k] = [missing_setting[k]]
                                # print(missing_setting)
                                # exit()
                                job_id = write_job_scripts(missing_setting, target_agents, target_envs, target_datasets,
                                                           num_runs=5, run_base=0, comb_num_base=p, prev_file=job_id, line_per_file=1)

                    else:
                        for pf in params:
                            fparam = os.path.join(fagent, pf)
                            p = int(pf.split("_")[0])

                            try:
                                missing_setting = read_param_setting(fparam, list(keys))
                            except:
                                missing_setting = copy.deepcopy(param_combinations[p])
                                for k in missing_setting:
                                    missing_setting[k] = [missing_setting[k]]

                            check_run(fparam, agent, missing_setting, target_agents, target_envs,
                                      target_datasets, p, job_id, max_run=5)

def check_run(fparam, agent, missing_setting, target_agents, target_envs, target_datasets, p, job_id, max_run=5):
    runs = [int(r.split("_")[0]) for r in os.listdir(fparam)]
    for r in range(max_run):
        if (r not in runs or
                not os.path.exists(os.path.join(fparam, "{}_run".format(r)) + "/evaluations.npy")):
            job_id = write_job_scripts(missing_setting, target_agents, target_envs,
                                       target_datasets,
                                       num_runs=1, run_base=r, comb_num_base=p,
                                       prev_file=job_id, line_per_file=1)
    return job_id


def parameter_run():
    # folders = [
    #     "../data/output/test_v0/naive_projection_baselines/HalfCheetah/medexp/TAWACqGC/qGaussian/3_param",
    #     "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/Walker2d/medexp/FTT/qGaussian/1_param"
    # ]
    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4],
    #     " --tau ": [1.0, 0.5, 0.01],
    #     " --rho ": [0.2],
    #     " --q_lr_prob ": [1.],
    #     " --actor_loss ": ["KL"],
    #     " --proposal_distribution ": ["SGaussian"],
    #     " --distribution ": ["qGaussian"],
    #     " --info ": ["test_v0/test_FTT/proposal_SG_actor_qG_actorloss_KL/"],
    # }

    # folders = [
    #     "../data/output/test_v1/baseline/SimEnv3/random/IQL/SGaussian/17_param",
    # ]
    # sweep = {
    #     " --pi_lr ": [1e-3, 3e-4, 1e-4, 3e-3],
    #     " --q_lr_prob ": [1.],
    #     " --tau ": [1./3, 0.7],
    #     " --expectile ": [0.7, 0.8, 0.9],
    #     " --rho ": [0.2],
    #     " --gamma ": [0.9],
    #     " --timeout ": [24],
    #     " --log_interval ": [10000],
    #     " --max_steps ": [500000],
    #     " --distribution ": ["SGaussian"],
    #     " --info ": ["test_v1/baseline/"],
    # }
    # job_id = 11

    folders = [
        "../data/output/test_v1/baseline/SimEnv3/random/XQL/SGaussian/5_param",
        "../data/output/test_v1/baseline/SimEnv3/random/SQL/SGaussian/7_param",
    ]
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
    job_id = 16

    for fparam in folders:
        pf = fparam.split("/")[-1]
        p = int(pf.split("_")[0])

        temp = copy.deepcopy(sweep)
        temp[" --info "] = ["/".join(fparam.split("output/")[1].split("/")[:2])]
        agent = fparam.split("/")[-3]
        target_agents = [agent]
        target_datasets = [fparam.split("/")[-4]]
        target_envs = [fparam.split("/")[-5]]
        print(temp[" --info "], agent, target_envs, target_datasets)
        keys, values = zip(*temp.items())
        missing_setting = read_param_setting(fparam, list(keys))

        job_id = check_run(fparam, agent, missing_setting, target_agents, target_envs,
                           target_datasets, p, job_id, max_run=10)


if __name__ == "__main__":
    # experiment_run()
    parameter_run()
