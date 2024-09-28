import copy
import os, itertools
from utils import load_exp_setting, choose_param, load_parameter


def extract_best_param():
    import itertools
    import sys
    sys.path.append('../')
    import json
    from configs.default import DEFAULT_AGENT

    # D4RL
    # envs = ["SimEnv3"]
    # datasets = ["random"]
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medium", "medrep"]

    agents = ["FTT"]
    distributions = ["qGaussian"]
    # pth_base = "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/{}/{}/"
    # pth_base = "../data/output/test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/{}/{}/{}/{}/"
    pth_base = "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_SPOT/{}/{}/{}/{}/"

    # agents = ["IQL", "TAWAC"]
    # distributions = ["SGaussian"]
    # pth_base = "../data/baseline_data/output/test_v1/{}/{}/{}/{}/"

    # # Simulator
    # envs = ["SimEnv3"]
    # datasets = ["random"]

    # agents = ["FTT"]
    # distributions = ["qGaussian"]
    # pth_base = "../data/output/test_v1/test_FTT/proposal_HTqG_actor_qG_actorloss_KL/{}/{}/{}/{}/"

    # agents = ["FTT"]
    # distributions = ["qGaussian"]
    # pth_base = "../data/output/test_v1/test_FTT/proposal_SG_actor_qG_actorloss_KL/{}/{}/{}/{}/"

    # agents = ["IQL", "TAWAC"]
    # distributions = ["SGaussian"]
    # pth_base = "../data/output/test_v1/baseline/{}/{}/{}/{}/"

    combs = list(itertools.product(envs, datasets, agents, distributions))
    write_json = {}
    for comb in combs:
        e, d, a, dist = comb
        pth = pth_base.format(e, d, a, dist)
        params_res = load_exp_setting(pth, "evaluations.npy", param_sweep=True)
        k = '{}-{}-{}'.format(e, d, dist)
        if k not in write_json:
            write_json[k] = {}
        if '{}-{}'.format(e, d) in DEFAULT_AGENT and \
                a in DEFAULT_AGENT['{}-{}'.format(e, d)]:
            write_json[k][a] = copy.deepcopy(DEFAULT_AGENT['{}-{}'.format(e, d)][a])
        else:
            write_json[k][a] = {}
        if len(params_res) > 0:
            best_param = choose_param(params_res)
            p_idx = int(best_param.split("_")[0])
            # if a == "TAWAC" and d == "medium":
            #     print(pth, best_param)
            # else:
            cfg = os.path.join(pth, "{}/0_run/log".format(best_param))
            values = load_parameter(cfg, ['pi_lr', 'tau', 'expectile'])
            write_json[k][a][' --param '] = [p_idx]
            write_json[k][a][' --pi_lr '] = [float(values['pi_lr'])]
            write_json[k][a][' --tau '] = [float(values['tau'])]
            if a == "TAWAC" and d == "medium":
                write_json[k][a][' --tau '] = [float(values['tau'])]

    with open('best_setting.json', 'w') as f:
        json.dump(write_json, f, indent=4)


if __name__ == "__main__":
    extract_best_param()