import copy
import itertools
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CIparam = 1
formal_dataset_name = {
    'medexp': 'Medium-Expert',
    'medium': 'Medium',
    'medrep': 'Medium-Replay',
}
formal_distribution_name = {
    "HTqGaussian": "Heavy-Tailed q-Gaussian",
    "SGaussian": "Squashed Gaussian",
    "Gaussian": "Gaussian",
    "Student": "Student's t",
    "Beta": "Beta"
}


def extract_best_param():
    import itertools
    import sys
    sys.path.append('../')
    import json
    from configs.default import DEFAULT_AGENT

    # lrs = [1e-3, 3e-4, 1e-4, 3e-3]
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medium", "medrep"]
    agents = ["IQL", "InAC", "TAWAC", "AWAC", "TD3BC"]
    distributions = ["SGaussian", "Beta", "Student", "HTqGaussian", "Gaussian"]
    combs = list(itertools.product(envs, datasets, agents, distributions))
    pth_base = "../data/output/test_v1/{}/{}/{}/{}/"
    write_json = {}
    for comb in combs:
        e, d, a, dist = comb
        pth = pth_base.format(e, d, a, dist)
        params_res = load_exp_setting(pth, "evaluations.npy", param_sweep=True)
        k = '{}-{}-{}'.format(e, d, dist)
        if k not in write_json:
            write_json[k] = {}
        write_json[k][a] = copy.deepcopy(DEFAULT_AGENT['{}-{}'.format(e, d)][a])
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
            if a == "TAWAC" and d == "medium":
                write_json[k][a][' --tau '] = [float(values['tau'])]

    with open('best_setting.json', 'w') as f:
        json.dump(write_json, f, indent=4)


def check_random_seeds(pth_base, envs, agents, distributions, datasets):
    combs = list(itertools.product(envs, datasets, agents, distributions))
    for comb in combs:
        e, d, a, dist = comb
        pth = pth_base.format(e, d, a, dist)
        params_res = load_exp_setting(pth, "evaluations.npy", param_sweep=True)
        best_param = choose_param(params_res)
        avg_res = average_seeds(params_res)
        if len(params_res[best_param]) < 10:
            print(pth, len(params_res[best_param]))
            for param in params_res:
                print(param, params_res[param].shape[0], avg_res[param].sum())
            print()



def find_suboptimal_setting():
    import sys
    sys.path.append('../')
    import json
    from configs.best_setting import BEST_AGENT

    # with open('best_setting.json', 'r') as f:
    #     new_best = json.load(f)
    # for k in BEST_AGENT.keys():
    #     for a in BEST_AGENT[k].keys():
    #         correct = new_best[k][a]
    #         old = BEST_AGENT[k][a]
    #         if not (a == "TAWAC" and "medium" in k):
    #             if correct[' --param '][0] != old[' --param '][0]:
    #                 print(k, a, correct[' --param '][0], old[' --param '][0])

    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medium", "medrep"]
    agents = ["IQL", "InAC", "TAWAC", "AWAC", "TD3BC"]
    distributions = ["SGaussian", "Beta", "Student", "HTqGaussian"]
    combs = list(itertools.product(envs, datasets, agents, distributions))
    pth_base = "../data/output/test_v1/{}/{}/{}/{}/"
    prev_file = 0
    for comb in combs:
        e, d, a, dist = comb
        pth = pth_base.format(e, d, a, dist)
        params_res = load_exp_setting(pth, "evaluations.npy", param_sweep=True)
        best_param = choose_param(params_res)

        params = os.listdir(pth)
        run_param = []
        for p in params:
            ppth = os.path.join(pth, p)
            seeds = os.listdir(ppth)
            if len(seeds) > 5:
                run_param.append(p)
        if run_param!=[] and best_param not in run_param:
            print(run_param, best_param)
            info = ""
            info += "target_agents = ['{}']\n".format(a)
            info += "target_envs = ['{}']\n".format(e)
            info += "target_datasets = ['{}']\n".format(d)
            info += "target_distributions = ['{}']\n".format(dist)
            info += "add_seed_scripts(sweep, target_agents, target_envs, target_datasets, target_distributions, defined_param=BEST_AGENT, num_runs=5, run_base=5, comb_num_base=0, prev_file={}, line_per_file=1)\n\n".format(prev_file)
            prev_file += 5
            print(info)

def check_param_consistancy():
    envs = ["HalfCheetah", "Hopper", "Walker2d"]
    datasets = ["medexp", "medium", "medrep"]
    agents = ["IQL", "InAC", "TAWAC", "AWAC", "TD3BC"]
    distributions = ["SGaussian", "Beta", "Student", "HTqGaussian"]
    combs = list(itertools.product(envs, datasets, agents, distributions))
    pth_base = "../data/output/test_v1/{}/{}/{}/{}/"
    for comb in combs:
        e, d, a, dist = comb
        pth = pth_base.format(e, d, a, dist)

        params = os.listdir(pth)
        for p in params:
            ppth = os.path.join(pth, p)
            seeds = os.listdir(ppth)
            if len(seeds) > 5:
                old_cfg = os.path.join(ppth, "0_run/log")
                new_cfg = os.path.join(ppth, "9_run/log")
                old_setting = load_parameter(old_cfg, ['pi_lr', 'tau', 'expectile'])
                new_setting = load_parameter(new_cfg, ['pi_lr', 'tau', 'expectile'])
                for k in old_setting:
                    if old_setting[k] != new_setting[k]:
                        print("Different settings in", ppth)
                        print("OLD")
                        print(old_setting)
                        print("NEW")
                        print(new_setting)
                        print()
    print("Done")

def load_parameter(fp, params):
    with open(fp, 'r') as f:
        lines = f.readlines()[:50]
    values = {}
    for l in lines:
        key = l.split("|")[1].split(":")[0].strip()
        if key in params:
            values[key] = l.split("|")[1].split(":")[1].strip()
    return values

def load_file(pth):
    if os.path.isfile(pth):
        return np.load(pth)

def load_param_setting(pth, file, param_sweep):
    if param_sweep:
        seeds = ["{}_run".format(i) for i in range(5)]
    else:
        seeds = os.listdir(pth)
    seeds_res = []
    for s in seeds:
        fpth = os.path.join(pth, s)
        fpth = os.path.join(fpth, file)
        res = load_file(fpth)
        if res is not None:
            seeds_res.append(res)
    if not param_sweep and len(seeds_res)<10:
        print("================ Warning on number of seeds ================", pth, len(seeds_res), "\n")
    return np.asarray(seeds_res)

def load_exp_setting(pth, file, param_sweep)->dict:
    params = os.listdir(pth)
    params_res = {}
    for p in params:
        ppth = os.path.join(pth, p)
        seeds_res = load_param_setting(ppth, file, param_sweep)
        if len(seeds_res) > 0:
            params_res[p] = seeds_res
    return params_res

def average_seeds(params_res:dict):
    avg_res = {}
    for p in params_res.keys():
        avg_res[p] = np.mean(params_res[p], axis=0)
    return avg_res

def choose_param(params_res: np.ndarray, default_sample=5):
    avg_res = average_seeds(params_res)
    aucs = {}

    # more_runs = []
    for p in avg_res.keys():
        if len(params_res[p]) >= default_sample:
            aucs[p] = avg_res[p].sum()
            # aucs[p] = avg_res[p][-1]

        # if params_res[p].shape[0] > 5:
        #     more_runs.append(p)
    # assert len(more_runs) <= 1
    # if len(more_runs) == 1:
    #     # print(max(aucs, key=aucs.get), more_runs[0])
    #     return more_runs[0]
    # else:

    best = max(aucs, key=aucs.get)
    return best

def draw_curve(ax, res:dict, smooth_window:int, colors:dict, styles:dict):
    for lb in res.keys():
        res[lb] = window_smoothing(res[lb], smooth_window)
        mu = np.mean(res[lb], axis=0)
        err = np.std(res[lb], axis=0) / np.sqrt(res[lb].shape[0]) * CIparam
        if lb in colors:
            ax.plot(mu, label=formal_distribution_name[lb], color=colors[lb], linestyle=styles[lb])
            ax.fill_between(np.arange(len(mu)), mu + err, y2=mu - err, alpha=0.3, color=colors[lb])
        else:
            ax.plot(mu, label=lb)
            ax.fill_between(np.arange(len(mu)), mu + err, y2=mu - err, alpha=0.3)
    return ax

def learning_curve(ax, pths, smoothing, colors, styles, file="evaluations.npy"):
    res = {}
    for l, pth in pths.items():
        params_res = load_exp_setting(pth, file, param_sweep=True)
        if len(params_res) > 0:
            best_param = choose_param(params_res)
            # print("Best param in {}: {}".format(pth, best_param))
            # res[l] = params_res[best_param]
            bpth = os.path.join(pth, best_param)
            res[l] = load_param_setting(bpth, file, param_sweep=False)

    ax = draw_curve(ax, res, smoothing, colors, styles)
    plt.legend()
    return ax

def learning_curve_sweep(ax, pths, smoothing, file="evaluations.npy"):
    for l, pth in pths.items():
        params_res = load_exp_setting(pth, file, param_sweep=True)
        for p in params_res:
            draw_curve(ax, {p: params_res[p]}, smoothing, {}, {})
    plt.legend()
    return ax

def window_smoothing(ary, window):
    new_ary = np.empty(ary.shape)
    for i in range(ary.shape[1]):
        new_ary[:, i] = ary[:, max(0, i-window): i+1].mean(axis=1)
    return new_ary

def fill_in_path(pth:dict, setting:list)->dict:
    new_pth = copy.deepcopy(pth)
    for lb in pth:
        new_pth[lb] = pth[lb].format(*setting)
    return new_pth

def c240604():
    pths = {
        "FTT-AWAC": "../data/output/test_v0/{}/{}/FTT_AWAC/qGaussian/",
    }
    colors = {
        "FTT-AWAC": "C0",
    }
    styles = {
        "FTT-AWAC": "-",
    }
    envs = ["Hopper"]
    datasets = ["expert"]
    fig, axs = plt.subplots(len(datasets), len(envs), figsize=(5*len(envs), 4*len(datasets)))
    if len(datasets) == 1:
        axs = [axs]
    if len(datasets) == 1:
        axs = [axs]
    for ie, e in enumerate(envs):
        for id_, d in enumerate(datasets):
            ed_pth = fill_in_path(pths, [e, d])
            learning_curve_sweep(axs[id_][ie], ed_pth, smoothing=10)
            axs[id_][ie].set_title("{}-{}".format(e, d))
    plt.show()


if __name__ == "__main__":
    # extract_best_param()
    # find_suboptimal_setting()
    # check_param_consistancy()
    c240604()