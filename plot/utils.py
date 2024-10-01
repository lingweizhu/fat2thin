import copy
import itertools
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


CIparam = 1


formal_env_name = {
    'SimEnv3': 'Synthetic Environment',
}
formal_agent_name = {
    'FTT': 'FtTPO'
}
formal_dataset_name = {
    'medexp': 'Medium-Expert',
    'medium': 'Medium',
    'medrep': 'Medium-Replay',
    'random': 'Random',
}
formal_distribution_name = {
    "qGaussian": "Light-Tailed q-Gaussian",
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
    if len(seeds_res) == 0:
        return np.asarray(seeds_res)
    min_len = np.min(np.asarray([len(r) for r in seeds_res]))
    seeds_res = [r[:min_len] for r in seeds_res]
    if not param_sweep and len(seeds_res)<10:
        print("================ Warning on number of seeds ================", pth, len(seeds_res), "\n")
        # print('"'+pth+'",')
    return np.asarray(seeds_res)

def load_key_param(pth, key_params):
    seeds = os.listdir(pth)
    seeds_res = []
    for s in seeds:
        fpth = os.path.join(pth, s)
        fpth = os.path.join(fpth, 'log')
        if os.path.isfile(fpth):
            break
    with open(fpth, 'r') as f:
        lines = f.readlines()
    temp = {}
    for l in lines:
        k = l.split("|")[1].split(":")[0].strip()
        if k in key_params:
            temp[k] = l.split("|")[1].split(":")[1].strip()
    rec = []
    for k in key_params:
        rec.append(temp[k])
    return rec

def load_exp_setting(pth, file, param_sweep, key_params=[])->dict:
    params = os.listdir(pth)
    params_res = {}
    for p in params:
        ppth = os.path.join(pth, p)
        seeds_res = load_param_setting(ppth, file, param_sweep)
        if len(seeds_res) > 0:
            setting = load_key_param(ppth, key_params)
            if key_params == []:
                params_res[p] = seeds_res
            else:
                params_res[p+" "+"_".join(setting)] = seeds_res
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
        # if len(params_res[p]) >= default_sample:
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

def draw_curve(ax, res:dict, smooth_window:int, colors:dict, styles:dict, CIparam=CIparam, highlight=False):
    for lb in res.keys():
        res[lb] = window_smoothing(res[lb], smooth_window)

    if highlight:
        perf_rankings = []
        agents = []
        for lb in res.keys():
            if lb != highlight:
                agents.append(lb)
                perf_rankings.append(np.mean(res[lb], axis=0)[-1])
        best_base = agents[np.asarray(perf_rankings).argmax()]

    for lb in res.keys():
        if highlight and (lb == highlight or lb == best_base):
            alpha_weight = 1.
        elif highlight and (lb != highlight and lb != best_base):
            alpha_weight = 0.2
        else:
            alpha_weight = 1.

        mu = np.mean(res[lb], axis=0)
        err = np.std(res[lb], axis=0) / np.sqrt(res[lb].shape[0]) * CIparam
        if lb in colors:
            ax.plot(mu, label=lb, color=colors[lb], linestyle=styles.get(lb, '-'), alpha=alpha_weight)
            ax.fill_between(np.arange(len(mu)), mu + err, y2=mu - err, alpha=0.3*alpha_weight, color=colors[lb])
        else:
            ax.plot(mu, label=lb, alpha=alpha_weight)
            ax.fill_between(np.arange(len(mu)), mu + err, y2=mu - err, alpha=0.3*alpha_weight)
    return ax

def load_final_perf(pths, smoothing, file="evaluations.npy"):
    res = {}
    for l, pth in pths.items():
        params_res = load_exp_setting(pth, file, param_sweep=True)
        if len(params_res) > 0:
            best_param = choose_param(params_res)
            # print("Best param in {}: {}".format(pth, best_param))
            # res[l] = params_res[best_param]
            bpth = os.path.join(pth, best_param)
            res[l] = load_param_setting(bpth, file, param_sweep=False)
            res[l] = window_smoothing(res[l], smoothing).mean(axis=0)[-1]
    return res

def learning_curve(ax, pths, smoothing, colors, styles, file="evaluations.npy", highlight=False):
    res = {}
    for l, pth in pths.items():
        params_res = load_exp_setting(pth, file, param_sweep=True)
        if len(params_res) > 0:
            best_param = choose_param(params_res)
            # print("Best param in {}: {}".format(pth, best_param))
            # res[l] = params_res[best_param]
            bpth = os.path.join(pth, best_param)
            res[l] = load_param_setting(bpth, file, param_sweep=False)

    ax = draw_curve(ax, res, smoothing, colors, styles, highlight=highlight)
    return ax

def learning_curve_sweep(ax, pth, smoothing, file="evaluations.npy", key_params=[]):
    params_res = load_exp_setting(pth, file, key_params=key_params, param_sweep=True)
    for p in params_res:
        draw_curve(ax, {p: params_res[p]}, smoothing, {}, {})
    plt.legend()
    return ax

def filter_target_parameter(ax, pth, smoothing, file="evaluations.npy", key_param_values={}):
    key_params, key_values = [], []
    for (k,v) in key_param_values.items():
        key_params.append(k)
        key_values.append(v)

    params_res = load_exp_setting(pth, file, key_params=key_params, param_sweep=True)
    filtered_res = {}
    for p in params_res:
        vs = p.split(" ")[1].split("_")
        fit = True
        for v,kv in zip(vs, key_values):
            if type(kv) == float:
                v = float(v)
            if v != kv:
                fit = False
        if fit:
            draw_curve(ax, {p: params_res[p]}, smoothing, {}, {})
    plt.legend()
    return ax

def draw_heatmap(ax, pth, smoothing, file="evaluations.npy", fixed_param={}, x_axis={}, y_axis={}):
    x_axis_key, x_axis_value = x_axis
    y_axis_key, y_axis_value = y_axis
    key_params, key_values = [], []
    for (k,v) in fixed_param.items():
        key_params.append(k)
        key_values.append(v)
    key_params.append(x_axis_key)
    key_params.append(y_axis_key)
    fixed_key_params = key_params[:-2]
    fixed_key_values = key_values

    params_res = load_exp_setting(pth, file, key_params=key_params, param_sweep=True)
    final_res = np.zeros((len(x_axis_value), len(y_axis_value)))
    for p in params_res:
        vs = p.split(" ")[1].split("_")
        fit = True
        for v,kv in zip(vs[:-2], fixed_key_values):
            if type(kv) == float:
                v = float(v)
            if v != kv:
                fit = False
        if fit:
            if type(x_axis_value[0]) == float:
                x = float(vs[-2])
            else:
                x = vs[-2]
            if type(y_axis_value[0]) == float:
                y = float(vs[-1])
            else:
                y = vs[-1]
            x_idx = x_axis_value.index(x)
            y_idx = y_axis_value.index(y)
            final_res[x_idx,y_idx] = params_res[p][:, -10:].mean()

    ax.set_yticks(np.arange(len(x_axis_value)), labels=x_axis_value)
    ax.set_xticks(np.arange(len(y_axis_value)), labels=y_axis_value)
    ax.set_ylabel(x_axis_key)
    ax.set_xlabel(y_axis_key)
    im = ax.imshow(final_res)
    for i in range(final_res.shape[0]):
        for j in range(final_res.shape[1]):
            text = im.axes.text(j-0.4, i+0.2, "{:.2f}".format(final_res[i, j]), fontsize=7)
    plt.colorbar(im, ax=ax)
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
