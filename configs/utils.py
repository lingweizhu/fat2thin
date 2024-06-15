# python run_ac_offline.py --seed 0 --env_name Walker2d --dataset medrep --discrete_control 0 --state_dim 17 --action_dim 6 --tau 0.5 --learning_rate 0.0003 --hidden_units 256 --batch_size 256 --timeout 1000 --max_steps 1000000 --log_interval 10000
import os
import copy
import itertools
from pathlib import Path
from default import DEFAULT_AGENT, DEFAULT_ENV
from best_setting import BEST_AGENT

def write_to_file(cmds, prev_file=0, line_per_file=1):
    curr_dir = os.getcwd()
    cmd_file_path = os.path.join(curr_dir, "scripts/tasks_{}.sh")

    cmd_file = Path(cmd_file_path.format(int(prev_file)))
    cmd_file.parent.mkdir(exist_ok=True, parents=True)

    count = 0
    print("First script:", cmd_file)
    file = open(cmd_file, 'w')
    for cmd in cmds:
        file.write(cmd + "\n")
        count += 1
        if count % line_per_file == 0:
            file.close()
            prev_file += 1
            cmd_file = Path(cmd_file_path.format(int(prev_file)))
            file = open(cmd_file, 'w')
    if not file.closed:
        file.close()
    print("Last script:", cmd_file_path.format(str(prev_file)), "\n")
    return prev_file

def generate_cmd(flag_collection, base="python run_ac_offline.py "):
    cmd = base
    for k, v in flag_collection.items():
        cmd += "{} {}".format(k, v)
    cmd += "\n"
    return cmd

# def generate_flag_combinations(sweep_parameters):
def write_job_scripts(sweep_params, target_agents, target_envs, target_datasets, num_runs=5, run_base=0, comb_num_base=0, prev_file=0, line_per_file=1):
    agent_parameters = copy.deepcopy(DEFAULT_AGENT)
    cmds = []
    aed_comb = list(itertools.product(target_agents, target_envs, target_datasets))
    for aed in aed_comb:
        agent, env, dataset = aed
        kwargs = {}
        kwargs[" --agent "] = agent
        kwargs[" --env_name "] = env
        kwargs[" --dataset "] = dataset
        kwargs.update(DEFAULT_ENV[env])

        # if agent in agent_parameters["{}-{}".format(env, dataset)]:
        #     default_params = agent_parameters["{}-{}".format(env, dataset)][agent]
        #     settings = {**sweep_params, **default_params}
        # else:
        settings = sweep_params

        keys, values = zip(*settings.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for comb_num, param_comb in enumerate(param_combinations):
            kwargs[" --param "] = comb_num + comb_num_base
            for (k, v) in param_comb.items():
                kwargs[k] = v
            for run in list(range(run_base, run_base+num_runs)):
                kwargs[" --seed "] = run
                cmds.append(generate_cmd(kwargs))
    return write_to_file(cmds, prev_file=prev_file, line_per_file=line_per_file)

def policy_evolution_scripts(sweep_params, target_agents, target_envs, target_datasets,
                             num_runs=5, run_base=0, comb_num_base=0, prev_file=0, line_per_file=1):

    cmds = []
    aed_comb = list(itertools.product(target_agents, target_envs, target_datasets))
    for aed in aed_comb:
        agent, env, dataset = aed
        kwargs = {}
        kwargs[" --agent "] = agent
        kwargs[" --env_name "] = env
        kwargs[" --dataset "] = dataset
        kwargs.update(DEFAULT_ENV[env])

        # default_params = agent_parameters["{}-{}-{}".format(env, dataset, dist)][agent]
        # settings = {**sweep_params, **default_params}
        settings = sweep_params

        keys, values = zip(*settings.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for comb_num, param_comb in enumerate(param_combinations):
            kwargs[" --param "] = comb_num + comb_num_base
            for (k, v) in param_comb.items():
                kwargs[k] = v
            for run in list(range(run_base, run_base+num_runs)):
                kwargs[" --seed "] = run
                cmds.append(generate_cmd(kwargs, base="python evolution_ac_offline.py "))
    write_to_file(cmds, prev_file=prev_file, line_per_file=line_per_file)

def add_seed_scripts(sweep_params, target_agents, target_envs, target_datasets, target_distributions,
                     defined_param=BEST_AGENT, num_runs=5, run_base=0, comb_num_base=0, prev_file=0, line_per_file=1):
    agent_parameters = copy.deepcopy(defined_param)
    cmds = []
    aedd_comb = list(itertools.product(target_agents, target_envs, target_datasets, target_distributions))
    for aedd in aedd_comb:
        agent, env, dataset, dist = aedd
        kwargs = {}
        kwargs[" --agent "] = agent
        kwargs[" --env_name "] = env
        kwargs[" --dataset "] = dataset
        kwargs[" --distribution "] = dist
        kwargs.update(DEFAULT_ENV[env])

        default_params = agent_parameters["{}-{}-{}".format(env, dataset, dist)][agent]
        settings = {**sweep_params, **default_params}

        keys, values = zip(*settings.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for comb_num, param_comb in enumerate(param_combinations):
            kwargs[" --param "] = comb_num + comb_num_base
            for (k, v) in param_comb.items():
                kwargs[k] = v
            for run in list(range(run_base, run_base+num_runs)):
                kwargs[" --seed "] = run
                cmds.append(generate_cmd(kwargs))
    write_to_file(cmds, prev_file=prev_file, line_per_file=line_per_file)

def add_param_scripts(sweep_params, target_agents, target_envs, target_datasets, target_distributions, num_runs=5, run_base=0, comb_num_base=0, prev_file=0, line_per_file=1):
    def check_repeated(param_comb, default_params):
        for k in default_params.keys():
            if param_comb[k] not in default_params[k]:
                return False
        # print("Repeated parameter", param_comb, default_params)
        return True

    agent_parameters = copy.deepcopy(DEFAULT_AGENT)
    cmds = []
    aedd_comb = list(itertools.product(target_agents, target_envs, target_datasets, target_distributions))
    for aedd in aedd_comb:
        agent, env, dataset, dist = aedd
        kwargs = {}
        kwargs[" --agent "] = agent
        kwargs[" --env_name "] = env
        kwargs[" --dataset "] = dataset
        kwargs[" --distribution "] = dist
        kwargs.update(DEFAULT_ENV[env])

        default_params = agent_parameters["{}-{}".format(env, dataset)][agent]
        # settings = {**sweep_params, **default_params}

        keys, values = zip(*sweep_params.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for comb_num, param_comb in enumerate(param_combinations):
            if not check_repeated(param_comb, default_params):
                kwargs[" --param "] = comb_num + comb_num_base
                for (k, v) in param_comb.items():
                    kwargs[k] = v
                for run in list(range(run_base, run_base+num_runs)):
                    kwargs[" --seed "] = run
                    cmds.append(generate_cmd(kwargs))
    write_to_file(cmds, prev_file=prev_file, line_per_file=line_per_file)
