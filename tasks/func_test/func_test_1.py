import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
import time
import pickle
import os, sys
import argparse
cwd = os.getcwd()
sys.path.append(cwd)
current_file_dir = os.path.dirname(__file__)
from planning.sampling_planner import Sampling_Planner
from crown_configs.func_test.model_def import synthetic_model
from others.helper import print_prop, parse_result

device = "cuda"
model_test = synthetic_model()
output_dir = f'output/func_test'
os.makedirs(output_dir, exist_ok=True)
ABS_OPT_VAL = -1.9803
ABS_OPT_SOL = 0.06270644
ABS_TOL = 2e-3

def quick_test():
    num_sample, n_update_iter = 128000, 50
    noise_list = [0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    weight_list = [0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250]
    lr_list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3]

    seed = 0
    action_dim = 100
    result_dict = {seed: {action_dim: {}}}
    result_dict[seed][action_dim]["GD"] = summarize_results(gd_test(action_dim, seed, num_sample, n_update_iter, lr_list, 5))
    result_dict[seed][action_dim]["MPPI"] = summarize_results(mppi_test(action_dim, seed, num_sample, n_update_iter, noise_list, weight_list))
    result_dict[seed][action_dim]["CEM"] = summarize_results(cem_test(action_dim, seed, num_sample, n_update_iter, 1e-6, 10, 60))
    result_dict[seed][action_dim]["Ours"] = crown_test(action_dim, seed) 
    output_text_file = f'{output_dir}/quick_test.txt'
    if os.path.exists(output_text_file):
        os.remove(output_text_file)
    report_results(result_dict, output_text_file)
    func_vis(output_text_file)

def complete_test():
    num_sample, n_update_iter = 128000, 50
    noise_list = [0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    weight_list = [0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250]
    lr_list = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3]

    dim_interval = 5
    dim_min = 5
    dim_max = 100
    global_seed = 0
    n_test = 1
    final_dict = {}

    for i in range(n_test):
        print(f"-------------- Test seed: {global_seed} --------------")
        result_dict = {}
        for action_dim in range(dim_min, dim_max+1, dim_interval):
            result_dict[action_dim] = {}
            result_dict[action_dim]["GD"] = summarize_results(gd_test(action_dim, global_seed, num_sample, n_update_iter, lr_list, 5))
            result_dict[action_dim]["MPPI"] = summarize_results(mppi_test(action_dim, global_seed, num_sample, n_update_iter, noise_list, weight_list))
            result_dict[action_dim]["CEM"] = summarize_results(cem_test(action_dim, global_seed, num_sample, n_update_iter, 1e-6, 10, 60))
            result_dict[action_dim]["Ours"] = crown_test(action_dim, global_seed) 

        final_dict[global_seed] = result_dict
        global_seed += 1

    # report_results(final_dict)
    plot_file_name = f'{output_dir}/complete_test.pdf'
    plot_optimization_results(final_dict, plot_file_name)
    func_vis()

def report_results(final_dict, output_text_file=None):
    output_string = "--- Test Results ---\n"
    output_string += "Seed | input_dim | method | opt_value | num_opt | runtime\n"
    for seed in final_dict:
        for input_dim in final_dict[seed]:
            for method in final_dict[seed][input_dim]:
                result = final_dict[seed][input_dim][method]
                output_string += f"{seed:<4} | {input_dim:<9} | {method:<6} | {result['cost']:<9.4f} | {result['num_opt']:<7} | {result['runtime']:.2f}\n"
            output_string += f"{seed:<4} | {input_dim:<9} | {'f^*':<6} | {ABS_OPT_VAL* input_dim:<9.4f} | {input_dim:<7} | ----\n"
    output_string += f"--- End of Test Results ---\n"
    output_string += f"opt_value: the best value of f(u) found, ~= -1.9803*input_dim. The smaller the better\n"
    output_string += f"num_opt: the dimension that the best solution found hitting optima, = input_dim. The larger the better\n"

    if output_text_file is not None:
        with open(output_text_file, "a") as f:
            f.write(output_string)
    print(output_string)
    

def func_vis(output_text_file=None):
    num_sample = 1000000
    u = np.linspace(-1, 1, num_sample+1).reshape(-1, 1)
    y = (5*(u**2) + 2 * np.cos(50*u)).sum(axis=1, keepdims=True)
    #  get argmin and min
    u_min = u[np.argmin(y)]
    y_min = np.min(y)
    print(f"f(u) in 1D: u^*: {u_min[0]}, f^*: {y_min}")
    with open(output_text_file, 'a') as f:
        print(f"f(u) in 1D: u^*: {u_min[0]}, f^*: {y_min}", file=f)
    
    plt.figure(figsize=(8, 6))
    plt.plot(u, y, label='f(u)')
    plt.title('Visualization of the function f(u)')
    plt.xlabel('u')
    plt.ylabel('f(u)')
    plt.legend()
    plt.grid(True)
    file_name = f'{output_dir}/function_plot_1d.pdf'
    plt.savefig(file_name)
    print(f"Visualization of f(u) in 1D: {file_name}")
    with open(output_text_file, 'a') as f:
        print(f"Visualization of f(u) in 1D: {file_name}", file=f)
    plt.close()

def model_rollout_fn(state_cur: torch.Tensor, action_seqs: torch.Tensor):
    # state_cur: [1, state_dim]
    # action_seqs: [n_sample, n_look_ahead=1, action_dim]
    # return: [n_sample, n_look_ahead=1, state_dim]
    return {"state_seqs": model_test.get_y(action_seqs)}

def evaluate_traj_fn(state_seqs: torch.Tensor, action_seqs: torch.Tensor):
    # state_seqs: [n_sample, n_look_ahead=1, state_dim]
    # action_seqs: [n_sample, n_look_ahead=1, action_dim]
    # return: [n_sample]
    return {"rewards": -torch.sum(state_seqs, dim=(1,2))}

def mppi_test(action_dim, seed,  num_sample, n_update_iter, noise_list, weight_list):
    action_lb = torch.tensor([-1]*action_dim, device=device)
    action_ub = torch.tensor([1]*action_dim, device=device)

    config = {
        "action_dim": action_dim,
        "model_rollout_fn": model_rollout_fn,
        "evaluate_traj_fn": evaluate_traj_fn,
        "n_look_ahead": 1,
        "action_lower_lim": action_lb,
        "action_upper_lim": action_ub,
        "seed": seed,
        "planner_type": "MPPI",
        "device": device,
        # "experiment_folder": output_dir,
        # "verbose": True,
        }
    mppi_config = {
        "n_sample": 10000,
        "n_update_iter": 200,
        "reward_weight": 1000000.0,
        "noise_type": "normal",
        "noise_level": 0.1,
        "noise_decay": 0.9,
        "reject_bad": True}
    config["mppi_config"] = mppi_config
    exp_results = []
    for noise in noise_list:
        for weight in weight_list:
            for noise_type in ["normal"]:
                start_time = time.time()
                mppi_config["n_sample"] = num_sample
                mppi_config["n_update_iter"] = n_update_iter
                mppi_config["noise_level"] = noise
                mppi_config["reward_weight"] = weight
                mppi_config["noise_type"] = noise_type
                reset_seed(config["seed"])
                planner = Sampling_Planner(config)
                action_seqs = torch.ones(1, action_dim, device=device)* (action_ub - action_lb)/2
                output = planner.trajectory_optimization(torch.tensor([0], device=device), action_seqs)
                mppi_time = time.time()-start_time
                act_seq = output["act_seq"]
                cost = -output["best_eval_output"]["rewards"].item()
                # best_cost_lst = output["best_cost_lst"]
                best_cost_lst = None
                # print(f"action sequence: {act_seq}, cost: {cost}")
                # print(f"cost: {cost}, time {mppi_time}")
                act_seq_list = act_seq.flatten().tolist()
                num_opt = count_optimal(act_seq_list)
                # print(result)
                # result = [num_sample, noise, weight, noise_type, act_seq_list, cost, best_cost_lst]
                result = [num_sample, noise, weight, noise_type, num_opt, cost, mppi_time]
                exp_results.append(result)
                # exit()

    return exp_results

def cem_test(action_dim, seed, num_sample, n_update_iter, jitter_factor, min_n_elites, n_agents):
    action_lb = torch.tensor([-1]*action_dim, device=device)
    action_ub = torch.tensor([1]*action_dim, device=device)

    config = {
        "action_dim": action_dim,
        "model_rollout_fn": model_rollout_fn,
        "evaluate_traj_fn": evaluate_traj_fn,
        "n_look_ahead": 1,
        "action_lower_lim": action_lb,
        "action_upper_lim": action_ub,
        "seed": seed,
        "planner_type": "DecentCEM",
        "device": device,
        # "experiment_folder": output_dir,
        # "verbose": True,
        }
    cem_config = {
        "n_sample": num_sample,
        "n_update_iter": n_update_iter,
        "jitter_factor": jitter_factor,
        "elite_ratio": 0.000000001,
        "min_n_elites": min_n_elites,
        "n_agents": n_agents}
    config["cem_config"] = cem_config
    exp_results = []

    start_time = time.time()
    reset_seed(config["seed"])
    planner = Sampling_Planner(config)
    action_seqs = torch.ones(1, action_dim, device=device)* (action_ub - action_lb)/2
    output = planner.trajectory_optimization(torch.tensor([0], device=device), action_seqs)
    mppi_time = time.time()-start_time
    act_seq = output["act_seq"]
    cost = -output["best_eval_output"]["rewards"].item()
    # best_cost_lst = output["best_cost_lst"]
    best_cost_lst = None
    # print(f"action sequence: {act_seq}, cost: {cost}")
    # print(f"cost: {cost}, time {mppi_time}")
    act_seq_list = act_seq.flatten().tolist()
    num_opt = count_optimal(act_seq_list)
    # print(result)
    # result = [num_sample, noise, weight, noise_type, act_seq_list, cost, best_cost_lst]
    result = [num_sample, jitter_factor, min_n_elites, n_agents, num_opt, cost, mppi_time]
    exp_results.append(result)
                # exit()

    return exp_results

def gd_test(action_dim, seed, num_sample, n_update_iter, lr_list, n_repeat):
    action_lb = torch.tensor([-1]*action_dim, device=device)
    action_ub = torch.tensor([1]*action_dim, device=device)

    config = {
        "action_dim": action_dim,
        "model_rollout_fn": model_rollout_fn,
        "evaluate_traj_fn": evaluate_traj_fn,
        "n_look_ahead": 1,
        "action_lower_lim": action_lb,
        "action_upper_lim": action_ub,
        "seed": seed,
        "planner_type": "GD",
        "device": device,
        # "experiment_folder": output_dir,
        # "verbose": True,
        }
    gd_config = {
        "n_sample": 10000,
        "n_update_iter": 200,
        "noise_type": "random",
        "lr": 0.01,
        "lr_decay": 1,
        "reject_bad": True}
    config["gd_config"] = gd_config
    exp_results = []
    for i in range(n_repeat):
        for lr in lr_list:
            for noise_type in ["normal"]:
                start_time = time.time()
                gd_config["n_sample"] = num_sample
                gd_config["n_update_iter"] = n_update_iter
                gd_config["lr"] = lr
                config["seed"] += 1
                reset_seed(config["seed"])
                planner = Sampling_Planner(config)
                action_seqs = torch.ones(1, action_dim, device=device)* (action_ub - action_lb)/2
                output = planner.trajectory_optimization(torch.tensor([0], device=device), action_seqs)
                mppi_time = time.time()-start_time
                act_seq = output["act_seq"]
                cost = -output["best_eval_output"]["rewards"].item()
                # best_cost_lst = output["best_cost_lst"]
                best_cost_lst = None
                # print(f"action sequence: {act_seq}, cost: {cost}")
                # print(f"lr: {lr}, cost: {cost}, time {mppi_time}")
                act_seq_list = act_seq.flatten().tolist()
                num_opt = count_optimal(act_seq_list)
                # print(result)
                # result = [num_sample, noise, weight, noise_type, act_seq_list, cost, best_cost_lst]
                result = [num_sample, lr, noise_type, num_opt, cost, mppi_time]
                exp_results.append(result)
                # exit()

    return exp_results

def crown_test(action_dim, seed):
    from planning.utils import analyze_search_progress
    from Verifier_Development.complete_verifier.abcrown import ABCROWN

    action_lb = torch.tensor([-1]*action_dim, device=device)
    action_ub = torch.tensor([1]*action_dim, device=device)
    crown_config_name = "crown_configs/func_test/planning_sample.yaml"
    with open(crown_config_name, "r") as f:
        crown_config_dict = yaml.load(f, Loader=yaml.Loader)
    
    crown_config_prefix = "crown_configs/func_test/"
    os.makedirs(crown_config_prefix+"vnnlib/", exist_ok=True)
    os.makedirs(crown_config_prefix+"output/", exist_ok=True)
    vnnlib_path = "crown_configs/func_test/vnnlib/func_test_1.vnnlib"
    with open(vnnlib_path, "w") as f:
        print_prop(0, f, action_dim, action_lb, action_ub)

    # hack to collect samples in crown
    sample_file = "crown_configs/func_test/output/sample.pkl"
    # remove the old sample file 
    if os.path.exists(sample_file):
        os.remove(sample_file)
        # os.mknod(sample_file)

    abcrown_verbose = False
    crown_config_dict["general"]["root_path"] = crown_config_prefix
    crown_config_dict["general"]["device"] = device
    crown_config_dict["general"]["seed"] = seed
    crown_config_dict["specification"]["vnnlib_path"] = vnnlib_path
    crown_config_dict["find_feasible_solution"]["report_per_iter"] = abcrown_verbose
    crown_config_dict["find_feasible_solution"]["use_upper_bound"] = True
    crown_sol_file = crown_config_prefix + f"output/crown_sol.txt"
    crown_config_dict["find_feasible_solution"]["save_solution_path"] = crown_sol_file
    bab_file = crown_config_prefix + "output/crown_bab.pkl"
    crown_config_dict["find_feasible_solution"]["save_bab_path"] = bab_file
    crown_config_dict["model"][
            "name"
        ] = f"""Customized("model_def.py", "synthetic_model")"""
    crown_config_dict["model"]["input_shape"] = [1, action_dim]
    with open(crown_config_name, "w") as f:
        yaml.dump(crown_config_dict, f, sort_keys=False)
    abcrown = ABCROWN(["--config", crown_config_name])
    start_time = time.time()
    abcrown.main()
    crown_time = time.time()-start_time

    feasible_sol, best_output, _, intermediate_best_outputs, final_lb = parse_result(crown_sol_file, abcrown_verbose, True)
    # print(intermediate_best_outputs)
    num_opt = count_optimal(feasible_sol)
    print(f"num_opt: {num_opt}, cost: {best_output}")

    vis_file_prefix = crown_config_prefix + "output/vis"
    if abcrown_verbose:
        space_size_dict = analyze_search_progress(bab_file, vis_file_prefix, "pdf")
        plot_intermediate_outputs(intermediate_best_outputs, f'{vis_file_prefix}_intermediate.pdf')
    return_dict = {"num_opt": num_opt, "cost": best_output[-1], "runtime": crown_time, "final_lb": final_lb}
    if abcrown_verbose:
        return_dict["search_space_size"] = space_size_dict["search_space_size"][-1]
        return_dict["total_space_size"] = space_size_dict["total_space_size"][-1]
    os.remove("out.txt")
    return return_dict

def summarize_results(exp_results):
    return {"num_opt": np.max([result[-3] for result in exp_results]), "cost": np.min([result[-2] for result in exp_results]), "runtime": np.sum([result[-1] for result in exp_results])}

def plot_optimization_results(final_dict, plot_file_name):
    seed_list = list(final_dict.keys())
    input_dim_list = list(final_dict[seed_list[0]].keys())
    method_list = list(final_dict[seed_list[0]][input_dim_list[0]].keys())

    aggregated_results = {method: {input_dim: [final_dict[seed][input_dim][method]['cost'] for seed in seed_list] for input_dim in input_dim_list} for method in method_list}
    
    font_size = 18
    plt.figure(figsize=(6, 6))
    plt.plot(input_dim_list, [-1.9803*d for d in input_dim_list], label="Optimal", linestyle='--', color='black', linewidth=4)

    for method in aggregated_results:
        medians = np.array([np.median(aggregated_results[method][input_dim]) for input_dim in input_dim_list])
        percentiles_25 = np.array([np.percentile(aggregated_results[method][input_dim], 25) for input_dim in input_dim_list])
        percentiles_75 = np.array([np.percentile(aggregated_results[method][input_dim], 75) for input_dim in input_dim_list])
        
        plt.plot(input_dim_list, medians, label=f"{method}", linewidth=2)
        plt.fill_between(input_dim_list, percentiles_25, percentiles_75, alpha=0.2)

    plt.legend(loc='best', fontsize=font_size)
    plt.xticks(ticks=range(0, 101, 25), fontsize = font_size)
    plt.yticks(ticks=range(-200, 1, 50), fontsize = font_size)
    plt.savefig(plot_file_name)
    print(plot_file_name)
    plt.close()

def plot_intermediate_outputs(intermediate_best_outputs, plot_file_name):
    # Create a figure and axis
    plt.figure(figsize=(10, 8))

    # Plot the intermediate best outputs
    plt.plot(intermediate_best_outputs)

    plt.xlabel('Iteration')
    plt.ylabel('Objective')
    plt.title('Intermediate Best Outputs')

    plt.savefig(plot_file_name)
    print(plot_file_name)
    plt.close()

def count_optimal(act_seq_list):
    num_opt = 0
    for act in act_seq_list:
        if np.abs(np.abs(act) - ABS_OPT_SOL) < ABS_TOL:
            num_opt += 1
    return num_opt

def reset_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests")
    parser.add_argument(
        "-complete",
        action="store_true",
        help="Run the complete test"
    )
    args = parser.parse_args()
    start_time = time.time()
    if args.complete:
        complete_test()
    else:
        quick_test()
    print(f"Test finished in {time.time()-start_time:.2f} seconds")
    