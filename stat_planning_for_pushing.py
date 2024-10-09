import os
import sys

sys.path.append(os.getcwd())
cwd = os.getcwd()
import traceback
import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import yaml
import json

from model.mlp import MLP

# torch.autograd.set_detect_anomaly(True)
# task specific
from planning.planning_for_pushing import planning
from others.plotter import (
    plot_planning_stat,
    plot_convergence_model,
    plot_search_space_size,
    plot_planning,
    plot_prediction_error_in_planning,
)

from others.helper import *
import tasks.merging_L.l_sim as l_sim
import tasks.inserting.hp_sim as hp_sim
from planning.utils import *

# default_config does not exist, just a placeholder
@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(config: DictConfig) -> None:
    assert config.planning.model_path is not None, "model_path is not provided"
    save_path = '/'.join(config.planning.model_path.split('/')[:-1])
    # config_path = save_path + '/config.yaml'
    # # load config file
    # with open(config_path, 'r') as f:
    #     new_config = yaml.safe_load(f)
    # new_config['planning'] = config['planning']

    # config = DictConfig(new_config)
    config["training"] = False
    task_name = config["task_name"]
    enable_latent = "latent" in task_name
    if not enable_latent and config["train"]["include_com"]:
        config["data"]["state_dim"] += 2 * config["data"]["obj_num"]
    planning_config = config["planning"]
    enable_real = config["real_exp"]["enable"]
    method_types = list(planning_config["method_types"])
    force_method_types = planning_config["force_method_types"]

    if enable_latent:
        aemodel = get_aemodel(config)
    
    open_loop = planning_config["open_loop"]
    abcrown_verbose = (
        planning_config["abcrown_verbose"] and open_loop and ("CROWN" in method_types or "CROWN" in force_method_types)
    )
    planning_config["abcrown_verbose"] = abcrown_verbose

    # for crown specific configs, time tag for avoiding overwriting
    now = time.localtime()
    time_tag = f"_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}"
    crown_config_dict, crown_name_dict = load_crown_config(config, time_tag)
    max_iter = crown_config_dict["bab"]["max_iterations"]

    # general configs
    config_for_planning = create_general_planning_config(config, crown_name_dict, crown_config_dict)
    task_spec_dict = config_for_planning["task_spec_dict"]
    json_file_path, vis_file_name_prefix = create_logging_file(config, save_path, config_for_planning)
    reload_json_path = planning_config.get("reload_json_path", None)
    num_test = planning_config["num_test"]

    # Load existing results from JSON file if it exists
    all_res = reload_res(json_file_path)
    if reload_json_path is not None:
        all_res = reload_res(reload_json_path)

    # prepare for experiment
    stat = {}
    inter_results = {}
    cost_types = get_cost_types(config)
    # since crown will reset seeds, we need to generate test instances before running crown
    init_pose_list, target_pose_list, init_pusher_pos_list, target_pusher_pos_list = generate_test_cases(
        config["seed"], num_test, task_spec_dict, planning_config["test_id"]
    )

    # try:
    model_file = config.planning.model_path
    print(f"load model from {model_file}")
    model: MLP = torch.load(model_file, map_location=planning_config["device"]).eval()
    model.config.planning = planning_config
    num_relu = sum(model.get_activation_stat())
    num_relu = str(num_relu)
    inter_result = []
    space_size_result = []

    method_obj_dict = {method_type: {cost_type: [] for cost_type in cost_types} for method_type in method_types}
    if num_relu not in all_res:
        all_res[num_relu] = {}

    for i in range(num_test):
        method_res = {}
        res_reload = {}
        # init_pusher_pos: [2], init_pose: [3], target_pose: [3]
        init_pusher_pos = init_pusher_pos_list[i]
        init_pose = init_pose_list[i]
        target_pose = target_pose_list[i]
        # init_pose: [num_obj, 3], target_pose: [num_obj, 3]
        if "merging_L" in task_name:
            if len(init_pose) == 3:
                if config["planning"]["exchange_obj"]:
                    init_pose = np.array(l_sim.parse_merging_pose(init_pose, task_spec_dict, 20, 3))
                else:
                    init_pose = np.array(l_sim.parse_merging_pose(init_pose, task_spec_dict, 20, 2))
            else:
                assert len(init_pose) == 6
                init_pose = np.array(init_pose).reshape(2, 3)
            target_pose = np.array(l_sim.parse_merging_pose(target_pose, task_spec_dict, 0, 0))
        elif "inserting" in task_name:
            init_pose = np.array(hp_sim.parse_merging_pose(init_pose, task_spec_dict, 20, 1))
            target_pose = np.array(hp_sim.parse_merging_pose(target_pose, task_spec_dict, 0, 0))
        elif "box_pushing" in task_name:
            init_pose = np.array([init_pose])
            target_pose = np.array([target_pose])
        elif "pushing_T" in task_name:
            init_pose = np.array([init_pose])
            target_pose = np.array([target_pose])
        i = str(i)
        print(f"instance {i}")
        print(f"init_pusher_pos:    \n    {init_pusher_pos.tolist()}")
        print(f"init_pose:          \n    {init_pose.tolist()}")
        print(f"target_pose:        \n    {target_pose.tolist()}")
        if enable_latent:
            target_pusher_pos = target_pusher_pos_list[int(i)]
            print(f"target_pusher_pos:    \n    {target_pusher_pos.tolist()}")
        for method_type in method_types:
            print(f"{'-'*20} {method_type} on instance {i}, num_ReLU: {num_relu} {'-'*20}")
            method_res[method_type] = {}
            if planning_config["plot_only"] or (
                (reload_json_path is not None)
            ):
                print("load from existing results")
                res_reload[method_type] = True
                method_res[method_type] = all_res[num_relu][i][method_type]
            else:
                res_reload[method_type] = False
                torch.cuda.empty_cache()
                # always create new crown for each instance
                if method_type == "CROWN":
                    add_crown_config(config, config_for_planning, model_file)
                method_res[method_type] = planning(
                    config_for_planning,
                    config,
                    open_loop,
                    method_type,
                    model,
                    init_pose,
                    target_pose,
                    init_pusher_pos,
                    vis_file_name_prefix + f"{num_relu}_{i}",
                    target_pusher_pos if enable_latent else None,
                    aemodel if enable_latent else None,
                )
            if abcrown_verbose and method_type == "CROWN":
                inter_result.append(
                    [res["intermediate_best_outputs"] for res in method_res[method_type]["all_res"]]
                )
                space_size_result.append(method_res[method_type]["all_res"][0]["space_size_dict"])
        
        exp_setting = {
            "init_pusher_pos": init_pusher_pos.tolist(),
            "init_pose": init_pose.tolist(),
            "target_pose": target_pose.tolist(),
            "methods": method_types,
            # "horizon": planning_config["horizon"],
        }

        all_res[num_relu][i] = {}
        all_res[num_relu][i].update(exp_setting)
        if enable_latent:
            all_res[num_relu][i]["target_pusher_pos"] = target_pusher_pos.tolist()
        all_reload = True
        for method_type in method_types:
            all_res[num_relu][i][method_type] = method_res[method_type]
            all_reload = all_reload and res_reload[method_type]
            if method_res[method_type]["result_summary"]["success"]:
                for cost_type in cost_types:
                    method_obj_dict[method_type][cost_type].append(
                        method_res[method_type]["result_summary"][cost_type]
                    )
        
        if not all_reload:
            plot_planning(
                all_res[num_relu][i],
                open_loop,
                vis_file_name_prefix + f"{num_relu}_{i}",
                task_spec_dict,
                aemodel if enable_latent else None,
            )

    if abcrown_verbose:
        # may have different number of iterations, to handle in plotting function
        inter_results[num_relu] = inter_result
    stat[num_relu] = {
        cost_type: {method_type: method_obj_dict[method_type][cost_type] for method_type in method_types}
        for cost_type in cost_types
    }
    # except KeyboardInterrupt:
    #     traceback.print_exc()
    #     raise
    # except Exception:
    #     # print the whole error stack
    #     traceback.print_exc()
    #     print(f"saving data to {json_file_path}")
    #     with open(json_file_path, "w") as f:
    #         json.dump(all_res, f, indent=4, separators=(",", ": "))
    #     # delete the tmp vnnlib, compiled vnnlib, warmstart file, output file
    #     remove_files(crown_name_dict, vis_file_name_prefix, config.get("debug", False))
    #     exit(1)
    # save data and visualize
    
    with open(json_file_path, "w") as f:
        json.dump(all_res, f, indent=4, separators=(",", ": "))
    print(f"data saved to {json_file_path}")
    vis_file_name_prefix = os.path.dirname(json_file_path)
    plot_planning_stat(stat, os.path.join(vis_file_name_prefix, "planning_stat.pdf"))
    plot_prediction_error_in_planning(all_res, num_test, os.path.join(vis_file_name_prefix, "prediction_error.pdf"))
    if abcrown_verbose:
        plot_convergence_model(stat, inter_results, method_types, max_iter, os.path.join(vis_file_name_prefix, "convergence.pdf"))
        search_space_list = [space_size_dict["search_space_size"] for space_size_dict in space_size_result]
        total_space_list = [space_size_dict["total_space_size"] for space_size_dict in space_size_result]
        plot_search_space_size(search_space_list, os.path.join(vis_file_name_prefix, "search_space_size.pdf"))
        plot_search_space_size(total_space_list, os.path.join(vis_file_name_prefix, "total_space_size.pdf"))
    remove_files(crown_name_dict, vis_file_name_prefix, config.get("debug", False))


if __name__ == "__main__":
    main()
