import os
import sys

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

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

from model.model import Rope_MLP, PyG_GNN
# torch.autograd.set_detect_anomaly(True)
# task specific
import planning.planning_for_pile as planning_for_pile
# from planning.planning_for_rope import planning
from others.plotter import (
    plot_planning_stat,
    plot_convergence_model,
    plot_search_space_size,
    plot_prediction_error_in_planning,
)
from others.helper import generate_test_cases, _dict_convert_np_to_list
from others.helper import plot_obj_pile, plot_cost, plot_task
from planning.utils import *

@hydra.main(version_base=None, config_path="configs", config_name="default_config")
def main(config: DictConfig):
    assert config.planning.model_path is not None, "model_path is not provided"
    save_path = '/'.join(config.planning.model_path.split('/')[:-1])
    # config_path = save_path + '/config.yaml'
    # # load config file
    # with open(config_path, 'r') as f:
    #     new_config = yaml.safe_load(f)
    # new_config['planning'] = config['planning']
    # # replace new params
    # if config['task_name'] == 'obj_pile':
    #     new_config['data']['classes'] = config['data']['classes']
    
    # config = DictConfig(new_config)
    config["training"] = False
    planning_config = config["planning"]
    task_name = config["task_name"]
    assert task_name in ["obj_pile", "rope_3d"], "task_name not supported"
    if task_name == "obj_pile":
        planning = planning_for_pile.planning
    elif task_name == "rope_3d":
        import planning.planning_for_rope as planning_for_rope
        planning = planning_for_rope.planning
    method_types = list(planning_config["method_types"])

    force_method_types = planning_config["force_method_types"]
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

    num_test = planning_config["num_test"]
    if task_name == "rope_3d":
        assert len(method_types) == 1 and num_test == 1, "pyflex can only run one instance at a time"
    # load result file
    all_res = {}
    if os.path.exists(json_file_path):
        print(f"Loading data from {json_file_path}")
        try:
            with open(json_file_path, "r") as f:
                all_res = json.load(f)
        except:
            print("failed to load json file.")
            all_res = {}

    stat = {}
    inter_results = {}
    cost_types = get_cost_types(config)
    # since crown will reset seeds, we need to generate test instances before running crown
    init_pose_list, target_pose_list, init_pusher_pos_list, _ = generate_test_cases(
        config["seed"], num_test, task_spec_dict, config['planning']['test_id'], config['planning']['cost_mode']
    )

    # # main loop
    # try:
    model_file = config.planning.model_path
    model_name = model_file.split('/')[-1].split('.')[0]
    model = torch.load(model_file, map_location=config.planning.device).eval()
    model.config = config
    
    print(f"load model from {model_file}, ")
    inter_result = []
    space_size_result = []

    method_obj_dict = {method_type: {cost_type: [] for cost_type in cost_types} for method_type in method_types}
    if model_name not in all_res:
        all_res[model_name] = {}
    for i in range(num_test):
        method_res = {}
        res_reload = {}
        init_pusher_pos = np.array(init_pusher_pos_list[i]).tolist()
        init_pose = np.array(init_pose_list[i]).tolist()
        target_pose = np.array(target_pose_list[i]).tolist() if target_pose_list is not None else None 
        
        i = str(i)
        print(f"instance {i}")
        print(f"init_pusher_pos:    \n    {init_pusher_pos}")
        print(f"init_pose:          \n    {init_pose}")
        print(f"target_pose:        \n    {target_pose}")
        for method_type in method_types:
            print(f"{'-'*20} {method_type} on instance {i},  {'-'*20}")
            method_res[method_type] = {}
            if planning_config["plot_only"] or (planning_config["reload"]
                and i in all_res[model_name]
                and method_type not in force_method_types
                and method_type in all_res[model_name][i]
                and all_res[model_name][i]["init_pusher_pos"] == init_pusher_pos
                and all_res[model_name][i]["init_pose"] == init_pose
                and all_res[model_name][i]["target_pose"] == target_pose
            ):
                print("load from existing results")
                res_reload[method_type] = True
                method_res[method_type] = all_res[model_name][i][method_type]
            else:
                res_reload[method_type] = False
                # always create new crown for each instance
                if method_type == "CROWN":
                    add_crown_config(config, config_for_planning, model_file)
                else:
                    if "rope_3d" in task_name:
                        config_for_planning["env"] = init_rope_env(task_spec_dict)[0]
                
                method_res[method_type] = planning(
                    config_for_planning,
                    config, 
                    open_loop,
                    method_type,
                    model,
                    init_pose,
                    target_pose,
                    init_pusher_pos,
                    vis_file_name_prefix + f"{i}",
                )
            if abcrown_verbose and method_type == "CROWN":
                inter_result.append(
                    [res["intermediate_best_outputs"] for res in method_res[method_type]["all_res"]]
                )
                space_size_result.append(method_res[method_type]["all_res"][0]["space_size_dict"])

        exp_setting = {
            "init_pusher_pos": init_pusher_pos,
            "init_pose": init_pose,
            "target_pose": target_pose,
            "methods": method_types,
            # "horizon": planning_config["horizon"],
        }

        all_res[model_name][i] = {}
        all_res[model_name][i].update(exp_setting)
        all_reload = True

        for method_type in method_types:
            all_res[model_name][i][method_type] = method_res[method_type]
            all_reload = all_reload and res_reload[method_type]
            if method_res[method_type]["result_summary"]["success"]:
                for cost_type in cost_types:
                    method_obj_dict[method_type][cost_type].append(
                        method_res[method_type]["result_summary"][cost_type]
                    )
        if not all_reload and not config["planning"]["gather_data"]:
            save_path = os.path.join('/'.join(json_file_path.split('/')[:-1]), "cases")
            if task_name == "obj_pile":
                for method_type in method_types:
                    state_gt = np.array(all_res[model_name][i][method_type]['result_summary']['gt_states'])[:, :config["data"]["state_dim"]] # seq, state_dim (not scaled)
                    state_pred = np.array(all_res[model_name][i][method_type]['result_summary']['pred_state_seq']) # seq-1, state_dim
                    state_pred = np.concatenate([state_gt[0:1], state_pred], axis=0) # add initial state (not scaled)
                    action_seq = np.array(all_res[model_name][i][method_type]['result_summary']['exec_action_seq']) # seq-1, action_dim (not scaled)
                    pusher_poses = np.array(all_res[model_name][i][method_type]['result_summary']['gt_states'])[:, -4:-2]
                    labels = np.array(all_res[model_name][i][method_type]['result_summary']['labels'])
                    plot_obj_pile(np.expand_dims(state_gt, axis=0), np.expand_dims(state_pred, axis=0), \
                            np.expand_dims(pusher_poses, axis=0), np.expand_dims(action_seq, axis=0), \
                            config, int(i), save_path=save_path, labels=labels, \
                            filename=f"{num_test}_{i}_cmp_{method_type}_{all_res[model_name][i][method_type]['result_summary']['final_cost']:.3f}")
                    pred_cost_seq = np.array(all_res[model_name][i][method_type]['result_summary']['planned_cost_seq'])
                    gt_cost_seq = np.array(all_res[model_name][i][method_type]['result_summary']['actual_cost_seq'])
                    cost_diff_seq = np.array(all_res[model_name][i][method_type]['result_summary']['cost_diff_seq'])
                    plot_cost(pred_cost_seq, gt_cost_seq, cost_diff_seq, \
                            save_path=save_path, \
                            filename=f"{num_test}_{i}_cost_{method_type}_{all_res[model_name][i][method_type]['result_summary']['final_cost']:.3f}.png")
            elif task_name == "rope_3d":
                # import pdb; pdb.set_trace()
                target_state = all_res[model_name][i][method_type]['exp_setting']['target_pose'].flatten()
                state_gt = np.array(all_res[model_name][i][method_type]['result_summary']['gt_states'])[:, :config["data"]["state_dim"]] # seq, state_dim (not scaled)
                state_pred = np.array(all_res[model_name][i][method_type]['result_summary']['pred_state_seq']) # seq-1, state_dim
                state_pred = np.concatenate([state_gt[0:1], state_pred], axis=0) # add initial state (not scaled)
                action_seq = np.array(all_res[model_name][i][method_type]['result_summary']['exec_action_seq']) # seq-1, action_dim (not scaled)
                pusher_poses = np.array(all_res[model_name][i][method_type]['result_summary']['gt_states'])[:, -6:-3]

                with open("rope.pkl", "wb") as f:
                    pickle.dump({"state_gt": state_gt, "state_pred": state_pred, "pusher_poses": pusher_poses, "action_seq": action_seq}, f)
                kwargs = {"state_gt": np.expand_dims(state_gt, axis=0), \
                        "state_pred": np.expand_dims(state_pred, axis=0), \
                        "pusher_poses": np.expand_dims(pusher_poses, axis=0), \
                        "action_seq": np.expand_dims(action_seq, axis=0), \
                        "config": config, \
                        "start_idx": int(i), \
                        "filename":f"{config['planning']['num_test']}_{i}_vis_{method_type}_{round(all_res[model_name][i][method_type]['result_summary']['final_cost'], 3):.3f}", \
                        "save_path": save_path, \
                        "dim_of_work_space": 3, \
                        "target_state": np.expand_dims(target_state, axis=0), \
                        "forbidden_area": np.array(all_res[model_name][i][method_type]['result_summary']['forbidden_area'])
                        }
                plot_task(task_name=config['task_name'], **kwargs)
                pred_cost_seq = np.array(all_res[model_name][i][method_type]['result_summary']['planned_cost_seq'])
                gt_cost_seq = np.array(all_res[model_name][i][method_type]['result_summary']['actual_cost_seq'])
                cost_diff_seq = np.array(all_res[model_name][i][method_type]['result_summary']['cost_diff_seq'])
                plot_cost(pred_cost_seq, gt_cost_seq, cost_diff_seq, \
                        save_path=save_path, \
                        filename=f"{config['planning']['num_test']}_{i}_cost_{method_type}_{round(all_res[model_name][i][method_type]['result_summary']['final_cost'], 3):.3f}.png")
            else:
                raise NotImplementedError
    if abcrown_verbose:
        # may have different number of iterations, to handle in plotting function
        inter_results[model_name] = inter_result
    stat[model_name] = {
        cost_type: {method_type: method_obj_dict[method_type][cost_type] for method_type in method_types}
        for cost_type in cost_types
    }

    all_res = _dict_convert_np_to_list(all_res)
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