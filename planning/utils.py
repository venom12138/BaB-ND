import os
import sys

sys.path.append(os.getcwd())
cwd = os.getcwd()
import time
import torch
import numpy as np
import yaml
import pickle
import seaborn as sns
from typing import List
import json
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf, open_dict
from Verifier_Development.complete_verifier.abcrown import ABCROWN
from auto_LiRPA.bound_ops import BoundLinear, BoundRelu
from others.helper import print_prop, get_task_spec_dict
import tasks.merging_L.l_sim as l_sim
import tasks.pushing_T.t_sim as t_sim
import tasks.box_pushing.box_sim as box_sim
import tasks.inserting.hp_sim as hp_sim
from model.wrapped_model import wrapped_model
from model.wrapped_model_gnn import wrapped_model_gnn
from model.wrapped_model_rope import wrapped_model_rope
try:
    import tasks.rope_3d.rope3d_sim as rope3d_sim
except:
    pass

def reload_res(json_file_path_p):
    all_res = {}
    if os.path.exists(json_file_path_p):
        print(f"Loading data from {json_file_path_p}")
        try:
            with open(json_file_path_p, "r") as f:
                all_res = json.load(f)
        except:
            print("failed to load json file.")
            all_res = {}
    return all_res

def check_state_obs(state_list, state_dim, obs_pos_list, obs_size_list, obs_type, obs_enlarge, scale):
    # state_list: [n_step, state_dim] -> [n_step, num_point, 2] -> [n_step*num_point, 2]
    state_seq = np.array(state_list)[:, :state_dim].reshape(-1, 2)
    obs_pos_list = np.array(obs_pos_list).reshape(-1, 1, 2) * scale
    obs_size_list = np.array(obs_size_list).reshape(-1, 1, 1) * (1 + obs_enlarge) * scale
    state_seq = state_seq[np.newaxis, :, :]

    safe = True
    if obs_type == "circle":
        distances = np.linalg.norm(state_seq - obs_pos_list, axis=-1)
        if np.any(distances < obs_size_list):
            safe = False
    elif obs_type == "square":
        if np.any(np.logical_and(state_seq >= obs_pos_list - obs_size_list, state_seq <= obs_pos_list + obs_size_list)):
            safe = False
    return safe

def get_state(env_state, state_dim, state_scale):
    state = env_state.copy()
    # relative coordinate
    state[0:state_dim:2] -= env_state[state_dim : state_dim + 1]
    state[1:state_dim:2] -= env_state[state_dim + 1 : state_dim + 2]
    return state[:state_dim] / state_scale


def load_crown_config(config, time_tag):
    cwd = os.getcwd()
    planning_config = config["planning"]
    crown_config_prefix = f"{cwd}/crown_configs/{config['task_name']}/"
    os.makedirs(crown_config_prefix, exist_ok=True)
    os.makedirs(crown_config_prefix + f"output/", exist_ok=True)
    os.makedirs(crown_config_prefix + f"vnnlib/", exist_ok=True)
    os.makedirs(crown_config_prefix + f"config/", exist_ok=True)
    # os.makedirs(crown_config_prefix + f"model/", exist_ok=True)
    vnnlib_name = f"vnnlib/{time_tag}.vnnlib"
    # write a dummy vnnlib file for crown
    action_dim = config["data"]["action_dim"]
    with open(f"{crown_config_prefix}{vnnlib_name}", "w") as f:
        print_prop(0, f, action_dim* planning_config["horizon"], [0] * action_dim, [0] * action_dim)
    crown_config_name = crown_config_prefix + f"config/planning_config{time_tag}.yaml"
    # if not os.path.exists(crown_config_name):
    # copy the template crown config file
    os.system(f"cp {crown_config_prefix}/planning_sample.yaml {crown_config_name}")
    # modify the crown config file
    crown_sol_file = crown_config_prefix + f"output/crown_sol{time_tag}.txt"
    warmstart_file = crown_config_prefix + f"output/crown_warmstart{time_tag}.pt"
    bab_file = crown_config_prefix + f"output/crown_bab{time_tag}.pkl"
    with open(crown_config_name, "r") as f:
        crown_config_dict = yaml.load(f, Loader=yaml.Loader)
    crown_config_dict["general"]["root_path"] = crown_config_prefix
    crown_config_dict["general"]["device"] = planning_config["device"]
    crown_config_dict["model"].pop("onnx_path", None)  # remove onnx related configs
    crown_config_dict["specification"]["vnnlib_path"] = vnnlib_name
    crown_config_dict["bab"]["override_timeout"] = planning_config["timeout"]
    crown_config_dict["find_feasible_solution"]["report_per_iter"] = planning_config["abcrown_verbose"]
    crown_config_dict["find_feasible_solution"]["use_upper_bound"] = planning_config["enable_ub"]
    crown_config_dict["find_feasible_solution"]["save_solution_path"] = crown_sol_file
    crown_config_dict["find_feasible_solution"]["save_bab_path"] = bab_file
    crown_config_dict["find_feasible_solution"]["preload_path"] = warmstart_file
    return crown_config_dict, {'crown_config_prefix': crown_config_prefix, 'vnnlib_name': vnnlib_name, 'crown_sol_file': crown_sol_file, 
                               'warmstart_file': warmstart_file, 'crown_config_name': crown_config_name, 'bab_file': bab_file}


def create_general_planning_config(config, crown_name_dict, crown_config_dict):
    config = dict(config)
    def _convert_omega_to_dict(config):
        for key, value in config.items():
            if isinstance(value, DictConfig):
                config[key] = dict(value)
                _convert_omega_to_dict(config[key])
    _convert_omega_to_dict(config)
    task_name = config["task_name"]
    data_config = config["data"]
    planning_config = config["planning"]
    task_spec_dict = get_task_spec_dict(config)
    task_spec_dict["save_img"] = True
    if planning_config["open_loop"]:
        planning_config["n_act_step"] = planning_config["horizon"]
    # planning_config["horizon"] = min(planning_config["horizon"], planning_config["n_sim_step"])
    config_for_planning = {
        "task_name": task_name,
        "task_spec_dict": task_spec_dict,
        "enable_latent": "latent" in config["task_name"],
        "real_exp_config": config["real_exp"],
        "n_sim_step": planning_config["n_sim_step"],
        "horizon": planning_config["horizon"],
        "n_act_step": planning_config["n_act_step"],
        "fixed_horizon": planning_config["fixed_horizon"],
        "use_prev_sol": planning_config["use_prev_sol"],
        "cost_norm": planning_config["cost_norm"],
        "action_bound": planning_config["action_bound"],
        "enable_vis": True,
        "save_result": True,
        "only_final_cost": planning_config["only_final_cost"],
        "action_dim": data_config["action_dim"],
        "state_dim": data_config["state_dim"] if "latent" not in config["task_name"] else config["latent"]["latent_dim"],
        "n_his": config["train"]["n_history"],
        "scale": data_config["scale"],
        "window_size": data_config.get("window_size", None),
        "device": planning_config["device"],
        "seed": config["seed"],
        # below are only used for MPPI
        "model_rollout_fn": None,
        "evaluate_traj_fn": None,
        "verbose": True,
        "mppi_config": planning_config.get("mppi", None),
        "cem_config": planning_config.get("cem", None),
        "gd_config": planning_config.get("gd", None),
        "mip_config": planning_config.get("mip", None),
        "rollout_best": True,
        # only used for CROWN, some are added later
        "warm_start_from_sampling": planning_config["warm_start_from_sampling"],
        "abcrown_verbose": planning_config["abcrown_verbose"],
        "use_empirical_bounds": planning_config["use_empirical_bounds"],
        "crown_name_dict": crown_name_dict,
        "crown_config_dict": crown_config_dict,
    }
    if task_name == "obj_pile" or task_name == "rope_3d":
        if task_name == "obj_pile":
            config_for_planning["noise_level"] = 0.1 if planning_config["mppi"]["noise_level"]=="auto" else planning_config["mppi"]["noise_level"]
        config_for_planning["cost_mode"] = planning_config["cost_mode"]
        config_for_planning["fix_others"] = planning_config["fix_others"]
        config_for_planning["cost_weight"] = planning_config["cost_weight"]
        # config_for_planning["classes"] = data_config["classes"]
        config_for_planning["pusher_lo"] = planning_config["pusher_lo"]
        config_for_planning["pusher_hi"] = planning_config["pusher_hi"]
    return config_for_planning

def create_logging_file(config, save_path, config_for_planning):
    task_name = config["task_name"]
    output_task_name = 'planning'
    planning_config = config["planning"]
    if "folder_tag" in planning_config:
        output_task_name = f"{planning_config['folder_tag']}_"+output_task_name
    real_exp_config = config["real_exp"]
    if "test_id" in planning_config:
        output_task_name += f"_id{planning_config['test_id']}"
    else:
        with open_dict(config):
            planning_config["test_id"] = 0
    os.makedirs(os.path.join(save_path, output_task_name), exist_ok=True)
    enable_real = real_exp_config["enable"]
    # if planning_config["open_loop"]:
    #     output_folder_name = f"open_{planning_config['horizon']}"
    # else:
    #     output_folder_name = f"closed_{planning_config['horizon']}_{planning_config['n_act_step']}_{planning_config['n_sim_step']}"
    # if task_name == "obj_pile":
    #     output_folder_name += f"_mode_{planning_config['cost_mode']}_cls_{config['data']['classes']}"
    #     if planning_config["fix_others"]:
    #         output_folder_name += f"_fix_{planning_config['cost_weight']}"
    # if not planning_config["only_final_cost"]:
    #     output_folder_name += "_acc_cost"
    # if enable_real:
    #     output_folder_name += f"_real_{real_exp_config['exp_id']}"
    #     if not real_exp_config["overwrite"]:
    #         now = time.localtime()
    #         output_folder_name += f"_{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}"
    # output_folder_name += f"_{planning_config['tag']}" if "tag" in planning_config else ""
    now = time.localtime()
    output_folder_name = f"{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}"
    experiment_folder = os.path.join(save_path, output_task_name, output_folder_name)
    os.makedirs(experiment_folder, exist_ok=True)
    json_file_path = os.path.join(experiment_folder, "experiment_results.json")
    vis_file_name_prefix = experiment_folder # os.path.join(experiment_folder, "cases")
    os.makedirs(vis_file_name_prefix, exist_ok=True)
    # vis_file_name_prefix = os.path.join(vis_file_name_prefix, f"{planning_config['num_test']}_")
    with open_dict(config):
        config["output_task_name"] = output_task_name
        config["experiment_folder"] = experiment_folder
    os.system(f"rm -rf {os.path.join(experiment_folder, 'mppi_bf.txt')}")
    config_for_planning["experiment_folder"] = experiment_folder
    with open(os.path.join(experiment_folder, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_object(config), f, default_flow_style=None, sort_keys=False)
    return json_file_path, vis_file_name_prefix

def add_crown_config(config, config_for_planning, model_file_name):
    task_name = config["task_name"]
    planning_config = config["planning"]
    data_config = config["data"]
    train_config = config["train"]
    obs_pos_list = planning_config.get("obs_pos_list", None)
    obs_size_list = planning_config.get("obs_size_list", None)
    obs_type = planning_config.get("obs_type", None)
    obs_enlarge = planning_config.get("obs_enlarge", 0)
    if obs_pos_list is not None:
        obs_pos_list = np.array(obs_pos_list).tolist()
        obs_size_list = np.array(obs_size_list).tolist()
    model_config = {
        "enable_latent": "latent" in task_name,
        "state_dim": data_config["state_dim"] if "latent" not in task_name else config["latent"]["latent_dim"], 
        "action_dim": data_config["action_dim"],
        "n_history": train_config["n_history"],
        "horizon": planning_config["horizon"],
        "cost_norm": planning_config["cost_norm"],
        "only_final_cost": planning_config["only_final_cost"],
        "device": planning_config["device"],
        "obs_pos_list": obs_pos_list,
        "obs_size_list": obs_size_list,
        "obs_type": obs_type,
        "obs_enlarge": obs_enlarge,
    }
    if task_name == "obj_pile":
        n_particle = data_config["state_dim"] // 2
        n_class = data_config["classes"]

        label_list = []
        assert n_particle % n_class == 0
        n_particle_per_class = n_particle // n_class
        for i in range(n_particle):
            label_list.append(i // n_particle_per_class)
        if data_config["push_single"] == True:
            for i in range(1, n_particle):
                label_list[i] = 1
        cls_idx = []
        label_list = np.array(label_list)
        for i in range(n_class):
            cls_idx.append(np.where(label_list == i)[0].tolist())

        action_bound = planning_config["action_bound"]
        pusher_lo = planning_config["pusher_lo"] / data_config["scale"]
        pusher_hi = planning_config["pusher_hi"] / data_config["scale"]
        # action_lb = torch.tensor([pusher_lo, pusher_lo, -action_bound, -action_bound], device=planning_config["device"])
        # action_ub = torch.tensor([pusher_hi, pusher_hi, action_bound, action_bound], device=planning_config["device"])
        # action_norm = (action_ub - action_lb).unsqueeze(0).unsqueeze(0)
        action_lb = [pusher_lo, pusher_lo, -action_bound, -action_bound]
        action_ub = [pusher_hi, pusher_hi, action_bound, action_bound]
        action_norm = np.subtract(action_ub, action_lb, dtype=np.float32).reshape(1, 1, -1).tolist()

        model_config.update({
            "n_particle": n_particle,
            "n_relation": n_particle * n_particle,
            "num_classes": n_class,
            "cls_idx": cls_idx,
            "fix_others": planning_config["fix_others"],
            "cost_weight": planning_config["cost_weight"],
            "action_norm": action_norm,
            "obj_size": data_config["obj_size"] / data_config["scale"],
            "forbidden": planning_config["forbidden"],
            "forbidden_radius": planning_config["forbidden_radius"],
            "far_factor": planning_config["far_factor"],
        })
    elif task_name == "rope_3d":
        task_spec_dict = config_for_planning["task_spec_dict"]
        env, target_pose, forbidden_area, rope_fixed_end, rope_length = init_rope_env(task_spec_dict)
        config_for_planning["env"] = env
        action_bound = planning_config["action_bound"]
        action_lb = [-action_bound] * data_config["action_dim"]
        action_ub = [action_bound] * data_config["action_dim"]  
        action_norm = np.subtract(action_ub, action_lb, dtype=np.float32).reshape(1, 1, -1).tolist()

        model_config.update({
            "target_state": target_pose.tolist(),
            "forbidden_area": forbidden_area.tolist(),
            "rope_fixed_end": rope_fixed_end.tolist(),
            "rope_length": rope_length,
            "action_norm": action_norm
        })
    else:
        model_config.update({
            "penalty_type": planning_config.get("penalty_type", 0),
        })
    crown_name_dict, crown_config_dict = config_for_planning["crown_name_dict"], config_for_planning["crown_config_dict"]
    (
        abcrown,
        cost_model,
        const_nodes_dict
    ) = create_abcrown(
        crown_name_dict,
        crown_config_dict,
        model_file_name,
        model_config,
        planning_config["use_empirical_bounds"],
    )
    with open(os.path.join(config["experiment_folder"], "crown_config.yaml"), "w") as f:
        yaml.dump(crown_config_dict, f, sort_keys=False)
    config_for_planning["abcrown"] = abcrown
    config_for_planning["cost_model"] = cost_model
    config_for_planning["const_nodes_dict"] = const_nodes_dict
    config_for_planning["crown_config_name"] = crown_name_dict["crown_config_name"]
    config_for_planning["crown_config_dict"] = crown_config_dict
    config_for_planning["model_file_name"] = model_file_name
    config_for_planning["model_config"] = model_config

def init_rope_env(task_spec_dict):
        env = rope3d_sim.Rope3dSim(task_spec_dict)
        target_pose = env.get_target_pose()
        forbidden_area=np.concatenate((env.left_forbidden_area, env.right_forbidden_area))
        rope_fixed_end=env.get_rope_fixed_end_coord()
        rope_length=env.get_rope_length()
        return env, target_pose, forbidden_area, rope_fixed_end, rope_length

def get_aemodel(config):
    assert config.planning.aemodel_path is not None, "model_path is not provided"
    aemodel_file = config.planning.aemodel_path
    aemodel = torch.load(aemodel_file, map_location=torch.device(config["planning"]["device"])).eval()
    aemodel.config = config
    return aemodel

def get_cost_types(config):
    if config["planning"]["open_loop"]:
        if config["planning"]["only_final_cost"]:
            cost_types = [
                "planned_cost",
                "actual_cost",
                "cost_diff",
                "best_cost",
                "planned_accumulated_cost",
                "actual_accumulated_cost",
            ]
        else:
            cost_types = [
                "planned_cost",
                "actual_cost",
                "cost_diff",
                "final_cost",
                "best_cost",
            ]
    else:
        cost_types = [
            "planned_accumulated_cost",
            "actual_accumulated_cost",
            "accumulated_cost_diff",
            "final_cost",
            "best_cost",
        ]
    return cost_types

def remove_files(crown_name_dict, exp_dir, debug=False):
    crown_config_prefix = crown_name_dict['crown_config_prefix']
    vnnlib_name = crown_name_dict['vnnlib_name']
    crown_sol_file = crown_name_dict['crown_sol_file']
    warmstart_file = crown_name_dict['warmstart_file']
    crown_config_name = crown_name_dict['crown_config_name']
    crown_bab_file = crown_name_dict['bab_file']
    os.system(f"rm -rf {crown_config_prefix}{vnnlib_name}")
    os.system(f"rm -rf {crown_config_prefix}{vnnlib_name}.compiled")
    os.system(f"rm -rf {crown_config_name}")
    os.system(f"rm -rf {warmstart_file}")
    os.system(f"rm -rf {crown_sol_file}")
    os.system(f"rm -rf {crown_bab_file}")
    print(f"Removed all tmp files in {crown_config_prefix}")
    os.system(f"rm -rf out.txt")
    if debug:
        os.system(f"rm -rf {exp_dir}*")
        print(f"Removed all files in {exp_dir}")


def format_all_res(method_type, action_scale, all_res):
    if "MPPI" in method_type or "CEM" in method_type or "GD" in method_type:
        for res in all_res:
            if not isinstance(res["act_seq"], List):
                res["act_seq"] = (res["act_seq"] * action_scale).tolist()
            if "best_eval_output" in res:
                res["planned_cost"] = -res["best_eval_output"]["rewards"].item()
                res["cost_seqs"] = res["best_eval_output"]["cost_seqs"].squeeze(0).tolist()
                del res["best_model_output"]
                del res["best_eval_output"]
            if "reward_list" in res and not isinstance(res["reward_list"], List):
                res["reward_list"] = res["reward_list"].tolist()
        # mppi specific
        # 'cost_seqs': [n_look_ahead],
    elif method_type == "CROWN":
        for res in all_res:
            if not isinstance(res["act_seq"], List):
                res["act_seq"] = (res["act_seq"] * action_scale).tolist()
            if not isinstance(res["planned_cost"], float):
                res["planned_cost"] = res["planned_cost"].tolist()[0]
            if not isinstance(res["intermediate_best_outputs"], List):
                res["intermediate_best_outputs"] = res["intermediate_best_outputs"].flatten().tolist()
        # crown specific
        # removed: 'intermediate_feasible_sols': intermediate_feasible_sols.tolist(),
        # 'intermediate_feasible_sols': list of feasible_sols in every iteration,
        # 'intermediate_best_outputs': list of best_outputs in every iteration,
    elif method_type == "MIP":
        for res in all_res:
            # unified with MPPI
            if "action_sequence" in res:
                res["act_seq"] = (res["action_sequence"] * action_scale).tolist()
                # res['state_seq'] = (res['observation_sequence']*scale).tolist()
                res["planned_cost"] = res["objective"]
                del res["action_sequence"]
                del res["observation_sequence"]
                del res["objective"]
                del res["varSol"]
        # mip specific
        # 'solve_time': float,
        # 'solve_status': int, 2 for optimal
    return all_res
    # common res format (for json, only dict, list, float, str, int):
    # {'act_seq': [n_look_ahead, action_dim],
    # 'state_seq': [n_look_ahead, state_dim],
    # 'planned_cost': float,
    # return a dict with all the results and parameters


def create_abcrown(
    crown_name_dict, crown_config_dict, model_file_name, model_config, use_empirical_bounds=False
):
    crown_config_name = crown_name_dict["crown_config_name"]
    vnnlib_path = f"{crown_name_dict['crown_config_prefix']}{crown_name_dict['vnnlib_name']}"
    action_dim, horizon = model_config["action_dim"], model_config["horizon"]
    act_seq_dim = action_dim * horizon
    # note that to use cost_model correctly, we should update target_state, known_input, init_pusher_pos in the model
    if "obj_pile" in crown_config_name:
        cost_model = wrapped_model_gnn(model_file_name, model_config)
        crown_config_dict["model"][
            "name"
        ] = f"""Customized("wrapped_model_gnn.py", "wrapped_model_gnn", model_path="{model_file_name}", model_config={model_config})"""
    elif "rope_3d" in crown_config_name:
        cost_model = wrapped_model_rope(model_file_name, model_config)
        crown_config_dict["model"][
            "name"
        ] = f"""Customized("wrapped_model_rope.py", "wrapped_model_rope", model_path="{model_file_name}", model_config={model_config})"""
    else:
        cost_model = wrapped_model(model_file_name, model_config)
        crown_config_dict["model"][
            "name"
        ] = f"""Customized("wrapped_model.py", "wrapped_model", model_path="{model_file_name}", model_config={model_config})"""
    crown_config_dict["model"]["input_shape"] = [1, act_seq_dim]
    with open(crown_config_name, "w") as f:
        yaml.dump(crown_config_dict, f, sort_keys=False)
    with open(vnnlib_path, "w") as f:
        print_prop(0, f, act_seq_dim, [0] * act_seq_dim, [0] * act_seq_dim)
    abcrown = ABCROWN(["--config", crown_config_name])
    if use_empirical_bounds and "obj_pile" not in crown_config_name:
        pre_act_layers = [node.input_name[0] for node in abcrown.model.net.nodes() if isinstance(node, BoundRelu) and isinstance(node.inputs[0], BoundLinear)]
        num_pre_act_layers = len(pre_act_layers)
        num_pre_bounds = len(cost_model.pre_bounds)
        # # pre-activation layers should be # of pre-bounds in dynamics model * horizon
        assert num_pre_act_layers == num_pre_bounds * model_config["horizon"], f"num_pre_act_layers: {num_pre_act_layers}, num_pre_bounds: {num_pre_bounds}, horizon: {model_config['horizon']}"
        reference_bounds = {}
        for i, node_name in enumerate(pre_act_layers):
            reference_bounds[node_name] = cost_model.pre_bounds[i % num_pre_bounds]
        abcrown.model.net.default_reference_bounds = reference_bounds
    return (
        abcrown,
        cost_model,
        get_const_nodes_dict(abcrown, cost_model),
    )

def update_const_nodes(const_nodes_dict, abcrown: ABCROWN, cost_model: wrapped_model_gnn, update_dict):
    for key, value in update_dict.items():
        abcrown.update_constant_node(const_nodes_dict[key], value)
        cost_model.set_const(key, value)
        abcrown.model.model_ori.set_const(key, value)

def get_const_nodes_dict(abcrown, cost_model):
    info_dict = cost_model.get_info()
    node_list_dict = {}
    for key in info_dict:
        node_list = abcrown.find_constant_node_by_value(info_dict[key])
        node_list_dict[key] = node_list
    return node_list_dict

def get_pose_from_state(curr_state, task_spec_dict):
    # curr_state: [B, 1, state_dim]
    # task_spec_dict: dict
    assert len(curr_state.shape) == 3 and curr_state.shape[1] == 1
    state_dim = curr_state.shape[-1]
    keypoints = curr_state.reshape(-1, state_dim // 2, 2)
    task_name = task_spec_dict["task_name"]
    if "merging_L" in task_name:
        poses = l_sim.get_pose_from_keypoints(keypoints[:, :4], task_spec_dict)
        poses = torch.cat([poses, l_sim.get_pose_from_keypoints(keypoints[:, 4:], task_spec_dict)], dim=1)
    elif "pushing_T" in task_name:
        poses = t_sim.get_pose_from_keypoints(keypoints, task_spec_dict)
    elif "box_pushing" in task_name:
        poses = box_sim.get_pose_from_keypoints(keypoints, task_spec_dict)
    elif "inserting" in task_name:
        poses = hp_sim.get_pose_from_keypoints(keypoints[:, :5], task_spec_dict)
        poses = torch.cat([poses, hp_sim.get_pose_from_keypoints(keypoints[:, 5:], task_spec_dict)], dim=1)
    else:
        raise NotImplementedError
    return poses


def sim_to_real_coord_2d(sim_coord, sim_to_real_scale=1000):
    return (np.array(sim_coord).reshape(-1, 2) / sim_to_real_scale - np.array([-0.2732, 0.2008])).flatten()


def real_to_sim_coord_2d(real_coord, sim_to_real_scale=1000):
    return sim_to_real_scale * (np.array(real_coord).reshape(-1, 2) + np.array([-0.2732, 0.2008])).flatten()


def analyze_search_progress(bab_file, vis_file_name_prefix, file_format="png"):
    assert file_format in ["png", "pdf", "gif"]
    bab_bounds = []
    # copy bab_file for backup, add a time tag after vis_file_name_prefix
    bab_file_backup = vis_file_name_prefix + f"_backup_{time.strftime('%H%M%S')}.pkl"
    os.system(f"cp {bab_file} {bab_file_backup}")
    # load bab bounds
    with open(bab_file, 'rb') as f:
        while True:
            try:
                bab_bounds.append(pickle.load(f))
            except EOFError:
                break
    global_x_L = bab_bounds[0]["x_L"]
    global_x_U = bab_bounds[0]["x_U"]
    bab_prefix = vis_file_name_prefix+"_bab/"
    os.makedirs(bab_prefix, exist_ok=True)
    space_size_list = [1]
    total_space_size_list = [1]
    bab_bounds = bab_bounds[1:]
    for bound_dict in bab_bounds:
        # print(f"------------- Iteration {bound_dict['num_iter']} -------------")
        domain_dict = analyze_domain_distribution(global_x_L, global_x_U, bound_dict["x_L"], bound_dict["x_U"], bab_prefix, bound_dict['num_iter'], file_format == "pdf")
        space_size_list.append(domain_dict["space_size"])
        total_space_size_list.append(bound_dict["total_space_size"])
    
    if file_format == "gif":
        os.system(f"convert -delay 100 -loop 0 {bab_prefix}*_domain_focus.png {bab_prefix}domain_focus.gif")
        os.system(f"convert -delay 100 -loop 0 {bab_prefix}*_domain_distance.png {bab_prefix}domain_distance.gif")
        os.system(f"convert -delay 100 -loop 0 {bab_prefix}*_domain_cosine_similarity.png {bab_prefix}domain_cosine_similarity.gif")

    iterations = list(range(1, len(space_size_list) + 1))
    plt.figure()
    # sns.set_palette("flare", 1)
    plt.plot(iterations, space_size_list)
    plt.title('Search space size over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Search space size')
    plt.ylim(0, 1)
    plt.tight_layout()
    plot_file = bab_prefix + 'search_space_size.png'
    if file_format == "pdf":
        plot_file = bab_prefix + 'search_space_size.pdf'
    plt.savefig(plot_file)
    plt.close()
    print(f"Search space size saved to {plot_file}")

    plt.figure()
    # sns.set_palette("flare", 1)
    plt.plot(iterations, total_space_size_list)
    plt.title('Total space size over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Total space size')
    plt.ylim(0, 1)
    plt.tight_layout()
    plot_file = bab_prefix + 'total_space_size.png'
    if file_format == "pdf":
        plot_file = bab_prefix + 'total_space_size.pdf'
    plt.savefig(plot_file)
    plt.close()
    print(f"Total space size saved to {plot_file}")

    return {"search_space_size": space_size_list, "total_space_size": total_space_size_list}


def analyze_domain_distribution(global_x_L, global_x_U, x_L, x_U, vis_file_name_prefix, num_iter, save_pdf=False):
    vis_file_name_prefix = vis_file_name_prefix + f"{num_iter:03d}_"
    num_domains, num_inputs = x_L.shape[:2]
    
    draw_dis = False
    draw_cos = False
    print_dom = False
    draw_dim = False

    # [num_domain, num_input]
    normed_x_L = (x_L - global_x_L) / (global_x_U - global_x_L)
    normed_x_U = (x_U - global_x_L) / (global_x_U - global_x_L)
    domain_center = (normed_x_L + normed_x_U) / 2
    # sort domain center by distance to origin
    sorted_indices = torch.norm(domain_center, p=1, dim=1).argsort()
    domain_center = domain_center[sorted_indices]
    normed_x_L = normed_x_L[sorted_indices]
    normed_x_U = normed_x_U[sorted_indices]

    # heatmap of distances
    domain_center_for_dist = domain_center.transpose(0, 1).unsqueeze(-1)
    # [num_input, num_domain, num_domain]
    distances_per_input = torch.cdist(domain_center_for_dist, domain_center_for_dist, p=1)
    # [num_domain, num_domain]
    distances = distances_per_input.sum(dim=0)
    heatmap_size = max(num_domains, 10)
    if draw_dis:
        plt.figure(figsize=(heatmap_size, heatmap_size))
        sns.heatmap(distances.cpu().numpy(), annot=True, fmt=".2f")
        plt.title(f'Distance between domain centers in iteration {num_iter}')
        plt.xlabel('Domain index')
        plt.ylabel('Domain index')
        plt.tight_layout()
        plot_file = vis_file_name_prefix + 'domain_distance.png'
        if save_pdf:
            plot_file = vis_file_name_prefix + 'domain_distance.pdf'
        plt.savefig(plot_file)
        plt.close()
        print(f"Domain distance saved to {plot_file}")
    
    # heatmap of cosine similarity
    domain_center_for_cos = 2 * domain_center - 1
    domain_center_for_cos = domain_center_for_cos / torch.norm(domain_center_for_cos, p=2, dim=1, keepdim=True)
    cosine_similarity = torch.mm(domain_center_for_cos, domain_center_for_cos.transpose(0, 1))
    if draw_cos:
        plt.figure(figsize=(heatmap_size, heatmap_size))
        sns.heatmap(cosine_similarity.cpu().numpy(), vmin=-1, vmax=1, annot=True, fmt=".2f")
        plt.title('Cosine similarity between domain centers')
        plt.xlabel('Domain index')
        plt.ylabel('Domain index')
        plt.tight_layout()
        plot_file = vis_file_name_prefix + 'domain_cosine_similarity.png'
        if save_pdf:
            plot_file = vis_file_name_prefix + 'domain_cosine_similarity.pdf'
        plt.savefig(plot_file)
        plt.close()
        print(f"Domain cosine similarity saved to {plot_file}")

    # [num_domain, num_input]
    domain_distribution = torch.stack([normed_x_L,normed_x_U], -1)
    covered_space = normed_x_U - normed_x_L
    # [num_domain]
    space_size = covered_space.prod(-1)
    total_space_size = space_size.sum().item()
    if print_dom:
        for i in range(num_domains):
            # print(f"Domain {i}: {normed_x_L[i].flatten().tolist()} to {normed_x_U[i].flatten().tolist()}")
            print(f"Domain {i}")
            print(f"  Center: {domain_center[i].flatten().tolist()}")
            print(f"  Converage: {covered_space[i].flatten().tolist()}")
            print(f"  Space size: {space_size[i].item()}")
    # print(f"Search space ratio: {total_space_size:.4f}")

    domain_distribution = 2 * domain_distribution - 1
    num_samples = 10000 + 1
    x_values = torch.linspace(-1, 1, steps=num_samples, device=domain_distribution.device)
    domain_distribution = domain_distribution.unsqueeze(2)
    is_within_bounds = (x_values >= domain_distribution[..., 0]) & (x_values < domain_distribution[..., 1])
    # [num_inputs, num_samples]
    domain_sum = is_within_bounds.sum(dim=0).float()/ num_domains
    
    x_values = x_values.cpu().numpy()
    domain_sum = domain_sum.cpu().numpy()
    divide_xy = False
    if draw_dim:
        if divide_xy:
            previous_sum_x = np.zeros(num_samples)
            previous_sum_y = np.zeros(num_samples)
            plt.figure(figsize=(20, num_inputs))
            sns.set_palette("crest", num_inputs)
            ax_x = plt.subplot(1, 2, 1)  # For x dimensions
            ax_y = plt.subplot(1, 2, 2)  # For y dimensions
            for i in range(num_inputs):
                if i % 2 == 0:
                    ax = ax_x
                    previous_sum = previous_sum_x
                else:
                    ax = ax_y
                    previous_sum = previous_sum_y
                # draw axis 
                ax.axhline(y=i, xmin=-1, xmax=1, color='black', linestyle='--', linewidth=0.5)
                ax.fill_between(x_values, previous_sum, previous_sum + domain_sum[i], label=f'Dim {i}') 
                previous_sum += 1
            for ax in [ax_x, ax_y]:
                ax.set_xlim(-1, 1)
                ax.set_ylim(0, num_inputs//2)
                ax.set_xlabel('Normalized input range')
                ax.set_ylabel('Stacked sum of domain indicators')
                ax.legend(loc='upper left')
            ax_x.set_title("X dimensions")
            ax_y.set_title("Y dimensions")
            plt.suptitle(f'Stacked domain focus across all input dimensions in iteration {num_iter}')
        else:
            previous_sum = np.zeros(num_samples)
            plt.figure(figsize=(10, num_inputs*2))
            # use sns color palette, crest, flare
            sns.set_palette("crest", num_inputs)
            for i in range(num_inputs):
                # draw axis 
                plt.axhline(y=i, xmin=-1, xmax=1, color='black', linestyle='--', linewidth=0.5)
                plt.fill_between(x_values, previous_sum, previous_sum + domain_sum[i], label=f'Dim {i}')
                previous_sum += 1
            plt.xlim(-1, 1)
            plt.ylim(0, num_inputs)
            plt.title(f'Stacked domain focus across all (splitted) input dimensions in iteration {num_iter}')
            plt.xlabel('Normalized input range')
            plt.ylabel('Stacked sum of domain indicators')
            plt.legend(loc='upper left')
            plt.tight_layout()
        plot_file = vis_file_name_prefix + 'domain_focus.png'
        if save_pdf:
            plot_file = vis_file_name_prefix + 'domain_focus.pdf'
        plt.savefig(plot_file)
        plt.close()
        print(f"Domain focus saved to {plot_file}")
    return {"space_size": total_space_size}

def align_observation(state: np.ndarray, gt_model: np.ndarray):
    # state: [B, state_dim]
    # gt_model: [num_object, num_keypoints, 2]
    # state_dim = num_object * num_keypoints * 2
    num_case = state.shape[0]
    num_object, num_keypoints = gt_model.shape[:2]
    state = state.reshape(num_case, num_object, num_keypoints, 2)
    gt_model = gt_model.reshape(1, num_object, num_keypoints, 2)
    obs_centers = state.mean(axis=-2, keepdims=True)
    obs_centered = state - obs_centers
    gt_centers = gt_model.mean(axis=-2, keepdims=True)
    gt_centered = gt_model - gt_centers

    U, _, Vt = np.linalg.svd(np.matmul(gt_centered.transpose(0, 1, 3, 2), obs_centered), full_matrices=False)
    R = np.matmul(Vt.transpose(0, 1, 3, 2), U.transpose(0, 1, 3, 2))
    det_R = np.linalg.det(R)
    corrections = det_R < 0
    Vt[corrections, -1, :] *= -1
    R = np.matmul(Vt.transpose(0, 1, 3, 2), U.transpose(0, 1, 3, 2))
    t = obs_centers - np.matmul(gt_centers, R.transpose(0, 1, 3, 2))
    model_aligned = np.matmul(gt_model, R.transpose(0, 1, 3, 2)) + t
    return model_aligned.reshape(num_case, -1)

    # num_inputs_to_plot = num_inputs
    # if num_inputs > 10:
    #     # only plot inputs been splitted: not covered_space == 1 for all domains
    #     splitted_mask = torch.any(covered_space != 1, dim=0)
    #     splitted_indices = torch.where(splitted_mask)[0]
    #     domain_sum = domain_sum[splitted_mask]
    #     num_inputs_to_plot = len(splitted_indices)
    
    # x_values = x_values.cpu().numpy()
    # domain_sum = domain_sum.cpu().numpy()
    # previous_sum = np.zeros(num_samples)
    # if draw_dim:
    #     plt.figure(figsize=(10, num_inputs_to_plot*2))
    #     # use sns color palette, crest, flare
    #     sns.set_palette("crest", num_inputs_to_plot)
    #     for i in range(num_inputs_to_plot):
    #         # draw axis 
    #         plt.axhline(y=i, xmin=-1, xmax=1, color='black', linestyle='--', linewidth=0.5)
    #         plt.fill_between(x_values, previous_sum, previous_sum + domain_sum[i], label=f'Dim {i}' if num_inputs <= 10 else f'Dim {splitted_indices[i]}')
    #         previous_sum += 1
    #     plt.xlim(-1, 1)
    #     plt.ylim(0, num_inputs_to_plot)
    #     plt.title(f'Stacked domain focus across all (splitted) input dimensions in iteration {num_iter}')
    #     plt.xlabel('Normalized input range')
    #     plt.ylabel('Stacked sum of domain indicators')
    #     plt.legend(loc='upper left')
    #     plt.tight_layout()
    #     plot_file = vis_file_name_prefix + 'domain_focus.png'
    #     if save_pdf:
    #         plot_file = vis_file_name_prefix + 'domain_focus.pdf'
    #     plt.savefig(plot_file)
    #     plt.close()
    #     print(f"Domain focus saved to {plot_file}")
