import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
import torch
import torch.nn.functional as F
import yaml
from functools import partial
import json
from model.wrapped_model import wrapped_model
from model.mlp import MLP
from model.encoder import AE
from planning.sampling_planner import Sampling_Planner
from others.helper import *
from others.meter import CostMeter
from Verifier_Development.complete_verifier.abcrown import ABCROWN

from base_sim import Base_Sim
import tasks.merging_L.l_sim as l_sim
import tasks.pushing_T.t_sim as t_sim
import tasks.box_pushing.box_sim as box_sim
import tasks.inserting.hp_sim as hp_sim
from planning.utils import *

# for debug only
from others.plotter import update_plot, draw_object_and_pusher

def planning(
    config,
    original_config,
    open_loop,
    method_type,
    model: MLP,
    init_pose,
    target_pose,
    init_pusher_pos,
    vis_file_name_prefix=None,
    target_pusher_pos=None,
    aemodel: AE = None,
):
    os.makedirs(vis_file_name_prefix, exist_ok=True)
    assert method_type in ["MPPI_GD", "MPPI", "GD", "CROWN", "MPPI_BF", "CEM", "DecentCEM"]
    start = time.time()
    task_name = config["task_name"]
    real_exp_config = config["real_exp_config"]
    enable_real = real_exp_config["enable"]
    enable_latent = config["enable_latent"]
    assert not enable_latent
    hier_config = original_config["planning"]["hierarchical"]
    enable_hier = hier_config["enable"]
    assert enable_hier == True
    enable_vis, save_result = config["enable_vis"], config["save_result"]
    use_prev_sol, fixed_horizon = config["use_prev_sol"], config["fixed_horizon"]
    use_prev_sol = False
    only_final_cost, cost_norm = config["only_final_cost"], config["cost_norm"]
    assert cost_norm in [1, 2]
    n_his = config["n_his"]
    assert n_his == 1
    n_sim_step = config["n_sim_step"]
    subgoal_interval = hier_config["subgoal_interval"] # steps interval between subgoals
    num_subgoal = hier_config["num_subgoal"]
    horizon = subgoal_interval * num_subgoal
    device, seed = config["device"], config["seed"]
    state_dim, action_dim, long_action_bound = config["state_dim"], config["action_dim"], config["action_bound"]
    mppi_config = config["mppi_config"]
    long_noise_level = mppi_config["noise_level"]
    long_action_lb = torch.tensor([-long_action_bound] * action_dim, device=device)
    long_action_ub = torch.tensor([long_action_bound] * action_dim, device=device)
    # scale between pixel and state space. 100 pixel -> 1 in state space
    scale = config["scale"]
    # state and action are in pixel, need to scale
    state_scale, action_scale = scale, scale
    window_size = config["window_size"] #TODO: what is window_size
    pusher_pos_lb, pusher_pos_ub = window_size * 0.05, window_size * 0.95
    warm_start_from_sampling, warmstart_file = config["warm_start_from_sampling"], config["crown_name_dict"]["warmstart_file"]
    task_spec_dict = config["task_spec_dict"]

    if "merging_L" in task_name:
        generate_init_target_states_func = l_sim.generate_init_target_states
        sim_class = l_sim.L_Sim
        keypoint_offsets_2d = l_sim.get_offests_w_origin(task_spec_dict)
        keypoint_offsets_2d = np.array([keypoint_offsets_2d, keypoint_offsets_2d])
    elif "pushing_T" in task_name:
        generate_init_target_states_func = t_sim.generate_init_target_states
        sim_class = t_sim.T_Sim
        keypoint_offsets_2d = [t_sim.get_offests_w_origin(task_spec_dict)]
    elif "box_pushing" in task_name:
        generate_init_target_states_func = box_sim.generate_init_target_states
        sim_class = box_sim.BoxSim
        keypoint_offsets_2d = [box_sim.get_offests_w_origin(task_spec_dict)]
    elif "inserting" in task_name:
        generate_init_target_states_func = hp_sim.generate_init_target_states
        sim_class = hp_sim.HP_Sim
        keypoint_offsets_2d = hp_sim.get_offests_w_origin(task_spec_dict)
    else:
        raise NotImplementedError
    # keypoint_offsets_2d: in pixel or mm in real world
    keypoint_offsets_2d = np.array(keypoint_offsets_2d)

    init_res = config.get("init_res", None)
    preloaded = init_res is not None
    
    penalty_type = original_config.planning.get("penalty_type", 0)
    obs_pos_list = None
    obs_size_list = None
    obs_type = None
    obs_enlarge = 0
    if "obs_pos_list" in original_config.planning and penalty_type != 0:
        obs_pos_list = np.array(original_config.planning["obs_pos_list"]).tolist()
        obs_size_list = np.array(original_config.planning["obs_size_list"]).tolist()
        obs_type = original_config.planning.get("obs_type", "circle")
        obs_enlarge = original_config.planning.get("obs_enlarge", 0)
        task_spec_dict["obs_pos_list"] = np.array(obs_pos_list)*scale
        task_spec_dict["obs_size_list"] = np.array(obs_size_list)*scale
        task_spec_dict["obs_type"] = obs_type
        task_spec_dict["obs_enlarge"] = obs_enlarge

    # init_state: [state_dim], target_state: [state_dim]
    init_state, target_state = generate_init_target_states_func(init_pose, target_pose, task_spec_dict, original_config["train"]["include_com"])
    enable_visualize = False # True #  False 
    if enable_real:
        from real_exp_new.real_wrapper_mp import RealWrapper, get_visible_areas
        param_dict = {
            "reset": False, 
            "capture_fps": 5,
            # "record_fps": 15,
            # "record_time": 10,
        }
        # serial_numbers = ["246322301893", "246322303954", "311322300308"]
        # serial_numbers = ["246322301893", ] # "311322303615", 
        serial_numbers = ["311322303615", "215122252880", "151422251501", "246322303954"]
        # centered by robot: RF, LB, LF, RB
        # visible_areas = get_visible_areas(serial_numbers)
        visible_areas = None # TODO: get visible areas
        env = RealWrapper(param_dict, serial_numbers, \
                        visible_areas, real_exp_config["object_path_list"], \
                        real_exp_config["block_color_list"], target_state)
        env.start()
        time.sleep(1)
        print(f"start real env finish")
        # scale: 500 pixel -> 500mm in real world
        # state_to_real_scale = 1
        image_index = 0
        real_image_prefix = vis_file_name_prefix + "_real_step_"
        print(f"real_image_prefix: {real_image_prefix}")
        # use preception to get the initial state
        env.init_pusher_pos(*init_pusher_pos)
        time.sleep(1)
        init_env_state = env.get_state(
            keypoint_offsets_2d,
            image_path=f"{real_image_prefix}{image_index}.jpeg",
            visualize=enable_visualize,
        )

        image_index += 1
        init_env_state = np.concatenate([init_env_state, [0] * action_dim])
        env_state = init_env_state
    else:
        env: Base_Sim = sim_class(task_spec_dict, init_pose, target_pose, init_pusher_pos)
        # assert np.allclose(target_state[:state_dim], np.array(env.get_all_object_keypoints(True)).flatten())
        # allow the simulator to a resting position
        env_state = None
        for i in range(2):
            env_dict = env.update((init_pusher_pos[0], init_pusher_pos[1]), rel=False)
            env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"], env_dict["action"]], axis=0)
            env.wait(1)

    scaled_target_state = target_state / state_scale
    scaled_target_state_tensor = torch.tensor(scaled_target_state, dtype=torch.float32, device=device)

    # Initialize the planner mppi
    config["planner_type"] = method_type
    if method_type == "CROWN" and warm_start_from_sampling:
        config["planner_type"] = original_config["planning"].get("warm_start_method", "MPPI")
    config["model_rollout_fn"] = partial(rollout, model=model)

    reward_config = {
        "action_scale": action_scale,
        "state_dim": state_dim,
        "enable_latent": enable_latent,
        "cost_norm": cost_norm,
        "only_final_cost": only_final_cost,
        "target_state": scaled_target_state_tensor,
        "penalty_type": penalty_type,
        "obs_pos_list": obs_pos_list,
        "obs_size_list": obs_size_list,
        "obs_type": obs_type,
        "obs_enlarge": obs_enlarge,
    }
    config["evaluate_traj_fn"] = partial(reward, reward_config=reward_config)
    config["action_lower_lim"] = long_action_lb
    config["action_upper_lim"] = long_action_ub
    config["n_look_ahead"] = horizon
    planner_mppi = Sampling_Planner(config)
    if not preloaded and method_type == "CROWN":
        abcrown: ABCROWN = config["abcrown"]
        cost_model: wrapped_model = config["cost_model"]
        const_nodes_dict = config["const_nodes_dict"]
        crown_config_dict = config["crown_config_dict"]
        model_config = config["model_config"]
        crown_name_dict, crown_config_dict = config["crown_name_dict"], config["crown_config_dict"]
        crown_config_prefix = crown_name_dict["crown_config_prefix"]
        vnnlib_name = crown_name_dict["vnnlib_name"]
        vnnlib_path = f"{crown_config_prefix}{vnnlib_name}"
        crown_config_name = crown_name_dict["crown_config_name"]
        model_file_name = config["model_file_name"]
        crown_sol_file = crown_name_dict["crown_sol_file"]
        abcrown_verbose = config["abcrown_verbose"]
        if abcrown_verbose:
            bab_file = crown_name_dict["bab_file"]
        use_empirical_bounds = config["use_empirical_bounds"]
        update_const_nodes(const_nodes_dict, abcrown, cost_model, {"target_state": scaled_target_state_tensor.unsqueeze(0)})

    all_res = []
    cost_meter = CostMeter()
    exec_action_seq, pred_state_seq = [], []
    success = False
    # for real, records state before planning
    gt_states = []
    # number of steps to actuate the pusher, increase after planning, decrease after acting
    n_to_act = 0

    # in hierarchical planning, we first launch the long-horizon planner MPPI-like or CROWN to get the subgoals (open loop)
    # then we launch the short-horizon MPPI planner to reach the subgoals (closed loop)
    reset_seed(seed)
    curr_cost = step_cost(env_state[:state_dim] / state_scale, scaled_target_state, cost_norm)
    state = get_state(env_state, state_dim, state_scale)
    x_pusher_for_plan, y_pusher_for_plan = env_state[state_dim : state_dim + 2]
    enable_sampling = "MPPI" in method_type or "GD" in method_type or 'CEM' in method_type or (method_type == "CROWN" and warm_start_from_sampling)
    if preloaded:
        action_sequence = np.array(init_res['act_seq']) / action_scale
        state_sequence = np.array(init_res['state_seq']) / state_scale
        planned_final_cost = init_res['planned_cost']
    else:
        if enable_sampling:
            mppi_init_act_seq = (
                torch.rand((horizon, action_dim), device=device) * (long_action_ub - long_action_lb) + long_action_lb
            )
            # mppi_init_act_seq = torch.zeros((horizon, action_dim), device=device)
            reward_config["pusher_pos"] = [x_pusher_for_plan, y_pusher_for_plan]
            start_time = time.time()
            
            # if enable_real:
            #     env.stop_realsense()
            #     print(f"stop realsense\n")
            print(f"start hierarchical MPPI")
            init_res = planner_mppi.trajectory_optimization(env_state_to_input(state, device=device), mppi_init_act_seq)
            print(f"{method_type} time: {time.time()-start_time}")
            # if enable_real:
            #     env.start_realsense()
            
            mppi_init_act_seq = init_res["act_seq"]
            # could save the result to warmstart crown
            if method_type == "CROWN" and warm_start_from_sampling:
                # [n_sample, horizon, action_dim] -> [n_sample, horizon*action_dim]
                warmstart_act_seqs = torch.cat(
                    [
                        init_res["act_seq"].view(-1, action_dim * horizon),
                        init_res["act_seqs_lst"][-1].view(-1, action_dim * horizon),
                    ]
                )
                # [n_sample]
                warmstart_cost_seqs = -torch.cat(
                    [
                        init_res["best_eval_output"]["rewards"],
                        init_res["eval_outputs"][-1]["rewards"],
                    ]
                )
                warmstart_cost_seqs, sort_idx = warmstart_cost_seqs.sort(descending=False)
                warmstart_act_seqs = warmstart_act_seqs[sort_idx]
                # [n_sample, 1]
                warmstart_cost_seqs = warmstart_cost_seqs.unsqueeze(1)
                torch.save([warmstart_act_seqs, warmstart_cost_seqs], warmstart_file)
                del warmstart_act_seqs, warmstart_cost_seqs
            # res['act_seq']: [horizon, action_dim]
            action_sequence = init_res["act_seq"].detach().cpu().numpy()
            state_sequence = init_res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
            init_res["state_seq"] = (state_sequence * state_scale).tolist()
            del init_res["model_outputs"], init_res["eval_outputs"], init_res["act_seqs_lst"]
            planned_final_cost = -init_res["best_eval_output"]["rewards"].item()
        reset_seed(seed)
        if method_type == "CROWN":
            known_dim = state_dim * n_his + action_dim * (n_his - 1)
            with open(vnnlib_path, "w") as f:
                print_prop(0, f, action_dim * horizon, long_action_lb, long_action_ub)
            # update model
            known_input = torch.Tensor(state[:known_dim][None])
            pos_pusher_crown = (torch.Tensor([x_pusher_for_plan, y_pusher_for_plan]) / action_scale).unsqueeze(0)
            update_const_nodes(const_nodes_dict, abcrown, cost_model, {"known_input": known_input, "refer_pos": pos_pusher_crown})

            start_time = time.time()
            abcrown.main()
            print(f"CROWN time: {time.time()-start_time}")
            os.system(f"rm -rf {warmstart_file}")
            feasible_sol, best_output, _, intermediate_best_outputs = parse_result(crown_sol_file, abcrown_verbose)[:4]
            assert feasible_sol is not None
            action_sequence = feasible_sol
            action_sequence = np.resize(action_sequence, (horizon, action_dim))
            # [B, n_history, obs_dim], [B, n_history + n_rollout - 1, action_dim], n_history = 1
            state_cur_tensor = torch.from_numpy(state[:state_dim][None][None]).float().to(device)
            action_cur_tensor = torch.from_numpy(action_sequence[None]).float().to(device)
            # state_pred: [B=1, n_rollout, obs_dim]
            state_sequence = model.rollout_model({"state_init": state_cur_tensor,  "action_seq": action_cur_tensor})["state_pred"]
            state_sequence = state_sequence.squeeze(0).detach().cpu().numpy()
            init_res = {
                "act_seq": action_sequence,
                "state_seq": state_sequence,
                "planned_cost": best_output,
                "intermediate_best_outputs": intermediate_best_outputs,
            }
            if abcrown_verbose:
                space_size_dict = analyze_search_progress(bab_file, vis_file_name_prefix)
                init_res["space_size_dict"] = space_size_dict
            planned_final_cost = best_output.item()
    # generate subgoals
    action_sequence_sub = action_sequence.copy() # highlevel planning action sequence
    pusher_pos_sequnce_sub = np.zeros_like(action_sequence) # highlevel planning pusher sequence in dynamics model space
    state_sequence_sub = state_sequence.copy()

    # update original data to absolute coordinate
    x_pusher_planned, y_pusher_planned = x_pusher_for_plan / action_scale, y_pusher_for_plan / action_scale
    for j in range(horizon):
        x_pusher_planned += action_sequence[j, 0]
        y_pusher_planned += action_sequence[j, 1]
        if not preloaded:
            state_sequence_sub[j, ::2] += x_pusher_planned
            state_sequence_sub[j, 1::2] += y_pusher_planned
        pusher_pos_sequnce_sub[j] = [x_pusher_planned, y_pusher_planned]

    # state_sequence_sub = align_observation(state_sequence_sub, keypoint_offsets_2d/state_scale)
    init_res["state_seq"] = (state_sequence_sub * state_scale).tolist() # in pixel space which is mm in this case
    # init_res["act_seq"] = (action_sequence_sub * action_scale).tolist()
    # init_res["curr_state"] = env_state.tolist()
    format_all_res(method_type, action_scale, [init_res])
    long_result = np.concatenate([state_sequence_sub, pusher_pos_sequnce_sub, action_sequence_sub], axis=-1) * state_scale
    long_result = np.concatenate([env_state[None], long_result], axis=0)
    init_res["result_summary"] = {"gt_states": long_result.tolist()}
    long_pred_safe = True
    if penalty_type != 0:
        long_pred_safe = check_state_obs(long_result, state_dim, obs_pos_list, obs_size_list, obs_type, obs_enlarge, scale)
    init_res["exp_setting"] = {"obs_pos_list": obs_pos_list, "obs_size_list": obs_size_list, "obs_type": obs_type, "obs_enlarge": obs_enlarge, "scale": scale}
    # init_res["target_pose"] = target_pose.tolist()
    init_res["start_step"] = 0
    print(f"Long-horizon planned cost: {planned_final_cost}.")

    # reset mppi config  
    max_horizon = hier_config["horizon"]  
    action_bound = hier_config["action_bound"]
    max_horizon = max(max_horizon, math.ceil(subgoal_interval * long_action_bound / action_bound))
    only_final_cost = hier_config["only_final_cost"]
    # n_sim_step = max(n_sim_step, (num_subgoal) * max_horizon))
    buffer_round = hier_config.get("buffer_round", 0)
    n_sim_step = int((num_subgoal + buffer_round) * max_horizon)
    # n_sim_step = max(n_sim_step, 50)
    if enable_real:
        time_limit = real_exp_config["time_limit"]
    else:
        time_limit = n_sim_step
    print(f"time_limit: {time_limit}, max_horizon: {max_horizon}.")
    action_lb = torch.tensor([-action_bound] * action_dim, device=device)
    action_ub = torch.tensor([action_bound] * action_dim, device=device)    
    planner_mppi.action_lower_lim = action_lb
    planner_mppi.action_upper_lim = action_ub
    planner_mppi.n_look_ahead = horizon
    planner_mppi.planner_type = "MPPI"
    # brute_force only applies for long-horizon planning not short-horizon planning
    if method_type == "MPPI_BF":
        planner_mppi.brute_force = False
    planner_mppi.n_sample = hier_config["n_sample"]
    planner_mppi.n_update_iter = hier_config["n_update_iter"]
    planner_mppi.reward_weight = hier_config["reward_weight"]
    planner_mppi.noise_level = hier_config["noise_level"]
    planner_mppi.noise_type = 'normal'
    reward_config["only_final_cost"] = only_final_cost
    if not enable_real:
        reward_config["obs_enlarge"] = 0 

    # select the subgoals
    pusher_pos_sequnce_full = pusher_pos_sequnce_sub.copy() # full horizon
    pusher_pos_sequnce_sub = pusher_pos_sequnce_sub[(subgoal_interval-1)::subgoal_interval] # only select subgoals which is a downsample of full horizon
    
    action_sequence_sub = action_sequence_sub.reshape(-1, subgoal_interval, action_dim).sum(axis=1, keepdims=False)
    state_sequence_sub = state_sequence_sub[(subgoal_interval-1)::subgoal_interval]
    state_sequence_sub = np.concatenate([(env_state[:state_dim] / state_scale)[None], state_sequence_sub], axis=0)

    diff_sequence_sub = np.linalg.norm(state_sequence_sub[1:] - state_sequence_sub[:-1], axis=-1)
    state_sequence_sub = state_sequence_sub[1:]
    # state_sequence_sub = np.concatenate([state_sequence_sub, pusher_pos_sequnce_sub], axis=-1)
    # we need to update target_state in reward_config
    # since now planner_mppi is used to reach the subgoals
    # Core simulation loop

    def update_subgoal(subgoal_id):
        subgoal_id += 1
        if subgoal_id < num_subgoal:
            subgoal_state = state_sequence_sub[subgoal_id]
            print(f"Plan to subgoal {subgoal_id}")
        else:
            subgoal_state = scaled_target_state
            print("Reach the final subgoal, plan to target state.")
        # step 2: determine mode of operation
        # if the distance between the two subgoals is small, the pusher doesn't touch the obj
        # so we just need to follow the open-loop pusher trajectory
        if subgoal_id < num_subgoal and diff_sequence_sub[subgoal_id] < 0.1:
            follow_pusher = True
            print(f"Follow pusher to {pusher_pos_sequnce_sub[subgoal_id]}.")
        else:
            follow_pusher = False
            print(f"Plan to subgoal state {subgoal_state}.")
        inner_idx = 0
        reward_config["target_state"] = torch.tensor(subgoal_state, dtype=torch.float32, device=device)
        return subgoal_id, subgoal_state, follow_pusher, inner_idx

    subgoal_cost_list = []
    i = 0
    start_time = time.time()
    exec_time = 0
    stop_time = 0.2
    frame_time = time.time()
    subgoal_id, subgoal_state, follow_pusher, inner_idx = update_subgoal(-1)
    reach_threshold = hier_config["reach_threshold"]
    subgoal_step = 0
    max_n_act_step = 1
    print(f"start real planning")
    
    # in sim, exec_time = sim step, in real, exec_time = real time
    debug_figs = []
    while exec_time < time_limit:
        # do planning only when there is no action to actuate the pusher
        # in hierarchical planning, that means we should plan to next subgoal
        # or if we are far from the current subgoal, we should replan

        # step 1: get the current state
        if enable_real:
            while time.time() - frame_time < stop_time:
                time.sleep(0.01)
            frame_time = time.time()
            print(f"\n----get state from camera----")
            # while True:
            env_state = env.get_state(
                keypoint_offsets_2d,
                image_path=f"{real_image_prefix}{image_index}.jpeg",
                visualize=enable_visualize,
            )
            
            
            print(f"after processing image:{time.time()}")
            print(f"----end getting state from camera----")
            image_index += 1
        else:
            env_state = env.get_env_state(False)[0]
        state = get_state(env_state, state_dim, state_scale, ) # visualize=True
        gt_states.append(env_state.tolist())
        x_pusher_for_plan, y_pusher_for_plan = env_state[state_dim : state_dim + 2] # in mm / pixel
        curr_pusher_pos = env_state[state_dim : state_dim + 2] / state_scale # turn to dynamic model space
        # update subgoal
        if i > 0:
            # record last step cost
            planned_step_cost = step_cost(state_planner[:state_dim], scaled_target_state, cost_norm)
            actual_step_cost = step_cost(env_state[:state_dim] / state_scale, scaled_target_state, cost_norm)
            cost_meter.update(actual_step_cost, planned_step_cost, i-1)
            # determine the subgoal to reach
            subgoal_cost = step_cost(env_state[:state_dim] / state_scale, subgoal_state, cost_norm)
            print(f"Subgoal cost: {subgoal_cost}.")
            # subgoal_target_cost = step_cost(subgoal_state, scaled_target_state, cost_norm)
            # next_subgoal_state = scaled_target_state
            # if subgoal_id + 1 < num_subgoal:
            #     next_subgoal_state = state_sequence_sub[subgoal_id + 1]
            
            # reach the subgoal
            if ((not follow_pusher)
                and (subgoal_cost < reach_threshold or subgoal_step >= max_horizon)) \
                or (follow_pusher and (reach_pusher_pos(curr_pusher_pos, pusher_pos_sequnce_sub[subgoal_id]) or inner_idx == len(inner_pusher_pos_seq)-1)):
                if (not follow_pusher) and (subgoal_step >= max_horizon) and subgoal_id < num_subgoal-1: # TODO: need more test to determine num_subgoal-1 or num_subgoal
                    print("Fail to reach the subgoal. Try to next one.")
                    # time_limit += int(max_horizon//2)
                subgoal_id, subgoal_state, follow_pusher, inner_idx = update_subgoal(subgoal_id)
                subgoal_step = 0
                if follow_pusher and step_cost(env_state[:state_dim] / state_scale, subgoal_state, cost_norm) > 0.3:
                    follow_pusher = False
                    print("Do not follow pusher since Far from the subgoal.")

            # reach the target state or the last subgoal
            if actual_step_cost < 0.2 or (buffer_round == 0 and subgoal_id == num_subgoal):
                success = True

        print(f"{'-'*5}step {i}{'-'*5}  Current subgoal: {subgoal_id}, follow_pusher: {follow_pusher}.")
        print(f"    Current pusher pos: {curr_pusher_pos}.")
        # print(f"    Current state: {(env_state[:state_dim]/state_scale).tolist()}.")
        if i > 0:
            # print(f"    Planned state: {state_planner[:state_dim].tolist()}.")
            print(f"    state diff: {step_cost(state_planner[:state_dim], env_state[:state_dim]/state_scale, cost_norm)}")
        # create reference action sequence
        # inner_pusher_pos_seq is the pusher sequence to reach the next subgoal
        inner_pusher_pos_seq = pusher_pos_sequnce_full[subgoal_id * subgoal_interval : (subgoal_id + 1) * subgoal_interval]
        if follow_pusher and reach_pusher_pos(curr_pusher_pos, inner_pusher_pos_seq[inner_idx]):
            inner_idx += 1
            inner_idx = min(inner_idx, len(inner_pusher_pos_seq)-1)

        # this in fact tries to compute the difference of the inner_pusher_pos_seq
        inner_action_seq = (inner_pusher_pos_seq[inner_idx:] - np.concatenate([curr_pusher_pos[None], inner_pusher_pos_seq[inner_idx:-1]], axis=0))
        max_action_seq = np.abs(inner_action_seq).max(axis=1)
        n_step_seq = np.ceil(max_action_seq / action_bound).astype(int) 
        horizon = n_step_seq.sum() # number of steps for local MPPI needed to reach the subgoal
        # inner_action_seq = (inner_action_seq / n_step_seq.reshape(-1,1)).repeat(n_step_seq, axis=0)
        # inner_action_seq = np.concatenate([inner_action_seq, np.zeros([max_horizon - len(inner_action_seq), action_dim])], axis=0)
        # horizon = max_horizon
        # short-horizon MPPI planning
        if success:
            horizon = 1
            mppi_init_act_seq = torch.zeros((horizon, action_dim), device=device)
        elif follow_pusher:
            horizon = 1
            next_action = inner_pusher_pos_seq[inner_idx] - curr_pusher_pos
            if np.abs(next_action).max() > action_bound:
                next_action = next_action / np.abs(next_action).max() * action_bound
            mppi_init_act_seq = torch.Tensor(next_action.reshape(1, -1)).to(device)
        elif subgoal_id < num_subgoal:
            # horizon = min(horizon, 3)
            mppi_init_act_seq = torch.zeros((horizon, action_dim), device=device)
            # mppi_init_act_seq = torch.Tensor(inner_action_seq[inner_step:]).to(device)
            # if not hier_config["refer_pos"]:
            #     mppi_init_act_seq = torch.zeros((max_horizon, action_dim), device=device)
            reward_config["only_final_cost"] = False # True
            planner_mppi.noise_level = hier_config["noise_level"]
            planner_mppi.action_lower_lim = action_lb
            planner_mppi.action_upper_lim = action_ub
        else:
            horizon = max_horizon
            mppi_init_act_seq = torch.zeros((horizon, action_dim), device=device)
            reward_config["only_final_cost"] = True
            planner_mppi.noise_level = long_noise_level
            planner_mppi.action_lower_lim = long_action_lb
            planner_mppi.action_upper_lim = long_action_ub
            # planner_mppi.noise_level = hier_config["noise_level"]
            # planner_mppi.action_lower_lim = action_lb
            # planner_mppi.action_upper_lim = action_ub
        reward_config["pusher_pos"] = [x_pusher_for_plan, y_pusher_for_plan] # get from real world in mm / pixel
        sampling_start_time = time.time()
        reset_seed(seed)
        print(f"++++start MPPI+++++, skip={follow_pusher or success}")
        # if not (follow_pusher or success):
        #     import pdb; pdb.set_trace()

        # if enable_real:
            # env.stop_realsense()
            # print(f"stop realsense\n")
        res = planner_mppi.trajectory_optimization(env_state_to_input(state, device=device), mppi_init_act_seq, skip=follow_pusher or success)
        # if enable_real:
        #     env.start_realsense()
            # print(f"stop realsense\n")
        
        print(f"{method_type} time: {time.time()-sampling_start_time}")

        # mppi_init_act_seq = res["act_seq"]
        # res['act_seq']: [horizon, action_dim]
        action_sequence = res["act_seq"].detach().cpu().numpy()
        state_sequence = res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
        del res["model_outputs"], res["eval_outputs"], res["act_seqs_lst"]
        print(f"Short-horizon: {horizon}, planned cost: {-res['best_eval_output']['rewards'].item()}.")
        action = action_sequence[0].copy()
        state_planner = state_sequence[0].copy()
        # convert to absolute coordinate
        state_planner[::2] += x_pusher_for_plan / action_scale + action[0] # get from real world in mm / pixel
        state_planner[1::2] += y_pusher_for_plan / action_scale + action[1]
        exec_action_seq.append((action_scale * action).tolist())
        pred_state_seq.append((state_scale * state_planner[:state_dim]).tolist())

        # update original data to absolute coordinate
        # in fact this is the pusher pos from real world
        x_pusher_planned, y_pusher_planned = x_pusher_for_plan / action_scale, y_pusher_for_plan / action_scale
        # add the planned action to the pusher pos, so that we can get the planned seq
        for j in range(horizon):
            x_pusher_planned += action_sequence[j, 0]
            y_pusher_planned += action_sequence[j, 1]
            state_sequence[j, ::2] += x_pusher_planned
            state_sequence[j, 1::2] += y_pusher_planned
        res["state_seq"] = (state_sequence * state_scale).tolist()
        res["start_step"] = i
        all_res.append(res)

        # actuate the pusher
        dx, dy = action_scale * action # only use the first action

        if enable_real:
            min_action_mm = 7
            if abs(dx) < min_action_mm and dx != 0:
                dx = min_action_mm * np.sign(dx)
            if abs(dy) < min_action_mm and dy != 0:
                dy = min_action_mm * np.sign(dy)
        print(f"Planned action in mm: {(action_scale * action[0], action_scale * action[1])} Planned action in mm after scale: {(dx,dy)}")
        # this is the pusher pos in real world
        next_pusher_pos = (np.clip(x_pusher_for_plan + dx, pusher_pos_lb, pusher_pos_ub),
                            np.clip(y_pusher_for_plan + dy, pusher_pos_lb, pusher_pos_ub))
        if enable_real:
            env.update(*next_pusher_pos)
        else:
            env.update(next_pusher_pos, rel=False)
        
        # real_pusher_pos = env.get_pusher_position()
        print(f"\ncurrent pusher pos: {x_pusher_for_plan, y_pusher_for_plan}")
        print(f"expected pusher pos: {next_pusher_pos}; real pusher pos: {env.get_pusher_position()[:2]*1000}")

        enable_debug = False
        if enable_real and enable_debug:
            # for debug only, plot the trajectory by step
            fig, axes = plt.subplots(1, 3, figsize=(5*2, 5))
            axes = axes.reshape(-1, 3)
            # for ax in axes:
            axes[0,0].set_xlim([0, 500])
            axes[0,0].set_ylim([0, 500])
            # for ax in axes:
            axes[0,1].set_xlim([0, 500])
            axes[0,1].set_ylim([0, 500])

            axes[0,2].set_xlim([0, 500])
            axes[0,2].set_ylim([0, 500])
            axes[0,0].set_aspect('equal')
            axes[0,1].set_aspect('equal')
            axes[0,2].set_aspect('equal')
            
            # plot target state
            draw_object_and_pusher(ax=axes[0,0], object_type="T", \
                                keypoints=state_scale*scaled_target_state.reshape(4,2), \
                                pusher=env_state[state_dim:state_dim+2], \
                                color='r', \
                                alpha=1.0
                                )
            draw_object_and_pusher(ax=axes[0,1], object_type="T", \
                                keypoints=state_scale*scaled_target_state.reshape(4,2), \
                                pusher=env_state[state_dim:state_dim+2], \
                                color='r', \
                                alpha=1.0
                                )
            draw_object_and_pusher(ax=axes[0,2], object_type="T", \
                                keypoints=state_scale*scaled_target_state.reshape(4,2), \
                                pusher=env_state[state_dim:state_dim+2], \
                                color='r', \
                                alpha=1.0
                                )
            # plot the current state and planned trajectory
            draw_object_and_pusher(ax=axes[0,0], object_type="T", \
                                keypoints=env_state[:state_dim].reshape(4,2), \
                                pusher=env_state[state_dim:state_dim+2], \
                                color='g', \
                                alpha=1.0,
                                )
            
            # plot subgoal sequence
            for i, subgoal_state_for_debug in enumerate(state_sequence_sub):
                if i == subgoal_id:
                    alpha = 1.0
                    c_for_subgoal = 'pink'
                else:
                    alpha = 0.8 - 0.6 / len(state_sequence_sub) * i
                    c_for_subgoal = 'b'
                draw_object_and_pusher(ax=axes[0,2], object_type="T", \
                                keypoints=state_scale*subgoal_state_for_debug.reshape(4,2), \
                                pusher=env_state[state_dim:state_dim+2], \
                                color=c_for_subgoal, \
                                alpha=alpha,
                                )

            new_env_state_for_debug = env.get_state(
                    keypoint_offsets_2d,
                    image_path=f"{real_image_prefix}{image_index}.jpeg",
                    visualize=False # enable_visualize,
                )
            # plot after pushing state
            draw_object_and_pusher(ax=axes[0,1], object_type="T", \
                                keypoints=new_env_state_for_debug[:state_dim].reshape(4,2), \
                                pusher=new_env_state_for_debug[state_dim:state_dim+2], \
                                color='b', \
                                alpha=1.0,
                                pusher_color='orange'
                                )
            draw_object_and_pusher(ax=axes[0,0], object_type="T", \
                                keypoints=new_env_state_for_debug[:state_dim].reshape(4,2), \
                                pusher=new_env_state_for_debug[state_dim:state_dim+2], \
                                color='b', \
                                alpha=0.5, \
                                pusher_color='orange'
                                )
            pusher_pos_for_plt = [x_pusher_for_plan, y_pusher_for_plan]
            for j in range(horizon):
                print(f"horizon: {horizon}, j: {j}")
                alpha = 0.8 - 0.6 / horizon * j
                pusher_pos_for_plt[0] += action_sequence[j, 0]*action_scale
                pusher_pos_for_plt[1] += action_sequence[j, 1]*action_scale
                draw_object_and_pusher(ax=axes[0,0], object_type="T", \
                                keypoints=state_scale*state_sequence[j].reshape(4,2), \
                                pusher=[pusher_pos_for_plt[0], pusher_pos_for_plt[1]], \
                                color='g', \
                                alpha=alpha, \
                                pusher_color='k'
                                )
            # visualize subgoal
            # import pdb; pdb.set_trace()
            # print(f"subgoal_state:{subgoal_state}")
            fig.savefig('debug.png', dpi=300, bbox_inches='tight')
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            debug_figs.append(data)
            # plt.show()
            # import pdb; pdb.set_trace()

        i += 1
        subgoal_step += 1
        if enable_real:
            exec_time = time.time() - start_time
        else:
            exec_time = i
        # if success:
        #     break
    print(f"subgoal cost list: {subgoal_cost_list}.")
    if len(debug_figs) > 0:
        # imageio.mimsave('debug.gif', debug_figs)
        # Assuming debug_figs is a list of images (numpy arrays)
        size = debug_figs[0].shape[:2]
        size = (size[1], size[0])  # Swapping (height, width) to (width, height)
        fps = 3
        out = cv2.VideoWriter('debug.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

        for fig in debug_figs:
            out.write(fig[:,:,::-1])

        out.release()


    # get the final state
    if enable_real:
        env_state = env.get_state(
            keypoint_offsets_2d,
            image_path=f"{real_image_prefix}{image_index}.jpeg",
            visualize=enable_visualize,
        )
        image_index += 1
    else:
        env_state = env.get_env_state(False)[0]
    gt_states.append(env_state.tolist())
    planned_step_cost = step_cost(state_planner[:state_dim], scaled_target_state, cost_norm)
    actual_step_cost = step_cost(env_state[:state_dim] / state_scale, scaled_target_state, cost_norm)
    cost_meter.update(actual_step_cost, planned_step_cost, i-1)
    actual_final_cost = actual_step_cost

    cost_meter.print_metric()
    if not enable_real and save_result:
        assert vis_file_name_prefix is not None
        env.save_gif(vis_file_name_prefix + f"sim_cost={round(actual_step_cost, 3)}.gif")
        # env.save_mp4(vis_file_name_prefix + f"_ori_{method_type}_{round(actual_step_cost, 3)}.mp4")
    if enable_real: 
        env.stop()
        del env
        
    elif not enable_real and enable_vis:
        env.close()
    # reformat the result,
    all_res = format_all_res('MPPI', action_scale, all_res)
    if penalty_type == 0:
        gt_safe = True
        pred_safe = True
    else:
        gt_safe = check_state_obs(gt_states, state_dim, obs_pos_list, obs_size_list, obs_type, 0, scale)
        pred_safe = check_state_obs(pred_state_seq, state_dim, obs_pos_list, obs_size_list, obs_type, 0, scale)
    print(f"{'-'*5}long pred safe: {long_pred_safe}, gt safe: {gt_safe}, pred safe: {pred_safe}{'-'*5}")
    exp_setting = {
        "open_loop": open_loop,
        "method_type": method_type,
        "init_pose": init_pose.tolist(),
        "target_pose": target_pose.tolist(),
        "init_pusher_pos": init_pusher_pos.tolist(),
        "n_sim_step": n_sim_step,
        # "horizon": horizon,
        "enable_vis": enable_vis,
        "save_result": save_result,
        "fixed_horizon": fixed_horizon,
        "obs_pos_list": obs_pos_list,
        "obs_size_list": obs_size_list,
        "obs_type": obs_type,
        "obs_enlarge": obs_enlarge,
        "scale": scale,
    }
    try:
        result_summary = {
            "success": success,
            "final_cost": actual_final_cost,
            "gt_safe": gt_safe,
            "pred_safe": pred_safe,
            "gt_states": gt_states,
            "exec_action_seq": exec_action_seq,
            "pred_state_seq": pred_state_seq,
            "runtime": time.time() - start,
            "init_res": init_res,
            "subgoal_cost_list": subgoal_cost_list,
        }
        result_summary.update(cost_meter.get_metric())
        result_summary["planned_cost"] = planned_final_cost
        # if open_loop:
        #     planned_cost = cost_meter.planned_cost_seq[-1] if only_final_cost else cost_meter.planned_acc_cost
        #     actual_cost = cost_meter.actual_cost_seq[-1] if only_final_cost else cost_meter.actual_acc_cost
        #     result_summary["planned_cost"] = planned_cost
        #     result_summary["actual_cost"] = actual_cost
        #     result_summary["cost_diff"] = actual_cost - planned_cost
    except:
        result_summary = {"success": success, "runtime": time.time() - start}
    return {
        "exp_setting": exp_setting,
        "result_summary": result_summary,
        "all_res": all_res,
    }
    
def reach_pusher_pos(curr_pusher_pos, target_pusher_pos):
    return np.linalg.norm(curr_pusher_pos - target_pusher_pos) < 0.02


# rollout function for mppi
# [n_his=1, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample, n_look_ahead, state_dim]
def rollout(state_cur: torch.Tensor, act_seqs: torch.Tensor, model: MLP):
    # [n_sample, n_his, state_dim]
    state_cur_mppi = state_cur.unsqueeze(0).repeat(act_seqs.shape[0], 1, 1)
    res_dict = model.rollout_model({"state_init": state_cur_mppi, "action_seq": act_seqs})
    return {"state_seqs": res_dict["state_pred"]}  # [n_sample, n_look_ahead, state_dim]

# reward function for mppi
# [n_sample, n_look_ahead, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample]
def reward(state_mppi, action_mppi, reward_config):
    device = action_mppi.device
    action_scale, state_dim = reward_config["action_scale"], reward_config["state_dim"]
    enable_latent, cost_norm, only_final_cost = reward_config["enable_latent"], reward_config["cost_norm"], reward_config["only_final_cost"]
    target_state, pusher_pos = reward_config["target_state"], reward_config["pusher_pos"]
    penalty_type, obs_pos_list = reward_config.get("penalty_type", 0), reward_config.get("obs_pos_list", None)
    if penalty_type != 0:
        assert obs_pos_list is not None
        obs_type = reward_config["obs_type"]
        assert obs_type in ["circle", "square"]
        obs_enlarge = reward_config.get("obs_enlarge", 0)
        obs_pos_list = [torch.tensor(obs_pos, device=device) for obs_pos in obs_pos_list]
        if obs_type == "circle":
            obs_size_list = [((1+obs_enlarge)*obs_size)**2 for obs_size in reward_config["obs_size_list"]]
        else:
            obs_size_list = [((1+obs_enlarge)*obs_size) for obs_size in reward_config["obs_size_list"]]
        assert len(obs_pos_list) == len(obs_size_list)
    n_sample, n_look_ahead, action_dim = action_mppi.shape
    state_mppi = state_mppi.clone()
    pos_pusher_mppi = torch.zeros((n_sample, n_look_ahead + 1, action_dim), device=device)
    pos_pusher_mppi[:, 0, 0:1] = pos_pusher_mppi[:, 0, 0:1] + (pusher_pos[0] / action_scale)
    pos_pusher_mppi[:, 0, 1:2] = pos_pusher_mppi[:, 0, 1:2] + (pusher_pos[1] / action_scale)
    for i in range(n_look_ahead):
        pos_pusher_mppi[:, i + 1, :] = pos_pusher_mppi[:, i, :] + action_mppi[:, i, :]
    if not enable_latent:
        state_mppi[:, :, :state_dim:2] = state_mppi[:, :, :state_dim:2] + pos_pusher_mppi[:, 1:, 0:1]
        state_mppi[:, :, 1:state_dim:2] = state_mppi[:, :, 1:state_dim:2] + pos_pusher_mppi[:, 1:, 1:2]
    
    penalty_factor = 100000
    # # add penalty
    penalty_sum = 0
    if penalty_type == 1 or penalty_type == 3:
        penalty = 0
        if obs_type == "circle":
            for obs_pos, obs_size in zip(obs_pos_list, obs_size_list):
                diff_to_obs = pos_pusher_mppi[:, 1:, :2] - obs_pos
                dist_to_obs = (diff_to_obs * diff_to_obs).sum(dim=-1, keepdim=False)
                penalty += F.relu(-dist_to_obs+obs_size)
            penalty_sum += penalty.sum(dim=-1, keepdim=False)
        else:
            for obs_pos, obs_size in zip(obs_pos_list, obs_size_list):
                penalty += F.relu(-abs(pos_pusher_mppi[:, 1:, :2] - obs_pos)+obs_size)
            penalty_sum += penalty.min(dim=-1).values.sum(dim=-1)
    if penalty_type == 2 or penalty_type == 3:
        penalty = 0
        if obs_type == "circle":
            for obs_pos, obs_size in zip(obs_pos_list, obs_size_list):
                diff_to_obs = state_mppi.view(n_sample, n_look_ahead, -1, 2) - obs_pos
                mid_point = (diff_to_obs[:, :, 1:2, :] + diff_to_obs[:, :, 3:4, :]) / 2
                diff_to_obs = torch.cat([diff_to_obs, mid_point], dim=-2)
                dist_to_obs = (diff_to_obs * diff_to_obs).sum(dim=-1, keepdim=False)
                penalty += F.relu(-dist_to_obs+obs_size)
            penalty_sum += penalty.sum(dim=-1).sum(dim=-1, keepdim=False)
        else:
            for obs_pos, obs_size in zip(obs_pos_list, obs_size_list):
                penalty += F.relu(-abs(state_mppi.view(n_sample, n_look_ahead, -1, 2) - obs_pos)+obs_size)
            penalty_sum += penalty.min(dim=-1).values.sum(dim=-1).sum(dim=-1)
    penalty_sum = penalty_factor * penalty_sum
    # [n_sample, n_look_ahead]
    cost_seqs = (
        torch.norm(state_mppi[:, :, :state_dim] - target_state[:state_dim], p=cost_norm, dim=-1)
        ** cost_norm
    ) 
    # init_weight = 1
    # final_weight = n_look_ahead + 1
    # step_weight = torch.arange(1, n_look_ahead + 1, (final_weight - init_weight)/n_look_ahead, device=device).float() / n_look_ahead
    step_weight = torch.arange(1, n_look_ahead + 1, device=device).float() / n_look_ahead
    costs = (cost_seqs*step_weight).sum(dim=-1)+penalty_sum if not only_final_cost else cost_seqs[:, -1] + penalty_sum
    # print("penalty: ", penalty.max().item(), penalty.min().item())
    # print("lowest cost: ", costs.min().item(), "highest cost: ", costs.max().item())
    rewards = -costs
    return {
        "rewards": rewards,
        "cost_seqs": cost_seqs,
    }

def step_cost(state, target_state, cost_norm):
    return (np.linalg.norm(target_state - state, cost_norm)) ** cost_norm
