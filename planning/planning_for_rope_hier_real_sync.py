import os
import sys
try:
    import pyflex
except ImportError:
    pass
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
import torch
import torch.nn.functional as F
import yaml
from functools import partial

from model.wrapped_model_gnn import wrapped_model_gnn
from planning.sampling_planner import Sampling_Planner
from others.helper import *
from others.meter import CostMeter
from Verifier_Development.complete_verifier.abcrown import ABCROWN

import tasks.rope_3d.rope3d_sim as rope3d_sim
from planning.utils import *
from planning.planning_for_rope import cost_function, reward, rollout

# for debug only
from others.plotter import update_plot, draw_object_and_pusher

def planning(
    config,
    original_config, 
    open_loop,
    method_type,
    model,
    init_pose,
    target_pose,
    init_pusher_pos,
    vis_file_name_prefix=None,
    verbose=True
):
    assert method_type in ["MPPI_GD", "MPPI", "GD", "CROWN", "MPPI_BF", "GD_BF", "MIP", "CEM", "DecentCEM"]
    start = time.time()
    task_name = config["task_name"]
    assert task_name in ["rope_3d"]
    real_exp_config = config["real_exp_config"]
    enable_real = real_exp_config["enable"]
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
    assert subgoal_interval == 1
    num_subgoal = hier_config["num_subgoal"]
    horizon = subgoal_interval * num_subgoal
    # horizon = num_subgoal
    device, seed = config["device"], config["seed"]
    state_dim, action_dim, long_action_bound = config["state_dim"], config["action_dim"], config["action_bound"]
    long_noise_level = config["noise_level"]
    fix_others = config["fix_others"]
    cost_weight = config["cost_weight"]
    warm_start_from_sampling, warmstart_file = config["warm_start_from_sampling"], config["crown_name_dict"]["warmstart_file"]
    lo, hi = config["pusher_lo"], config["pusher_hi"]
    long_action_bound, scale = config["action_bound"], config["scale"]
    long_action_lb = torch.tensor([-long_action_bound, -long_action_bound, -long_action_bound], device=device)
    long_action_ub = torch.tensor([long_action_bound, long_action_bound, long_action_bound], device=device)
    long_action_norm = long_action_ub - long_action_lb
    long_action_lb /= long_action_norm
    long_action_ub /= long_action_norm
    task_spec_dict = config["task_spec_dict"]
    task_spec_dict["save_img"] = save_result
    task_spec_dict["enable_vis"] = enable_vis

    enable_visualize = False
    if enable_real:
        raise NotImplementedError
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
        # visible_areas = get_visible_areas(serial_numbers)
        visible_areas = None # TODO: get visible areas
        env = RealWrapper(param_dict, serial_numbers, \
                        visible_areas, real_exp_config["object_path_list"], \
                        real_exp_config["block_color_list"], target_state)
        env.start()
        print(f"start real env finish")
        # scale: 500 pixel -> 500mm in real world
        # state_to_real_scale = 1
        image_index = 0
        real_image_prefix = vis_file_name_prefix + "_real_step_"
        print(f"real_image_prefix: {real_image_prefix}")
        # use preception to get the initial state
        env.init_pusher_pos(*init_pusher_pos)
        init_env_state = env.get_state(
            keypoint_offsets_2d,
            image_path=f"{real_image_prefix}{image_index}.jpeg",
            visualize=enable_visualize,
        )
        image_index += 1
        init_env_state = np.concatenate([init_env_state, [0] * action_dim])
        env_state = init_env_state
    else:
        env = config["env"]
        # if method_type == "CROWN":
        #     env = config["env"]
        # else:
        #     env = rope3d_sim.Rope3dSim(task_spec_dict, )

    target_pose = env.get_target_pose()
    init_pose = env.get_current_state()[:-2].flatten()
    target_state = np.array(target_pose) / scale
    cost_func = partial(cost_function, target_state = target_state,\
            forbidden_area=np.concatenate((env.left_forbidden_area, env.right_forbidden_area)),
            rope_fixed_end=env.get_rope_fixed_end_coord(),
            rope_length=env.get_rope_length())
    env_state = env.get_current_state().flatten()

    # Initialize the planner mppi
    first_pusher_pos = env.get_pusher_position()
    reward_config = {"scale": scale, "state_dim": state_dim, "cost_func": cost_func, "action_norm": long_action_norm,
                    "only_final_cost": only_final_cost, "pusher_pos": first_pusher_pos, "target_state": target_state}
    reward_func = partial(reward, param_dict=reward_config)
    config["planner_type"] = method_type
    if method_type == "CROWN" and warm_start_from_sampling:
        config["planner_type"] = original_config["planning"].get("warm_start_method", "MPPI")
    config["model_rollout_fn"] = partial(rollout, model=model, state_dim=state_dim, action_norm=long_action_norm, scale=scale)
    config["evaluate_traj_fn"] = reward_func
    config["action_lower_lim"] = long_action_lb
    config["action_upper_lim"] = long_action_ub
    config["n_look_ahead"] = horizon
    planner_mppi = Sampling_Planner(config)
    if method_type == "CROWN":
        abcrown: ABCROWN = config["abcrown"]
        cost_model: wrapped_model_gnn = config["cost_model"]
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
        update_const_nodes(const_nodes_dict, abcrown, cost_model, 
                            {"target_state": torch.tensor(target_state, dtype=torch.float32)})

    all_res = []
    cost_meter = CostMeter()
    exec_action_seq, pred_state_seq = [], []
    success = False
    # for real, records state before planning
    gt_states = []

    # in hierarchical planning, we first launch the long-horizon planner MPPI-like or CROWN to get the subgoals (open loop)
    # then we launch the short-horizon MPPI planner to reach the subgoals (closed loop)
    reset_seed(seed)
    state = env_state.copy()[:state_dim]
    for j in range(action_dim):
        state[j:state_dim:action_dim] -= env_state[state_dim + j]
    state /= scale
    enable_sampling = "MPPI" in method_type or "GD" in method_type or 'CEM' in method_type or (method_type == "CROWN" and warm_start_from_sampling)
    if enable_sampling:
        mppi_init_act_seq = (
            torch.rand((horizon, action_dim), device=device) * (long_action_ub - long_action_lb) + long_action_lb
        )
        # mppi_init_act_seq = torch.zeros((horizon, action_dim), device=device)
        sampling_start_time = time.time()
        
        # if enable_real:
        #     env.stop_realsense()
        #     print(f"stop realsense\n")
        print(f"start hierarchical MPPI")
        init_res = planner_mppi.trajectory_optimization(env_state_to_input(state, device=device), mppi_init_act_seq)
        print(f"{method_type} time: {time.time()-sampling_start_time}")
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
        init_res["act_seq"] = init_res["act_seq"] * long_action_norm
        action_sequence = init_res["act_seq"].detach().cpu().numpy()
        state_sequence = init_res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
        init_res["state_seq"] = (state_sequence * scale).tolist()
        del init_res["model_outputs"], init_res["eval_outputs"], init_res["act_seqs_lst"]
        planned_final_cost = -init_res["best_eval_output"]["rewards"].item()
    reset_seed(seed)
    if method_type == "CROWN":
        known_dim = state_dim * n_his + action_dim * (n_his - 1)
        with open(vnnlib_path, "w") as f:
            print_prop(0, f, action_dim * horizon, long_action_lb, long_action_ub)
        # update model
        known_input = env_state_to_input(state[:state_dim], device).view(1, state_dim//3, 3)
        pos_pusher_crown = env.get_pusher_position() # finger position + 0.5
        pos_pusher_crown = torch.tensor(pos_pusher_crown, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        update_const_nodes(const_nodes_dict, abcrown, cost_model, {"curr_state": known_input, "refer_pos": pos_pusher_crown})
        crown_start_time = time.time()
        abcrown.main()
        print(f"CROWN time: {time.time()-crown_start_time}")
        os.system(f"rm -rf {warmstart_file}")
        feasible_sol, best_output, _, intermediate_best_outputs = parse_result(crown_sol_file, abcrown_verbose)[:4]
        assert feasible_sol is not None
        action_sequence = feasible_sol.reshape(horizon, action_dim) * long_action_norm.cpu().numpy()
        # [B, n_history, obs_dim], [B, n_history + n_rollout - 1, action_dim], n_history = 1
        state_cur_tensor = torch.from_numpy(state[:state_dim][None][None]).float().to(device)
        action_cur_tensor = torch.from_numpy(action_sequence[None]).float().to(device)
        # state_pred: [B=1, n_rollout, obs_dim]
        state_sequence = model.rollout_model({"state_init": state_cur_tensor,  "action_seq": action_cur_tensor})["state_pred"]
        reward_config["pusher_pos"] = env.get_pusher_position()
        # reward can change state to absolute position
        reward(state_sequence, action_cur_tensor, reward_config)
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
    pusher_pos_sequnce_sub = np.zeros_like(action_sequence)
    state_sequence_sub = state_sequence.copy()

    # update original data to absolute coordinate
    x_pusher_planned, y_pusher_planned, z_pusher_planned = first_pusher_pos / scale
    for j in range(horizon):
        x_pusher_planned += action_sequence[j, 0]
        y_pusher_planned += action_sequence[j, 1]
        z_pusher_planned += action_sequence[j, 2]
        state_sequence_sub[j, ::3] += x_pusher_planned
        state_sequence_sub[j, 1::3] += y_pusher_planned
        state_sequence_sub[j, 2::3] += z_pusher_planned
        pusher_pos_sequnce_sub[j] = [x_pusher_planned, y_pusher_planned, z_pusher_planned]

    # state_sequence_sub = align_observation(state_sequence_sub, keypoint_offsets_2d/scale)
    init_res["state_seq"] = (state_sequence_sub * scale).tolist() # in pixel space which is mm in this case
    # init_res["act_seq"] = (action_sequence_sub * scale).tolist()
    # init_res["curr_state"] = env_state.tolist()
    format_all_res(method_type, scale, [init_res])
    long_result = np.concatenate([state_sequence_sub, pusher_pos_sequnce_sub, action_sequence_sub], axis=-1) * scale
    long_result = np.concatenate([env_state[None], long_result], axis=0)
    init_res["result_summary"] = {"gt_states": long_result.tolist()}
    init_res["exp_setting"] = {"scale": scale}
    # init_res["target_pose"] = target_pose.tolist()
    init_res["start_step"] = 0
    print(f"Long-horizon planned cost: {planned_final_cost}.")

    # reset mppi config  
    max_horizon = hier_config["horizon"]  
    action_bound = hier_config["action_bound"]
    max_horizon = max(max_horizon, math.ceil(subgoal_interval * long_action_bound / action_bound)+1)
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
    action_lb = torch.tensor([-action_bound, -action_bound, -action_bound], device=device)
    action_ub = torch.tensor([action_bound, action_bound, action_bound], device=device)
    action_norm = action_ub - action_lb
    action_lb /= action_norm
    action_ub /= action_norm
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
    planner_mppi.noise_type = 'normal'
    noise_level = 0.2 if hier_config.get("noise_level", 'auto') else hier_config["noise_level"]
    planner_mppi.model_rollout = partial(rollout, model=model, state_dim=state_dim, action_norm=action_norm, scale=scale)
    # for short horizon planning, we only need to reach the subgoals, do not consider initial_state
    reward_config["only_final_cost"] = only_final_cost
    reward_config["action_norm"] = action_norm

    # select the subgoals
    action_sequence_sub = action_sequence_sub.reshape(-1, subgoal_interval, action_dim).sum(axis=1, keepdims=False)
    state_sequence_sub = state_sequence_sub[(subgoal_interval-1)::subgoal_interval]
    state_sequence_sub = np.concatenate([(env_state[:state_dim] / scale)[None], state_sequence_sub], axis=0)

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
            subgoal_state = target_state
            print("Reach the final subgoal, plan to target state.")
        # step 2: determine mode of operation
        # if the distance between the two subgoals is small, the pusher doesn't touch the obj
        # so we just need to follow the open-loop pusher trajectory
        skip = False
        if subgoal_id < num_subgoal and diff_sequence_sub[subgoal_id] < 0.1:
            skip = True
            print(f"state do not change much, skip to the next subgoal.")

        reward_config["target_state"] = torch.tensor(subgoal_state, dtype=torch.float32, device=device)
        return subgoal_id, subgoal_state, skip

    i = 0
    start_time = time.time()
    exec_time = 0
    stop_time = 0.2
    frame_time = time.time()
    subgoal_id, subgoal_state, skip = update_subgoal(-1)
    reach_threshold = hier_config["reach_threshold"]
    subgoal_step = 0
    max_n_act_step = 1
    print(f"start real planning")
    
    # in sim, exec_time = sim step, in real, exec_time = real time
    while exec_time <= time_limit:
        # do planning only when there is no action to actuate the pusher
        # in hierarchical planning, that means we should plan to next subgoal
        # or if we are far from the current subgoal, we should replan

        # step 1: get the current state
        if enable_real:
            raise NotImplementedError
            while time.time() - frame_time < stop_time:
                time.sleep(0.01)
            frame_time = time.time()
            print(f"\n----get state from camera----")
            
            env_state = env.get_state(
                keypoint_offsets_2d,
                image_path=f"{real_image_prefix}{image_index}.jpeg",
                visualize=enable_visualize,
            )
            print(f"after processing image:{time.time()}")
            print(f"----end getting state from camera----")
            image_index += 1
        else:
            env_state = env.get_current_state().flatten()
        if i > 0:
            planned_step_cost = cost_func((state_planner[:state_dim])[np.newaxis,np.newaxis,:,])[-1, -1]
            actual_step_cost = cost_func((env_state[:state_dim]/scale)[np.newaxis,np.newaxis,:,])[-1, -1]
            cost_meter.update(actual_step_cost, planned_step_cost, i-1)
            subgoal_cost = cost_func((env_state[:state_dim] / scale)[np.newaxis,np.newaxis,:,], target_state=subgoal_state,
                                        forbidden_area=None, rope_fixed_end=None, rope_length=None)[-1, -1]
            print(f"predicted state: {state_planner[:state_dim].reshape(-1, 3)}")
            print(f"actual state: {(env_state[:state_dim]/scale).reshape(-1, 3)}")
            print(f"subgoal state: {subgoal_state.reshape(-1, 3)}")
            print(f"---- Subgoal cost: {subgoal_cost}.")

            # reach the subgoal
            if skip or ((not skip)
                and (subgoal_cost < reach_threshold or subgoal_step >= max_horizon)):
                if (not skip) and (subgoal_step >= max_horizon) and subgoal_id < num_subgoal-1 and subgoal_cost >= reach_threshold: # TODO: need more test to determine num_subgoal-1 or num_subgoal
                    print("Fail to reach the subgoal. Try to next one.")
                    # time_limit += int(max_horizon//2)
                subgoal_id, subgoal_state, skip = update_subgoal(subgoal_id)
                subgoal_step = 0

            # reach the target state or the last subgoal
            if actual_step_cost < 1 or (buffer_round == 0 and subgoal_id == num_subgoal):
                success = True
        gt_states.append(env_state.tolist())
        state = env_state.copy()[:state_dim]
        for j in range(action_dim):
            state[j:state_dim:action_dim] -= env_state[state_dim + j]
        state /= scale

        if exec_time >= time_limit:
            # final step
            actual_final_cost = actual_step_cost
            break
        print(f"{'-'*5}step {i}{'-'*5}  Current subgoal: {subgoal_id}, skip: {skip}.")
        # print(f"    Current state: {(env_state[:state_dim]/scale).tolist()}.")
        if i > 0:
            # print(f"    Planned state: {state_planner[:state_dim].tolist()}.")
            print(f"    state diff: {cost_func((state_planner[:state_dim])[np.newaxis,np.newaxis,:,], target_state=env_state[:state_dim]/scale)[-1, -1]}")

        # short-horizon MPPI planning
        if success or skip:
            horizon = 1
            mppi_init_act_seq = torch.zeros((horizon, action_dim), device=device)
            if success:
                reward_config["target_state"] = target_state
        elif (not skip) and (subgoal_id < num_subgoal):
            # horizon = min(horizon, 3)
            horizon = 1
            mppi_init_act_seq = torch.zeros((horizon, action_dim), device=device)
            planner_mppi.noise_level = noise_level
            planner_mppi.action_lower_lim = action_lb
            planner_mppi.action_upper_lim = action_ub
        elif (not skip):
            horizon = max_horizon
            mppi_init_act_seq = torch.zeros((horizon, action_dim), device=device)
            planner_mppi.noise_level = long_noise_level
            planner_mppi.action_lower_lim = long_action_lb
            planner_mppi.action_upper_lim = long_action_ub
            # planner_mppi.noise_level = hier_config["noise_level"]
            # planner_mppi.action_lower_lim = action_lb
            # planner_mppi.action_upper_lim = action_ub
        reward_config["pusher_pos"] = env.get_pusher_position()
        sampling_start_time = time.time()
        reset_seed(seed)
        print(f"++++start MPPI+++++")

        # if enable_real:
            # env.stop_realsense()
            # print(f"stop realsense\n")
        res = planner_mppi.trajectory_optimization(env_state_to_input(state[:state_dim], device=device), mppi_init_act_seq, skip=success or skip)
        # if enable_real:
        #     env.start_realsense()
            # print(f"stop realsense\n")
        
        print(f"{method_type} time: {time.time()-sampling_start_time}")

        # mppi_init_act_seq = res["act_seq"]
        # res['act_seq']: [horizon, action_dim]
        res["act_seq"] = res["act_seq"] * action_norm
        action_sequence = res["act_seq"].detach().cpu().numpy()
        state_sequence = res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
        del res["model_outputs"], res["eval_outputs"], res["act_seqs_lst"]
        print(f"Short-horizon: {horizon}, planned cost: {-res['best_eval_output']['rewards'].item()}.")

        res["state_seq"] = (state_sequence * scale).tolist()
        res["start_step"] = i
        all_res.append(res)

        # calculate the action
        action = action_sequence[0].copy()
        state_planner = state_sequence[0].copy()

        pusher_pos = env.get_pusher_position() # finger position + 0.5
        eef_pos = env.get_end_effector_position() # finger position + 1.616-0.5, which is eef position
        eef_pos_prev = eef_pos
        
        dx, dy, dz = action[:action_dim] * scale
        
        state_planner[::3] += pusher_pos[0] / scale + action[0]
        state_planner[1::3] += pusher_pos[1] / scale + action[1]
        state_planner[2::3] += pusher_pos[2] / scale + action[2]

        x_eef_start, y_eef_start, z_eef_start = eef_pos[0], eef_pos[1], eef_pos[2]
        x_eef = np.clip(eef_pos[0] + dx, lo[0], hi[0])
        y_eef = np.clip(eef_pos[1] + dy, lo[1], hi[1])
        z_eef = np.clip(eef_pos[2] + dz, lo[2], hi[2])
        eef_pos = np.array([x_eef, y_eef, z_eef])
        # if any clipping happens, just print out the information
        if (x_eef != eef_pos_prev[0] + dx) or (y_eef != eef_pos_prev[1] + dy) or (z_eef != eef_pos_prev[2] + dz):
            raise ValueError(f"clipping happens at step {i}, eef_pos: {eef_pos}, eef_pos_prev: {eef_pos_prev}, action: {action}")
            print(f"clipping happens at step {i}, eef_pos: {eef_pos}, eef_pos_prev: {eef_pos_prev}, action: {action}")
        
        exec_action_seq.append((scale * np.array([x_eef-x_eef_start, y_eef-y_eef_start, \
                            z_eef-z_eef_start])).tolist())
        pred_state_seq.append((scale * state_planner[:state_dim]).tolist())
        if verbose:
            print(f"{'-'*5}step {i}{'-'*5}  Planned action: {(dx,dy,dz)}")        

        if enable_real:
            raise NotImplementedError
        else:
            env.update([x_eef, y_eef, z_eef])
            if open_loop and enable_vis:
                time.sleep(0.1)
        
        i += 1
        subgoal_step += 1
        if enable_real:
            exec_time = time.time() - start_time
        else:
            exec_time = i

    # print(f"subgoal action sequence: {action_sequence_sub}")
    # print(f"subgoal state sequence: {state_sequence_sub}")
    # print(f"executed action sequence: {exec_action_seq}")
    # print(f"executed state sequence: {pred_state_seq}")
    # print(f"gt_states: {gt_states}")

    cost_meter.print_metric()
    if not enable_real and save_result:
        assert vis_file_name_prefix is not None
        imageio.mimsave(vis_file_name_prefix + f"_ori_{method_type}_{round(actual_final_cost, 3)}.gif", env.frames, fps=5)
    if enable_real: 
        env.stop()
        del env
        
    elif not enable_real and enable_vis:
        env.close()
    # reformat the result,
    all_res = format_all_res('MPPI', scale, all_res)

    exp_setting = {
        "open_loop": open_loop,
        "method_type": method_type,
        "init_pose": init_pose,
        "target_pose": target_pose,
        "init_pusher_pos": init_pusher_pos,
        "n_sim_step": n_sim_step,
        # "horizon": horizon,
        "enable_vis": enable_vis,
        "save_result": save_result,
        "fixed_horizon": fixed_horizon,
        "scale": scale,
    }
    try:
        result_summary = {
            "success": success,
            "final_cost": actual_final_cost,
            "gt_states": gt_states,
            "exec_action_seq": exec_action_seq,
            "pred_state_seq": pred_state_seq,
            "runtime": time.time() - start,
            "init_res": init_res
        }
        result_summary.update(cost_meter.get_metric())
        result_summary["planned_cost"] = planned_final_cost
        result_summary["forbidden_area"] = np.concatenate([env.left_forbidden_area, env.right_forbidden_area]).tolist()

    except:
        result_summary = {"success": success, "runtime": time.time() - start}
    return {
        "exp_setting": exp_setting,
        "result_summary": result_summary,
        "all_res": all_res,
    }
    