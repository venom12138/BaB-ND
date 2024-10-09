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

from model.wrapped_model_gnn import wrapped_model_gnn
from planning.sampling_planner import Sampling_Planner
from others.helper import *
from others.meter import CostMeter
from Verifier_Development.complete_verifier.abcrown import ABCROWN

from base_sim import Base_Sim
import tasks.obj_pile.objpile_sim as objpile_sim
from planning.utils import *
from planning.planning_for_pile import cost_function, reward, rollout
from others.helper import plot_obj_pile_single_img
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
    pusher_lo = config["pusher_lo"] / scale
    pusher_hi = config["pusher_hi"] / scale
    long_action_lb = torch.tensor([pusher_lo, pusher_lo, -long_action_bound, -long_action_bound], device=device)
    long_action_ub = torch.tensor([pusher_hi, pusher_hi, long_action_bound, long_action_bound], device=device)
    long_action_norm = long_action_ub - long_action_lb
    long_action_lb /= long_action_norm
    long_action_ub /= long_action_norm
    task_spec_dict = config["task_spec_dict"]
    task_spec_dict["save_img"] = False # save_result
    task_spec_dict["enable_vis"] = False # enable_vis

    env = objpile_sim.ObjectPileSim(task_spec_dict, init_poses=init_pose, pusher_pos=init_pusher_pos)

    # define cost function
    cls_num = env.classes
    labels = env.label_list
    
    cls_idx = []
    for i in range(cls_num):
        cls_idx.append(np.where(np.array(labels) == i)[0])
    

    enable_visualize = False
    if enable_real:
        from real_exp_new.real_wrapper_mp_obj_pile import RealWrapper
        from real_exp_new.dino_wrapper import DinoWrapper
        
        serial_number = "311322300308" # ["311322303615", "215122252880", "151422251501", "246322303954"]
        dino = DinoWrapper(obj_num=len(labels), cls_num=cls_num)
        env = RealWrapper(serial_number, dino)
        env.start()
        init_pusher_pos = [50,50,300]
        env.update(*init_pusher_pos)
        print(f"start real env finish")
        # scale: 500 pixel -> 500mm in real world
        # state_to_real_scale = 1
        image_index = 0
        real_image_prefix = vis_file_name_prefix + "_real_step_"
        initial_state = env.get_state(
            visualize=enable_visualize,
        )
        image_index += 1
        env_state = initial_state.flatten()
        initial_state = initial_state.flatten()[:state_dim] / scale
    else:
        env = objpile_sim.ObjectPileSim(task_spec_dict, init_poses=init_pose, pusher_pos=init_pusher_pos)
        # allow the simulator to a resting position
        for i in range(2):
            env.update(init_pusher_pos)
            env.wait(1)
        env_state = env.get_current_state().flatten()
        initial_state = env_state[:state_dim]/scale # if fix_others else None
    # target_state = np.array(target_pose) / scale
    target_state = []
    for obj in labels:
        target_state.append(target_pose[obj])
    target_state = np.array(target_state)/scale
    obj_num = len(labels)
    if fix_others:
        fixed_idx = np.where(np.array(labels) != 0)[0]
        target_state[fixed_idx] = initial_state.reshape(obj_num, state_dim//obj_num)[fixed_idx]
    target_state = target_state.flatten()

    cost_func = partial(cost_function, cls_idx=cls_idx, cost_norm=cost_norm, target_state=target_state, initial_state=initial_state, cost_weight=cost_weight, fix_others=fix_others)

    # Initialize the planner mppi
    reward_config = {"scale": scale, "state_dim": state_dim, "cost_func": cost_func, "obj_size": original_config['data']["obj_size"], 
                            "forbidden_radius": original_config['planning']["forbidden_radius"], \
                            "far_factor": original_config['planning']["far_factor"], "fix_others": fix_others,
                            "only_final_cost": only_final_cost, "target_state":target_state, "initial_state": initial_state, \
                            "curr_state": env_state_to_input(initial_state[:state_dim], device=device), "action_norm": long_action_norm}
    reward_func = partial(reward, param_dict=reward_config, forbidden=original_config["planning"]["forbidden"])

    config["planner_type"] = method_type
    if method_type == "CROWN" and warm_start_from_sampling:
        config["planner_type"] = original_config["planning"].get("warm_start_method", "MPPI")
    config["model_rollout_fn"] = partial(rollout, model=model, state_dim=state_dim, action_norm=long_action_norm)
    config["evaluate_traj_fn"] = reward_func
    config["action_lower_lim"] = long_action_lb
    config["action_upper_lim"] = long_action_ub
    config["n_look_ahead"] = horizon
    planner_mppi = Sampling_Planner(config)
    if method_type == "CROWN":
        abcrown: ABCROWN = config["abcrown"]
        cost_model:wrapped_model_gnn = config["cost_model"]
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
        # [1, N*N, N], [1, N*N, N]
        Rr, Rs = model.get_Rr_Rs(state_dim//2)
        # target_pose: n_cls, 2
        update_const_nodes(const_nodes_dict, abcrown, cost_model, 
                            {"target_state": torch.tensor(target_state, dtype=torch.float32).view(1, state_dim//2, 2), "Rr": Rr, "Rs": Rs})
        if fix_others:
            update_const_nodes(const_nodes_dict, abcrown, cost_model, 
                                {"initial_state": torch.tensor(initial_state.reshape(1, state_dim//2, 2), dtype=torch.float32)})
        cost_model.cls_idx = cls_idx
        abcrown.model.model_ori.cls_idx = cls_idx

    all_res = []
    cost_meter = CostMeter()
    exec_action_seq, pred_state_seq = [], []
    success = False
    # for real, records state before planning
    gt_states = []

    # in hierarchical planning, we first launch the long-horizon planner MPPI-like or CROWN to get the subgoals (open loop)
    # then we launch the short-horizon MPPI planner to reach the subgoals (closed loop)
    reset_seed(seed)
    state = env_state[:state_dim] / scale

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
        # print(f"start hierarchical MPPI")
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
        known_input = env_state_to_input(state[:state_dim], device).view(1, state_dim//2, 2)
        update_const_nodes(const_nodes_dict, abcrown, cost_model, {"known_input": known_input})
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
    state_sequence_sub = state_sequence.copy()

    # state_sequence_sub = align_observation(state_sequence_sub, keypoint_offsets_2d/scale)
    init_res["state_seq"] = (state_sequence_sub * scale).tolist() # in pixel space which is mm in this case
    # init_res["act_seq"] = (action_sequence_sub * scale).tolist()
    # init_res["curr_state"] = env_state.tolist()
    format_all_res(method_type, scale, [init_res])
    long_result = np.concatenate([state_sequence_sub, action_sequence_sub], axis=-1) * scale
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
    action_lb = torch.tensor([pusher_lo, pusher_lo, -action_bound, -action_bound], device=device)
    action_ub = torch.tensor([pusher_hi, pusher_hi, action_bound, action_bound], device=device)
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
    noise_level = 0.15 if hier_config["noise_level"]=='auto' else hier_config["noise_level"]
    planner_mppi.model_rollout = partial(rollout, model=model, state_dim=state_dim, action_norm=action_norm)
    # for short horizon planning, we only need to reach the subgoals, do not consider initial_state
    reward_config = {"scale": scale, "state_dim": state_dim, "cost_func": cost_func, "obj_size": original_config['data']["obj_size"], 
                            "forbidden_radius": original_config['planning']["forbidden_radius"], 
                            "far_factor": original_config['planning']["far_factor"], "fix_others": False,
                            "only_final_cost": only_final_cost, "target_state":target_state, \
                            "curr_state":env_state_to_input(initial_state[:state_dim], device=device), "action_norm": long_action_norm}
    reward_func = partial(reward, param_dict=reward_config, forbidden=original_config["planning"]["forbidden"])
    planner_mppi.evaluate_traj = reward_func
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
        skip = True
        while skip:
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

        reward_config["target_state"] = subgoal_state
        return subgoal_id, subgoal_state, skip

    subgoal_cost_list = []
    i = 0
    start_time = time.time()
    exec_time = 0
    stop_time = 0.2
    frame_time = time.time()
    subgoal_id, subgoal_state, skip = update_subgoal(-1)
    reach_threshold = hier_config["reach_threshold"]
    success_threshold = hier_config["success_threshold"]
    subgoal_step = 0
    max_n_act_step = 1
    print(f"start real planning")
    





    # plot long-horizon planning result
    # plot long horizon
    long_state_gt = np.array(init_res["result_summary"]['gt_states']) # seq, state_dim (not scaled)
    action_seq = long_state_gt[1:, -2:] # seq-1, action_dim (not scaled)
    pusher_poses = np.concatenate([long_state_gt[1:, -4:-2], [long_state_gt[-1, -4:-2] + action_seq[-1]]])
    long_state_gt = long_state_gt[:, :original_config["data"]["state_dim"]]
    plot_obj_pile(np.expand_dims(long_state_gt, axis=0), np.expand_dims(long_state_gt, axis=0), \
            np.expand_dims(pusher_poses, axis=0), np.expand_dims(action_seq, axis=0), \
            original_config, int(i), save_path='/'.join(original_config.planning.model_path.split('/')[:-1]), \
            labels=np.array(labels), filename=f"1_{i}_cmp_{method_type}_{planned_final_cost:.3f}_long")






    # in sim, exec_time = sim step, in real, exec_time = real time
    while exec_time <= time_limit:
        # do planning only when there is no action to actuate the pusher
        # in hierarchical planning, that means we should plan to next subgoal
        # or if we are far from the current subgoal, we should replan

        # step 1: get the current state
        if enable_real:
            while time.time() - frame_time < stop_time:
                time.sleep(0.01)
            frame_time = time.time()
            print(f"\n----get state from camera----")
            
            env_state = env.get_state(
                visualize=enable_visualize,
            ).flatten()
            print(f"after processing image:{time.time()}")
            print(f"----end getting state from camera----")
            image_index += 1
        else:
            env_state = env.get_current_state().flatten()

        if i > 0:
            gt_states[-1][-4:-2] = [x_pusher_start, y_pusher_start]
            gt_states[-1][-2:] = env_state[-2:]
            # planned cost to target state
            planned_step_cost = cost_func((state_planner[:state_dim])[np.newaxis,np.newaxis,:,])[-1, -1]
            # actual cost to target state
            actual_step_cost = cost_func((env_state[:state_dim]/scale)[np.newaxis,np.newaxis,:,])[-1, -1]
            cost_meter.update(actual_step_cost, planned_step_cost, i-1)
            if subgoal_id < num_subgoal:
                # cost difference between the subgoal and the actual state
                subgoal_cost = cost_func((env_state[:state_dim] / scale)[np.newaxis,np.newaxis,:,], target_state=subgoal_state, fix_others=False)[-1, -1]
            else:
                # cost difference between the target state and the actual state
                subgoal_cost = cost_func((env_state[:state_dim] / scale)[np.newaxis,np.newaxis,:,], target_state=subgoal_state, fix_others=fix_others)[-1, -1]
            print(f"predicted state: {state_planner[:state_dim].reshape(-1, 2)}")
            print(f"actual state: {(env_state[:state_dim]/scale).reshape(-1, 2)}")
            print(f"subgoal state: {subgoal_state.reshape(-1, 2)}")
            print(f"---- Subgoal cost: {subgoal_cost}.")

            # reach the subgoal
            if skip or ((not skip)
                and (subgoal_cost < reach_threshold or subgoal_step >= max_horizon)):
                if (not skip) and (subgoal_step >= max_horizon) and subgoal_id < num_subgoal-1 and subgoal_cost >= reach_threshold: # TODO: need more test to determine num_subgoal-1 or num_subgoal
                    print("Fail to reach the subgoal. Try to next one.")
                    # time_limit += int(max_horizon//2)
                subgoal_id, subgoal_state, skip = update_subgoal(subgoal_id)
                subgoal_step = 0
            if subgoal_id == num_subgoal:
                buffer_round -= 1
                buffer_round = max(buffer_round, 0)

            # reach the target state or the last subgoal
            if actual_step_cost < success_threshold or (buffer_round == 0 and subgoal_id == num_subgoal):
                success = True
        gt_states.append(env_state.tolist())
        state = env_state[:state_dim] / scale
        if exec_time >= time_limit or success:
            # final step
            actual_final_cost = actual_step_cost
            break
        print(f"{'-'*5}step {i}{'-'*5}  Current subgoal: {subgoal_id}, skip: {skip}.")
        # print(f"    Current state: {(env_state[:state_dim]/scale).tolist()}.")
        if i > 0:
            # print(f"    Planned state: {state_planner[:state_dim].tolist()}.")
            # cost difference between the planned state and the actual state
            print(f"state diff: {cost_func((state_planner[:state_dim])[np.newaxis,np.newaxis,:,], target_state=env_state[:state_dim]/scale, fix_others=False)[-1, -1]}")

        # short-horizon MPPI planning
        if success or skip:
            horizon = 1
            mppi_init_act_seq = torch.rand((horizon, action_dim), device=device) * (action_ub - action_lb) + action_lb
            planner_mppi.noise_level = noise_level
            if success:
                reward_config = {"scale": scale, "state_dim": state_dim, "cost_func": cost_func, "obj_size": original_config['data']["obj_size"], 
                            "forbidden_radius": original_config['planning']["forbidden_radius"], 
                            "far_factor": original_config['planning']["far_factor"], fix_others: False,
                            "only_final_cost": only_final_cost, "target_state":target_state, \
                            "curr_state":env_state_to_input(env_state[:state_dim]/scale, device=device), "action_norm": long_action_norm}

                reward_func = partial(reward, param_dict=reward_config, forbidden=original_config["planning"]["forbidden"])
                planner_mppi.evaluate_traj = reward_func
        elif (not skip) and (subgoal_id < num_subgoal):
            # horizon = min(horizon, 3)
            horizon = 1
            # mppi_init_act_seq = torch.tensor([action_sequence_sub[subgoal_id]], device=device, dtype=torch.float32) / action_norm
            mppi_init_act_seq = torch.rand((horizon, action_dim), device=device) * (action_ub - action_lb) + action_lb
            planner_mppi.noise_level = noise_level
            planner_mppi.action_lower_lim = action_lb
            planner_mppi.action_upper_lim = action_ub
            # reward_config = {"scale": scale, "state_dim": state_dim, "cost_func": cost_func, "obj_size": original_config['data']["obj_size"], 
            #                 "forbidden_radius": original_config['planning']["forbidden_radius"], 
            #                 "far_factor": original_config['planning']["far_factor"], fix_others: False,
            #                 "only_final_cost": only_final_cost, "target_state":target_state, \
            #                 "curr_state":env_state_to_input(env_state[:state_dim]/scale, device=device), "action_norm": long_action_norm}

            reward_func = partial(reward, param_dict=reward_config, forbidden=original_config["planning"]["forbidden"])
            planner_mppi.evaluate_traj = reward_func
        elif (not skip):
            horizon = max_horizon
            mppi_init_act_seq = torch.rand((horizon, action_dim), device=device) * (action_ub - action_lb) + action_lb
            planner_mppi.noise_level = long_noise_level
            planner_mppi.action_lower_lim = long_action_lb
            planner_mppi.action_upper_lim = long_action_ub
            reward_config = {"scale": scale, "state_dim": state_dim, "cost_func": cost_func, "obj_size": original_config['data']["obj_size"], \
                            "forbidden_radius": original_config['planning']["forbidden_radius"], \
                            "far_factor": original_config['planning']["far_factor"], "fix_others": fix_others, \
                            "only_final_cost": only_final_cost, "target_state":target_state, "initial_state": initial_state, \
                            "curr_state":env_state_to_input(env_state[:state_dim]/scale, device=device), "action_norm": long_action_norm}

            reward_func = partial(reward, param_dict=reward_config, forbidden=original_config["planning"]["forbidden"])
            planner_mppi.evaluate_traj = reward_func
            # planner_mppi.noise_level = hier_config["noise_level"]
            # planner_mppi.action_lower_lim = action_lb
            # planner_mppi.action_upper_lim = action_ub

        sampling_start_time = time.time()
        reset_seed(seed)
        print(f"++++start MPPI+++++")

        # if enable_real:
            # env.stop_realsense()
            # print(f"stop realsense\n")
        reward_config["curr_state"] = env_state_to_input(state[:state_dim], device=device)
        res = planner_mppi.trajectory_optimization(env_state_to_input(state[:state_dim], device=device), mppi_init_act_seq, skip=success or skip)
        while -res['best_eval_output']['rewards'].item() > 30:
            planner_mppi.noise_level = np.random.uniform(0.1, 0.5)
            mppi_init_act_seq = torch.rand((horizon, action_dim), device=device) * (action_ub - action_lb) + action_lb
            res = planner_mppi.trajectory_optimization(env_state_to_input(state[:state_dim], device=device), mppi_init_act_seq, skip=success or skip)
            print(f"replan: cost: {res['best_eval_output']['rewards'].item()}")
        # if enable_real:
        #     env.start_realsense()
            # print(f"stop realsense\n")
        
        # print(f"{method_type} time: {time.time()-sampling_start_time}")

        # mppi_init_act_seq = res["act_seq"]
        # res['act_seq']: [horizon, action_dim]
        res["act_seq"] = res["act_seq"] * action_norm
        action_sequence = res["act_seq"].detach().cpu().numpy()
        state_sequence = res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
        del res["model_outputs"], res["eval_outputs"], res["act_seqs_lst"]
        print(f"Short-horizon: {horizon}, planned cost: {-res['best_eval_output']['rewards'].item()}, time: {time.time()-sampling_start_time}.")

        res["state_seq"] = (state_sequence * scale).tolist()
        res["start_step"] = i
        all_res.append(res)

        # calculate the action
        action = action_sequence[0].copy()
        state_planner = state_sequence[0].copy()
        x_pusher_start, y_pusher_start = action[:2] * scale
        dx, dy = action[2:] * scale
        if verbose:
            print(f"{'-'*5}step {i}{'-'*5} state:{scale*state[:state_dim].reshape(-1,2)} Planned action: {(x_pusher_start,y_pusher_start,dx,dy)}")
        exec_action_seq.append((scale * action[2:]).tolist())
        pred_state_seq.append((scale * state_planner[:state_dim]).tolist())
        x_pusher = np.clip(x_pusher_start + dx, lo, hi)
        y_pusher = np.clip(y_pusher_start + dy, lo, hi)
        if enable_real:
            min_action_mm = 7
            if abs(dx) < min_action_mm and dx != 0:
                dx = min_action_mm * np.sign(dx)
            if abs(dy) < min_action_mm and dy != 0:
                dy = min_action_mm * np.sign(dy)
            # import pdb; pdb.set_trace()
            env.update(x_pusher_start, y_pusher_start, ud=[dy, -dx, 0])
            env.update(x_pusher, y_pusher, ud=[dy, -dx, 0])
        else:
            env.set_pusher_position(x_pusher_start, y_pusher_start)
            env.update((x_pusher_start + (x_pusher-x_pusher_start)/1e4, y_pusher_start + (y_pusher-y_pusher_start)/1e4), steps=1)
            env.update((x_pusher, y_pusher))
            if enable_vis:
                time.sleep(0.1)
        
        if target_pose is not None:
            print(f"state_planner: {state_planner[:state_dim]} target_state:{target_state}")       







        # for debug 
        plot_state = np.stack([env_state[:state_dim]/scale, state_planner[:state_dim]], axis=0)
        plot_obj_pile_single_img(state=plot_state*scale, \
                        pusher_poses=np.expand_dims(action[:2], axis=0)*scale, \
                        action_seq=np.expand_dims(action[2:], axis=0)*scale, \
                        config=original_config, 
                        save_path='./', 
                        labels=labels, \
                        filename="debug")
            # import pdb; pdb.set_trace()






        i += 1
        subgoal_step += 1
        if enable_real:
            exec_time = time.time() - start_time
        else:
            exec_time = i
        if success:
            break

    print(f"subgoal action sequence: {action_sequence_sub}")
    # print(f"subgoal state sequence: {state_sequence_sub}")
    print(f"executed action sequence: {exec_action_seq}")
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
            "init_res": init_res,
            "labels": labels,
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
    