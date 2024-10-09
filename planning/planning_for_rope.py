import os
import sys
try:
    import pyflex
    
except ImportError:
    pass
sys.path.append(os.getcwd())
cwd = os.getcwd()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
import copy
import torch
import yaml
from functools import partial
import torch.nn.functional as F
from planning.sampling_planner import Sampling_Planner
from others.helper import *
from others.meter import CostMeter
from Verifier_Development.complete_verifier.abcrown import ABCROWN
from planning.utils import *
import tasks.rope_3d.rope3d_sim as rope3d_sim

penalty_factor = 1e5
# # target_state: 1, 2
# # state: n_sample, length, state_dim
# # forbidden_area: (6, 3) (left_bottom, left_top, prj_dir, right_bottom, right_top, prj_dir)
def cost_function(state, target_state, forbidden_area, rope_fixed_end, rope_length):
    return_np = False
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
        return_np = True
    n_sample, length, state_dim = state.shape # B, L, state_dim
    state = state.reshape(n_sample, length, state_dim//3, 3) # B, L, N, 3
    forbidden_penalty_factor = penalty_factor
    fixed_end_penalty_factor = penalty_factor
    rope_length_norm = 2
    if rope_length_norm == 2:
        rope_length = rope_length ** 2
    cost_norm = 1
    target_state = torch.tensor(target_state, dtype=state.dtype, device=state.device).view(-1, 3)
    if forbidden_area is not None:
        forbidden_area = torch.tensor(forbidden_area, dtype=state.dtype, device=state.device)
    if rope_fixed_end is not None:
        rope_fixed_end = torch.tensor(rope_fixed_end, dtype=state.dtype, device=state.device)
    penalty = torch.zeros((n_sample, length), device=state.device)
    if target_state.shape[0] == 1:
        # only consider the x, z
        # dis = torch.norm(state[..., 0::2]- target_state[..., 0::2], p=2, dim=-1)  # (B, L, N)
        diff = state[..., 0::2]- target_state[..., 0::2]
    else:
        # dis = torch.norm(state - target_state, p=2, dim=-1)
        diff = (target_state - state)
        diff += diff[:, :, :, 0:1] * 0.3
        # diff = state[..., 0::2]- target_state[..., 0::2]
    # dis = torch.norm(diff, p=2, dim=-1)  # (B, L, N)
    if cost_norm == 2:
        dis = (diff * diff).sum(dim=-1)
    else:
        dis = torch.abs(diff).sum(dim=-1)
    # we only consider the 3 points near the grasp point
    # currently, we let the target state's x coordinate to be the same as the fix end's x coordinate
    # dis = dis[:,:,:3] # if 3 points satisfy the condition, then it is successful
    cost = torch.sum(dis, dim=-1) # (B, L)
    
    if rope_fixed_end is not None:
        rope_end_diff = state - rope_fixed_end # (B, L, N, 3)
        if rope_length_norm == 2:
            rope_end_dis = (rope_end_diff * rope_end_diff).sum(dim=-1) # (B, L, N)
        else:
            rope_end_dis = torch.abs(rope_end_diff).sum(dim=-1)
        # rope_end_dis = torch.norm(state - rope_fixed_end, p=1, dim=-1) # (B, L, N)
        rope_end_dis = F.relu(rope_end_dis - rope_length)
        # rope_end_dis = torch.norm(state - rope_fixed_end, p=1, dim=-1) # (B, L, N)
        rope_end_cost = torch.sum(rope_end_dis, dim=-1) # (B, L)
        penalty = rope_end_cost * fixed_end_penalty_factor
        # print(f"max rope_end_cost: {rope_end_cost.max().item()}")
    
    # # compute the forbidden area cost
    if forbidden_area is not None:
        left_bottom, left_top, right_bottom, right_top = forbidden_area #
        left_center = (left_bottom + left_top) / 2
        left_half_edge = torch.abs(left_top - left_bottom) / 2
        right_center = (right_bottom + right_top) / 2
        right_half_edge = torch.abs(right_top - right_bottom) / 2

        constraint_cost1, constraint_cost2, constraint_cost3 = 0, 0, 0
        constraint_cost1 = F.relu(left_half_edge - torch.abs(state - left_center)) # B, L, N, 3
        constraint_cost1 = torch.sum(constraint_cost1.min(dim=-1, keepdim=False).values, dim=-1) # B, L
        constraint_cost2 = F.relu(right_half_edge - torch.abs(state - right_center)) # B, L, N, 3
        constraint_cost2 = torch.sum(constraint_cost2.min(dim=-1, keepdim=False).values, dim=-1)
        constraint_cost3 = F.relu(0.5-state[..., 2]).sum(dim=-1) # B, L
        penalty += (constraint_cost1 + constraint_cost2 + constraint_cost3) * forbidden_penalty_factor
    if return_np:
        cost += penalty
        cost = cost.cpu().numpy()
        return cost # B, L
    return cost, penalty

# [n_his=1, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample, n_look_ahead, state_dim]
def rollout(state_cur: torch.Tensor, act_seqs: torch.Tensor, model, \
        state_dim, action_norm, scale):
    # [n_sample, n_his, state_dim]
    n_sample = act_seqs.shape[0]
    
    state_cur = state_cur.unsqueeze(0).repeat(n_sample, 1, 1)
    assert state_cur.shape[1] == 1 # n_his=1
    n_his = state_cur.shape[1]
    output = []
    n_look_ahead = act_seqs.shape[1]
    # state_cur: [n_sample, n_his, state_dim]
    # act_seqs: [n_sample, n_look_ahead, action_dim]
    # n_his=1 simplified the operation
    act_seqs = act_seqs * action_norm
    action_dim = act_seqs.shape[-1]

    for i in range(n_look_ahead):
        input = {
            "state": state_cur[:, :, :state_dim],
            "action": act_seqs[:, i : i + 1, :],
        }
        # [n_sample, state_dim]
        state_nxt = model(input)
        # [n_sample, 1, state_dim]
        state_nxt = state_nxt.unsqueeze(1)
        
        state_cur = torch.cat([state_cur[:, 1:], state_nxt], 1)
        # import pdb; pdb.set_trace()
        # state_nxt += input["action"][:, None, -1].repeat(1, 1, state_dim//action_dim)
        output.append(state_nxt)

    output = torch.cat(output, 1)
    return {"state_seqs": output}  # [n_sample, n_look_ahead, state_dim]

# [n_sample, n_look_ahead, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample]
def reward(state_mppi, action_mppi, param_dict):
    state_mppi = state_mppi.clone()
    scale, cost_func, only_final_cost = param_dict["scale"], param_dict["cost_func"], param_dict["only_final_cost"]
    n_sample, n_look_ahead, state_dim = state_mppi.shape
    action_dim = action_mppi.shape[-1]
    pusher_pos, target_state, action_norm = param_dict["pusher_pos"], param_dict["target_state"], param_dict["action_norm"]
    if action_norm is not None:
        action_mppi = action_mppi * action_norm

    pos_pusher_mppi = torch.zeros((n_sample, n_look_ahead + 1, action_dim), device=action_mppi.device)
    # pusher_pos is the finger_pos+0.5
    for i in range(action_dim):
        pos_pusher_mppi[:, 0, i:i+1] = pos_pusher_mppi[:, 0, i:i+1] + pusher_pos[i] / scale
    for i in range(n_look_ahead):
        pos_pusher_mppi[:, i + 1, :] = pos_pusher_mppi[:, i, :] + action_mppi[:, i, :]
    for i in range(action_dim):
        state_mppi[:, :, i:state_dim:action_dim] = state_mppi[:, :, i:state_dim:action_dim] + pos_pusher_mppi[:, 1:, i:i+1]

    step_weight = (torch.arange(1, n_look_ahead + 1, device=state_mppi.device).float() / n_look_ahead)
    # if pusher is under table
    position_cost = F.relu(1.1-pos_pusher_mppi[..., -1])[:,1:]
    position_cost = position_cost + F.relu(-2.45-pos_pusher_mppi[..., 1])[:,1:]
    # [n_sample, n_look_ahead]
    # import pdb; pdb.set_trace()
    ret = cost_func(state_mppi, target_state=target_state) # [n_sample, n_look_ahead]
    cost_seqs = ret[0]
    z_penalty_factor = penalty_factor
    penalty_seqs = ret[1] + position_cost * z_penalty_factor
    # height_reward = 0
    # height_reward = - abs(pos_pusher_mppi[:, 8, -1]-1.1)*0 + abs(pos_pusher_mppi[:, 8, 0]-0.56)*0
    # height_reward = -abs(action_mppi[:, :, -1]).sum(dim=-1)
    penalty_sum = penalty_seqs.sum(dim=-1) 
    # penalty_sum += height_reward * 1
    # - abs(state_mppi.view(n_sample, n_look_ahead, state_dim//3,3)[:,:,7, 2] - 0.5).sum(dim=-1)*5
    # - abs(pos_pusher_mppi[:, 8, -1]-1.1) + abs(pos_pusher_mppi[:, 8, 0]-0)
    # cost_seqs -= 6 
    return {
        "rewards": -(cost_seqs*step_weight).sum(dim=-1) - penalty_sum if not only_final_cost else -(cost_seqs[:, -1])-penalty_sum,
        "cost_seqs": cost_seqs+penalty_seqs,
    }


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
    enable_vis = config["enable_vis"]
    # enable_vis = True
    save_result = config["save_result"]
    only_final_cost = config["only_final_cost"]
    fixed_horizon = config["fixed_horizon"]
    use_prev_sol = config["use_prev_sol"]
    n_sim_step = config["n_sim_step"]
    n_his = config["n_his"]
    assert n_his == 1
    n_sim_step = config["n_sim_step"]
    if open_loop:
        max_horizon = config["horizon"]
        max_n_act_step = max_horizon
        n_sim_step = max_horizon
    else:
        max_horizon = min(config["horizon"], n_sim_step)
        max_n_act_step = min(config["n_act_step"], max_horizon)
    if enable_real:
        time_limit = real_exp_config["time_limit"]
        fixed_horizon = True
    else:
        time_limit = n_sim_step
    device, seed = config["device"], config["seed"]
    state_dim, action_dim = config["state_dim"], config["action_dim"]
    scale = config["scale"]
    warm_start_from_sampling, warmstart_file = config["warm_start_from_sampling"], config["crown_name_dict"]["warmstart_file"]
    lo, hi = config["pusher_lo"], config["pusher_hi"] # list
    action_bound, scale = config["action_bound"], config["scale"]
    action_lb = torch.tensor([-action_bound, -action_bound, -action_bound], device=device)
    action_ub = torch.tensor([action_bound, action_bound, action_bound], device=device)
    action_norm = action_ub - action_lb
    action_lb /= action_norm
    action_ub /= action_norm

    task_spec_dict = config["task_spec_dict"]
    task_spec_dict["save_img"] = save_result
    task_spec_dict["enable_vis"] = enable_vis
    
    # import pdb; pdb.set_trace()
    if enable_real:
        from real_exp_new.real_wrapper_mp_rope3d import RealWrapper
        serial_number = ["311322303615", "215122252880", "213622251153",] 
        env = RealWrapper(task_spec_dict, serial_number,)
        env.start()
        env.update_real(*list(env.get_pusher_position_in_real()*1000))
        env.get_state()
        # init_pusher_pos = [50,50,300]
        # env.update_real(*init_pusher_pos)
    else:
        env = config["env"]
        # if method_type == "CROWN":
        #     env = config["env"]
        # else:
        #     env = rope3d_sim.Rope3dSim(task_spec_dict, )
    if enable_real:
        init_pose = env.get_state()[:-2].flatten()
    else:
        init_pose = env.get_current_state()[:-2].flatten()
    target_pose = env.get_target_pose()
    target_state = np.array(target_pose) / scale
    cost_func = partial(cost_function, \
            forbidden_area=np.concatenate((env.left_forbidden_area, env.right_forbidden_area)),
            rope_fixed_end=env.get_rope_fixed_end_coord(),
            rope_length=env.get_rope_length())

    first_pusher_pos = env.get_pusher_position() # unscaled for reorientation: [[x,y],[x,y]]
    reward_config = {"scale": scale, "state_dim": state_dim, "cost_func": cost_func, "action_norm": action_norm,
                "only_final_cost": only_final_cost, "pusher_pos": first_pusher_pos, "target_state": target_state}
    reward_func = partial(reward, param_dict=reward_config)
    # Initialize planner based on method_type
    enable_sampling = "MPPI" in method_type or "GD" in method_type or 'CEM' in method_type or (method_type == "CROWN" and warm_start_from_sampling)
    if enable_sampling:
        config["planner_type"] = method_type
        if method_type == "CROWN" and warm_start_from_sampling:
            config["planner_type"] = original_config["planning"].get("warm_start_method", "MPPI")
        config["model_rollout_fn"] = partial(rollout, model=model, state_dim=state_dim, \
            action_norm=action_norm, scale=scale)
        config["evaluate_traj_fn"] = reward_func
        config["action_lower_lim"] = action_lb
        config["action_upper_lim"] = action_ub
        config["n_look_ahead"] = max_horizon
        planner_mppi = Sampling_Planner(config)
    if method_type == "CROWN":
        abcrown: ABCROWN = config["abcrown"]
        cost_model = config["cost_model"]
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

    # Core simulation loop
    all_res = []
    cost_meter = CostMeter()
    exec_action_seq, pred_state_seq = [], []
    success = True
    gt_states = []
    # number of steps to actuate the pusher, increase after planning, decrease after acting
    n_to_act = 0
    i = 0
    start_time = time.time()
    frame_time = time.time()
    stop_time = 0.2
    exec_time = 0
    planning_time = 0

    while exec_time <= time_limit:
        if enable_real:
            while time.time() - frame_time < stop_time:
                time.sleep(0.01)
            frame_time = time.time()
            # get state from real world
            env_state = env.get_state().flatten()
        else:
            env_state = env.get_current_state().flatten()
        # update the cost meter
        if i > 0:
            # import pdb; pdb.set_trace()
            planned_step_cost = cost_func((state_planner[:state_dim])[np.newaxis,np.newaxis,:,], target_state=target_state)[-1, -1]
            actual_step_cost = cost_func((env_state[:state_dim]/scale)[np.newaxis,np.newaxis,:,], target_state=target_state)[-1, -1]
            cost_meter.update(actual_step_cost, planned_step_cost, i-1)
            # if verbose:
            #     print(f"state_planner: {state_planner[:state_dim].reshape(-1,3)} target_state:{target_state}")

        gt_states.append(env_state.tolist())

        if exec_time >= time_limit:
            # final step
            actual_final_cost = actual_step_cost
            break
        state = env_state.copy()
        state[0:state_dim:3] -= env_state[state_dim : state_dim + 1]
        state[1:state_dim:3] -= env_state[state_dim + 1 : state_dim + 2]
        state[2:state_dim:3] -= env_state[state_dim + 2 : state_dim + 3]
        state /= scale
        # do planning only when there is no action to actuate the pusher
        if n_to_act == 0:
            reset_seed(seed)
            horizon = max_horizon if fixed_horizon else min(max_horizon, n_sim_step - i)
            n_act_step = min(max_n_act_step, horizon)
            # x_pusher, y_pusher = env.get_pusher_position()
            if enable_sampling:
                if use_prev_sol and (i != 0):
                    mppi_init_act_seq = torch.cat(
                        (
                            mppi_init_act_seq[prev_n_act_step:],
                            torch.rand(
                                (
                                    horizon - (len(mppi_init_act_seq) - prev_n_act_step),
                                    action_ub.shape[0],
                                ),
                                device=device,
                            )
                            * (action_ub - action_lb)
                            + action_lb,
                        ),
                        dim=0,
                    )
                else:
                    mppi_init_act_seq = (
                        torch.rand((horizon, action_lb.shape[0]), device=device) * (action_ub - action_lb) + action_lb
                    )
                # import pdb; pdb.set_trace()
                planning_start = time.time()
                res = planner_mppi.trajectory_optimization(env_state_to_input(state[:state_dim], device=device), mppi_init_act_seq)
                # import pdb; pdb.set_trace()
                planning_time += time.time() - planning_start
                mppi_init_act_seq = res["act_seq"]
                if open_loop:
                    planned_final_cost = -res['best_eval_output']['rewards'].item()
                    print(f"{planner_mppi.planner_type} cost: {planned_final_cost}, runtime: {planning_time}") 
                # could save the result to warmstart crown
                if warm_start_from_sampling and method_type == "CROWN":
                    # [n_sample, horizon, action_dim] -> [n_sample, horizon*action_dim]
                    warmstart_act_seqs = torch.cat(
                        [
                            res["act_seq"].view(-1, action_dim * horizon),
                            res["act_seqs_lst"][-1].view(-1, action_dim * horizon),
                        ]
                    )
                    # [n_sample]
                    warmstart_cost_seqs = -torch.cat(
                        [
                            res["best_eval_output"]["rewards"],
                            res["eval_outputs"][-1]["rewards"],
                        ]
                    )
                    warmstart_cost_seqs, sort_idx = warmstart_cost_seqs.sort(descending=False)
                    warmstart_act_seqs = warmstart_act_seqs[sort_idx]
                    # [n_sample, 1]
                    warmstart_cost_seqs = warmstart_cost_seqs.unsqueeze(1)
                    torch.save([warmstart_act_seqs, warmstart_cost_seqs], warmstart_file)
                    del warmstart_act_seqs, warmstart_cost_seqs
                # res['act_seq']: [horizon, action_dim]
                res["act_seq"] = res["act_seq"] * action_norm
                action_sequence = res["act_seq"].detach().cpu().numpy()
                state_sequence = res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
                del res["model_outputs"]
                del res["eval_outputs"]
                del res["act_seqs_lst"]
            reset_seed(seed)
            if method_type == "CROWN":
                if (i != 0) and (horizon != prev_horizon):
                    model_config["horizon"] = horizon
                    (
                        abcrown,
                        cost_model,
                        const_nodes_dict
                    ) = create_abcrown(
                        crown_name_dict,
                        crown_config_dict,
                        model_file_name,
                        model_config,
                        use_empirical_bounds,
                    )
                with open(vnnlib_path, "w") as f:
                    print_prop(
                        i,
                        f,
                        action_dim*horizon,
                        action_lb,
                        action_ub,
                    )
                # # update model
                known_input = env_state_to_input(state[:state_dim], device).view(1, state_dim//3, 3)
                pos_pusher_crown = env.get_pusher_position() # finger position + 0.5
                pos_pusher_crown = torch.tensor(pos_pusher_crown, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                update_const_nodes(const_nodes_dict, abcrown, cost_model, {"curr_state": known_input, "refer_pos": pos_pusher_crown})
                if use_prev_sol and (not warm_start_from_sampling) and (i != 0):
                    # action_sequence_copy[prev_n_act_step:]: [prev_horizon-prev_n_act_step, action_dim]
                    # repeat it to [B, prev_horizon-prev_n_act_step, action_dim] and concat with random action
                    num_warmstart = 10000
                    warmstart_act_seqs = (
                        torch.Tensor(action_sequence_copy[prev_n_act_step:])[None]
                        .to(device)
                        .repeat(num_warmstart, 1, 1)
                    )
                    warmstart_act_seqs = torch.cat(
                        (
                            warmstart_act_seqs,
                            torch.rand(
                                (
                                    num_warmstart,
                                    horizon - (action_sequence_copy.shape[0] - prev_n_act_step),
                                    action_ub.shape[0],
                                ),
                                device=device,
                            )
                            * (action_ub - action_lb)
                            + action_lb,
                        ),
                        dim=1,
                    )
                    # warmstart_act_seqs: [B, horizon, action_dim], warmstart_cost_seqs: [B, 1]
                    warmstart_cost_seqs = cost_model(warmstart_act_seqs).view(-1, 1)
                    warmstart_cost_seqs, sort_idx = warmstart_cost_seqs.sort(descending=False, dim=0)
                    warmstart_act_seqs = warmstart_act_seqs[sort_idx.squeeze()]
                    torch.save(
                        [
                            warmstart_act_seqs,
                            warmstart_cost_seqs,
                        ],
                        warmstart_file,
                    )
                    del warmstart_act_seqs, warmstart_cost_seqs
                planning_start = time.time()
                abcrown.main()
                planning_time += time.time() - planning_start
                os.system(f"rm -rf {warmstart_file}")
                feasible_sol, best_output, _, intermediate_best_outputs = parse_result(crown_sol_file, abcrown_verbose)[:4]
                assert feasible_sol is not None
                if open_loop:
                    planned_final_cost = best_output[0]
                    print(f"crown cost: {planned_final_cost}, runtime: {planning_time}")
                feasible_sol = feasible_sol.reshape(horizon, action_dim)
                action_sequence = feasible_sol * action_norm.cpu().numpy()
                action_sequence = np.resize(action_sequence, (horizon, action_dim))
                # [B, n_history, obs_dim], [B, n_history + n_rollout - 1, action_dim], n_history = 1
                state_cur_tensor = torch.from_numpy(state[:state_dim][None][None]).float().to(device)
                action_cur_tensor = torch.from_numpy(action_sequence[None]).float().to(device)
                # state_pred: [B=1, n_rollout, obs_dim]
                state_sequence = model.rollout_model({"state_init": state_cur_tensor, "action_seq": action_cur_tensor})["state_pred"]
                reward_config["pusher_pos"] = env.get_pusher_position()
                # reward can change state to absolute position
                abs_state_sequence = state_sequence.clone()
                reward(abs_state_sequence, action_cur_tensor, reward_config)
                abs_state_sequence = abs_state_sequence.squeeze(0).detach().cpu().numpy()
                state_sequence = state_sequence.squeeze(0).detach().cpu().numpy()
                res = {
                    "act_seq": action_sequence,
                    "state_seq": abs_state_sequence,
                    "planned_cost": best_output,
                    "intermediate_best_outputs": intermediate_best_outputs,
                }
                if abcrown_verbose:
                    space_size_dict = analyze_search_progress(bab_file, vis_file_name_prefix)
                    res["space_size_dict"] = space_size_dict
            n_to_act = n_act_step
            prev_n_act_step = n_act_step
            prev_horizon = horizon
            
            # copy for later use
            action_sequence_copy = action_sequence.copy()
            state_sequence_copy = state_sequence.copy()
            
            res["state_seq"] = (state_sequence * scale).tolist()
            res["start_step"] = i
            all_res.append(res)

        # actuate the pusher
        action = action_sequence_copy[n_act_step - n_to_act]
        state_planner = state_sequence_copy[n_act_step - n_to_act]

        pusher_pos = env.get_pusher_position() # finger position + 0.5
        eef_pos = env.get_end_effector_position() # finger position + 1.616-0.5, which is eef position
        eef_pos_prev = eef_pos
        
        dx, dy, dz = action[:action_dim] * scale
        
        state_planner[::3] += pusher_pos[0] / scale + action[0]
        state_planner[1::3] += pusher_pos[1] / scale + action[1]
        state_planner[2::3] += pusher_pos[2] / scale + action[2]
        
        # x_pusher_start, y_pusher_start, z_pusher_start = pusher_pos[0], pusher_pos[1], pusher_pos[2]
        # # update the state
        # x_pusher = np.clip(pusher_pos[0] + dx, lo[0], hi[0])
        # y_pusher = np.clip(pusher_pos[1] + dy, lo[1], hi[1])
        # z_pusher = np.clip(pusher_pos[2] + dz, lo[2], hi[2])
        # pusher_pos = np.array([x_pusher, y_pusher, z_pusher])
        x_eef_start, y_eef_start, z_eef_start = eef_pos[0], eef_pos[1], eef_pos[2]
        x_eef = np.clip(eef_pos[0] + dx, lo[0], hi[0])
        y_eef = np.clip(eef_pos[1] + dy, lo[1], hi[1])
        z_eef = np.clip(eef_pos[2] + dz, lo[2], hi[2])
        eef_pos = np.array([x_eef, y_eef, z_eef])
        # if any clipping happens, just print out the information
        if (x_eef != eef_pos_prev[0] + dx) or (y_eef != eef_pos_prev[1] + dy) or (z_eef != eef_pos_prev[2] + dz):
            raise ValueError(f"clipping happens at step {i}, eef_pos: {eef_pos}, eef_pos_prev: {eef_pos_prev}, action: {action}")
            print(f"clipping happens at step {i}, eef_pos: {eef_pos}, eef_pos_prev: {eef_pos_prev}, action: {action}")
        
        # exec_action_seq.append((scale * np.array([x_pusher-x_pusher_start, y_pusher-y_pusher_start, z_pusher-z_pusher_start])).tolist())
        exec_action_seq.append((scale * np.array([x_eef-x_eef_start, y_eef-y_eef_start, \
                            z_eef-z_eef_start])).tolist())
        pred_state_seq.append((scale * state_planner[:state_dim]).tolist())
        
        if verbose:
            print(f"{'-'*5}step {i}{'-'*5}  Planned action: {(dx,dy,dz)}")        




        # debug for visualization
        if enable_real:
            state_pred_for_vis = pred_state_seq.copy()
            state_pred_for_vis = np.concatenate([[gt_states[0][:state_dim]], state_pred_for_vis], axis=0)
            target_state_for_vis = target_pose.copy()
            action_seq_for_vis = exec_action_seq.copy()
            pusher_poses_for_vis = np.array(gt_states)[:, -6:-3]
            pusher_poses_for_vis = np.concatenate([pusher_poses_for_vis, \
                                                [pusher_poses_for_vis[-1] + np.array(action_seq_for_vis[-1])]])
            state_gt_for_vis = np.array(gt_states)[:, :state_dim]
            state_gt_for_vis = np.concatenate([state_gt_for_vis, state_pred_for_vis[-1:]], axis=0)
            kwargs_for_vis = {"state_gt": np.expand_dims(state_gt_for_vis, axis=0), \
                        "state_pred": np.expand_dims(state_pred_for_vis, axis=0), \
                        "pusher_poses": np.expand_dims(pusher_poses_for_vis, axis=0), \
                        "action_seq": np.expand_dims(action_seq_for_vis, axis=0), \
                        "config": config, \
                        "start_idx": int(i), \
                        "save_path": './vis_rope3d', \
                        "filename": 'debug_rope_3d',\
                        "dim_of_work_space": 3, \
                        "target_state": np.expand_dims(target_state_for_vis, axis=0), \
                        "forbidden_area": np.array(np.concatenate([env.left_forbidden_area, env.right_forbidden_area]).tolist())
                        }
            plot_task(task_name=config['task_name'], **kwargs_for_vis)






        # if len(env.fixed_action_sequence) > 1:
        #     x_eef, y_eef, z_eef = env.fixed_action_sequence.pop(0)
        # else:
        #     x_eef, y_eef, z_eef = env.fixed_action_sequence[-1]
        if enable_real:
            env.update(*[x_eef, y_eef, z_eef])
            if open_loop and enable_vis:
                time.sleep(0.1)
        else:
            env.update([x_eef, y_eef, z_eef])
            if open_loop and enable_vis:
                time.sleep(0.1)
            print(env.get_end_effector_position())
            print([x_eef, y_eef, z_eef])
        
        n_to_act -= 1
        i += 1
        if enable_real:
            exec_time = time.time() - start_time
        else:
            exec_time = i

    if success and verbose:
        cost_meter.print_metric()
    if not enable_real and save_result:
        assert vis_file_name_prefix is not None
        print(f"env.frames:{len(env.frames)}")
        imageio.mimsave(vis_file_name_prefix + f"_ori_{method_type}_{round(actual_final_cost, 3):.3f}.gif", env.frames, fps=5)
    if not enable_real and enable_vis:
        env.close()
    if enable_real:
        env.stop()
    # reformat the result,
    all_res = format_all_res(method_type, scale, all_res)

    exp_setting = {
        "open_loop": open_loop,
        "method_type": method_type,
        "init_pose": init_pose,
        "target_pose": target_pose,
        "init_pusher_pos": init_pusher_pos,
        "n_sim_step": n_sim_step,
        "horizon": horizon,
        "enable_vis": enable_vis,
        "save_result": save_result,
        "fixed_horizon": fixed_horizon,
    }
    if success:
        result_summary = {
            "success": success,
            "runtime": planning_time,
            "final_cost": actual_final_cost,
            "gt_states": gt_states,
            "exec_action_seq": exec_action_seq,
            "pred_state_seq": pred_state_seq,
        }
        result_summary.update(cost_meter.get_metric())
        result_summary["forbidden_area"] = np.concatenate([env.left_forbidden_area, env.right_forbidden_area]).tolist()

        if open_loop:
            planned_cost = cost_meter.planned_cost_seq[-1] if only_final_cost else planned_final_cost
            actual_cost = cost_meter.actual_cost_seq[-1] if only_final_cost else cost_meter.actual_acc_cost
            result_summary["planned_cost"] = planned_cost
            result_summary["actual_cost"] = actual_cost
            result_summary["cost_diff"] = actual_cost - planned_cost
    else:
        result_summary = {"success": success, "runtime": planning_time}
    print("target_offest", task_spec_dict["target_offest"], "obs_gap", task_spec_dict["obs_gap"], "init_y_angle", task_spec_dict["init_y_angle"])
    return {
        "exp_setting": exp_setting,
        "result_summary": result_summary,
        "all_res": all_res,
    }
