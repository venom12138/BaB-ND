import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import time
import torch
import yaml
from functools import partial

from model.wrapped_model import wrapped_model
from model.mlp import MLP
from model.encoder import AE
from planning.sampling_planner import Sampling_Planner
from planning.mip_planner import MIP_Planner
from others.helper import *
from others.meter import CostMeter
from Verifier_Development.complete_verifier.abcrown import ABCROWN

from base_sim import Base_Sim
import tasks.merging_L.l_sim as l_sim
import tasks.pushing_T.t_sim as t_sim
import tasks.pushing_T_w_obs.t_obs_sim as t_obs_sim
import tasks.box_pushing.box_sim as box_sim
import tasks.inserting.hp_sim as hp_sim
from planning.utils import *
from planning_hier_real_sync_for_pushing_T import rollout, reward, step_cost

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
    assert method_type in ["MPPI_GD", "MPPI", "GD", "CROWN", "MPPI_BF", "GD_BF", "MIP", "CEM", "DecentCEM"]
    start = time.time()
    task_name = config["task_name"]
    real_exp_config = config["real_exp_config"]
    enable_real = real_exp_config["enable"]
    enable_latent = config["enable_latent"]
    enable_vis, save_result = config["enable_vis"], config["save_result"]
    use_prev_sol, fixed_horizon = config["use_prev_sol"], config["fixed_horizon"]
    only_final_cost, cost_norm = config["only_final_cost"], config["cost_norm"]
    assert cost_norm in [1, 2]
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
    state_dim, action_dim, action_bound = config["state_dim"], config["action_dim"], config["action_bound"]
    action_lb = torch.tensor([-action_bound] * action_dim, device=device)
    action_ub = torch.tensor([action_bound] * action_dim, device=device)
    # scale between pixel and state space. 100 pixel -> 1 in state space
    scale = config["scale"]
    if enable_latent:
        # norm of latent state is ~1, no need to scale, action is in pixel
        state_scale, action_scale = 1, scale
    else:
        # state and action are in pixel, need to scale
        state_scale, action_scale = scale, scale
    window_size = config["window_size"]
    pusher_pos_lb, pusher_pos_ub = window_size * 0.01, window_size * 0.99
    warm_start_from_sampling, warmstart_file = config["warm_start_from_sampling"], config["crown_name_dict"]["warmstart_file"]
    task_spec_dict = config["task_spec_dict"]

    if "merging_L" in task_name:
        generate_init_target_states_func = l_sim.generate_init_target_states
        sim_class = l_sim.L_Sim
        keypoint_offsets_2d = l_sim.get_offests_w_origin(task_spec_dict)
    elif "pushing_T_latent" in task_name:
        generate_init_target_states_func = t_obs_sim.generate_init_target_states
        sim_class = t_obs_sim.T_Obs_Sim
        keypoint_offsets_2d = [t_obs_sim.get_offests_w_origin(task_spec_dict)]
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
    init_state, target_state = generate_init_target_states_func(init_pose, target_pose, task_spec_dict, original_config["train"]["include_com"])
    enable_visualize = False
    # init_state: [state_dim], target_state: [state_dim]
    if enable_real:
        assert not enable_latent
        from real_exp_new.real_wrapper_mp import RealWrapper, get_visible_areas
        param_dict = {
            "reset": False, 
            "capture_fps": 5,
            # "record_fps": 15,
            # "record_time": 10,
        }
        # serial_numbers = ["246322301893", "246322303954", "311322303615", "311322300308"]
        serial_numbers = ["246322301893", ]
        visible_areas = get_visible_areas(serial_numbers)
        env = RealWrapper(param_dict, serial_numbers, \
            visible_areas, real_exp_config["object_path_list"], \
                real_exp_config["block_color_list"], target_state)
        env.start()
        # scale: 500 pixel -> 500mm in real world
        # state_to_real_scale = 1
        image_index = 0
        real_image_prefix = vis_file_name_prefix + "_real_step_"
        # use preception to get the initial state
        env.init_pusher_pos(*init_pusher_pos)
        # time.sleep(20)
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
        if enable_latent:
            assert not enable_real
            assert target_pusher_pos is not None
            init_img = env.gen_img_from_poses(init_pose, init_pusher_pos, vis_file_name_prefix + f"_init.png")
            target_img = env.gen_img_from_poses(target_pose, target_pusher_pos, vis_file_name_prefix + f"_target.png")
            init_img = torch.tensor(init_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
            target_img = torch.tensor(target_img).float().permute(2, 0, 1).unsqueeze(0).to(device)
            init_state = aemodel.encoder(init_img).squeeze(0).detach().cpu().numpy()
            init_state = np.concatenate([init_state, init_pusher_pos])
            target_state = aemodel.encoder(target_img).squeeze(0).detach().cpu().numpy()
        # assert np.allclose(target_state[:state_dim], np.array(env.get_all_object_keypoints(True)).flatten())
        # allow the simulator to a resting position
        env_state = None
        for i in range(2):
            env_dict = env.update((init_pusher_pos[0], init_pusher_pos[1]), rel=False)
            if enable_latent:
                img_state = torch.tensor(env_dict["image"]).float().permute(2, 0, 1).unsqueeze(0).to(device)
                # # debug
                # latent_state = aemodel.encoder(img_state)
                # decoded_img = aemodel.decoder(latent_state).detach().cpu().numpy().transpose(0, 2, 3, 1)
                # cv2.imwrite(f"{vis_file_name_prefix}_step_{i}.png", decoded_img[0] * 255)
                # print(f"saved image to {vis_file_name_prefix}_step_{i}.png")
                state = aemodel.encoder(img_state).squeeze(0).detach().cpu().numpy()
                env_state = np.concatenate([state, env_dict["pusher_pos"], env_dict["action"]], axis=0)
            else:
                env_state = np.concatenate([env_dict["state"][:state_dim], env_dict["pusher_pos"], env_dict["action"]], axis=0)
            env.wait(1)

    scaled_target_state = target_state / state_scale
    scaled_target_state_tensor = torch.tensor(scaled_target_state, dtype=torch.float32, device=device)

    # Initialize planner based on method_type
    enable_sampling = "MPPI" in method_type or "GD" in method_type or 'CEM' in method_type or (method_type == "CROWN" and warm_start_from_sampling)
    if enable_sampling:
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
        config["action_lower_lim"] = action_lb
        config["action_upper_lim"] = action_ub
        config["n_look_ahead"] = max_horizon
        planner_mppi = Sampling_Planner(config)
    if method_type == "CROWN":
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
    if method_type == "MIP":
        planner_mip = MIP_Planner(model)

    # Core simulation loop
    all_res = []
    cost_meter = CostMeter()
    exec_action_seq, pred_state_seq = [], []
    success = True
    # for real, records state before planning
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
        if enable_real:
            env_state = env.get_state(
                keypoint_offsets_2d,
                image_path=f"{real_image_prefix}{image_index}.jpeg",
                visualize=enable_visualize,
            )
            image_index += 1
        else:
            env_state, env_dict = env.get_env_state(False)
            if enable_latent:
                state = aemodel.encoder(img_state).squeeze(0).detach().cpu().numpy()
                env_state = np.concatenate([state, env_dict["pusher_pos"], env_dict["action"]], axis=0)
        gt_states.append(env_state.tolist())
        if i > 0:
            planned_step_cost = step_cost(state_planner[:state_dim], scaled_target_state, cost_norm)
            actual_step_cost = step_cost(env_state[:state_dim] / state_scale, scaled_target_state, cost_norm)
            cost_meter.update(actual_step_cost, planned_step_cost, i-1)
        if exec_time >= time_limit:
            # final step
            actual_final_cost = actual_step_cost
            break
        x_pusher_for_plan, y_pusher_for_plan = env_state[state_dim : state_dim + 2]
        # do planning only when there is no action to actuate the pusher
        if n_to_act == 0:
            reset_seed(seed)
            # data processing: convert to relative coordinate
            # (x1, y1, x2, y2, x3, y3, x4, y4, ...,xp, yp, vx, vy)
            # ->(x1-xp, y1-yp, x2-xp, y2-yp, x3-xp, y3-yp, x4-xp, y4-yp, ..., xp, yp, vx, vy)
            if not enable_latent:
                # relative coordinate
                env_state[0:state_dim:2] -= env_state[state_dim : state_dim + 1]
                env_state[1:state_dim:2] -= env_state[state_dim + 1 : state_dim + 2]
            state = env_state[:state_dim] / state_scale
            horizon = max_horizon if fixed_horizon else min(max_horizon, n_sim_step - i)
            n_act_step = min(max_n_act_step, horizon)
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
                    # mppi_init_act_seq = torch.zeros((horizon, action_lb.shape[0]), device=device)
                reward_config["pusher_pos"] = [x_pusher_for_plan, y_pusher_for_plan]
                planning_start = time.time()
                res = planner_mppi.trajectory_optimization(env_state_to_input(state, device=device), mppi_init_act_seq)
                planning_time += time.time() - planning_start
                mppi_init_act_seq = res["act_seq"]
                if open_loop:
                    planned_final_cost = -res['best_eval_output']['rewards'].item()
                    print(f"mppi cost: {planned_final_cost}") 
                # could save the result to warmstart crown
                if method_type == "CROWN" and warm_start_from_sampling:
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
                action_sequence = res["act_seq"].detach().cpu().numpy()
                state_sequence = res["best_model_output"]["state_seqs"][0].detach().cpu().numpy()
                del res["model_outputs"], res["eval_outputs"], res["act_seqs_lst"]
            reset_seed(seed)
            if method_type == "CROWN":
                known_dim = state_dim * n_his + action_dim * (n_his - 1)
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
                    update_const_nodes(const_nodes_dict, abcrown, cost_model, 
                                       {"target_state": scaled_target_state_tensor.unsqueeze(0)})
                with open(vnnlib_path, "w") as f:
                    print_prop(
                        i,
                        f,
                        action_dim * horizon,
                        action_lb,
                        action_ub,
                        # curr_cost * horizon,
                    )
                # update model
                known_input = torch.Tensor(state[:known_dim][None])
                pos_pusher_crown = (torch.Tensor([x_pusher_for_plan, y_pusher_for_plan]) / action_scale).unsqueeze(0)
                update_const_nodes(const_nodes_dict, abcrown, cost_model, {"known_input": known_input, "refer_pos": pos_pusher_crown})

                if use_prev_sol and (i != 0):
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
                    warmstart_cost_seqs = cost_model(warmstart_act_seqs.view(-1, action_dim * horizon)).view(-1, 1)
                    warmstart_cost_seqs, sort_idx = warmstart_cost_seqs.sort(descending=False, dim=0)
                    warmstart_act_seqs = warmstart_act_seqs[sort_idx.squeeze()]
                    torch.save(
                        [
                            warmstart_act_seqs.view(-1, action_dim * horizon),
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
                    print(f"crown cost: {planned_final_cost}") 
                action_sequence = feasible_sol
                action_sequence = np.resize(action_sequence, (horizon, action_dim))
                # [B, n_history, obs_dim], [B, n_history + n_rollout - 1, action_dim], n_history = 1
                state_cur_tensor = torch.from_numpy(state[:state_dim][None][None]).float().to(device)
                action_cur_tensor = torch.from_numpy(action_sequence[None]).float().to(device)
                # state_pred: [B=1, n_rollout, obs_dim]
                state_sequence = model.rollout_model({"state_init": state_cur_tensor,  "action_seq": action_cur_tensor})["state_pred"]
                state_sequence = state_sequence.squeeze(0).detach().cpu().numpy()
                res = {
                    "act_seq": action_sequence,
                    "state_seq": state_sequence,
                    "planned_cost": best_output,
                    "intermediate_best_outputs": intermediate_best_outputs,
                }
                if abcrown_verbose:
                    space_size_dict = analyze_search_progress(bab_file, vis_file_name_prefix)
                    res["space_size_dict"] = space_size_dict
                # if enable_real:
                #     planned_cost_real = best_output.tolist()[0]
            if method_type == "MIP":
                planning_start = time.time()
                # current state (rel coord), [n_his, state_dim], current pusher position (abs coord), [2]
                # scaled_target_state (abs coord), [state_dim],  initial action sequence, [-1, action_dim]
                res = planner_mip.trajectory_optimization(
                    state[:state_dim][None],
                    np.array([x_pusher_for_plan, y_pusher_for_plan]) / scale,
                    scaled_target_state,
                    np.empty((0, action_dim)),
                    action_bound,
                    horizon,
                    only_final_cost,
                )
                planning_time += time.time() - planning_start
                # if enable_real:
                #     planned_cost_real = res["objective"]
                # action_sequence: [n_roll, action_dim], observation_sequence: [n_roll, obs_dim]
                if "action_sequence" in res:
                    action_sequence = res["action_sequence"]
                    state_sequence = res["observation_sequence"]
                    if open_loop:
                        planned_final_cost = res["objective"]
                        print(f"mip cost: {planned_final_cost}") 
                else:
                    print(f'MIP failed to find a solution: {res["solve_status"]}')
                    success = False
                    all_res.append(res)
                    break
            n_to_act = n_act_step
            prev_n_act_step = n_act_step
            prev_horizon = horizon
            # copy for later use
            action_sequence_copy = action_sequence.copy()
            state_sequence_copy = state_sequence.copy()
            if not enable_latent:
                # update original data to absolute coordinate
                x_pusher_planned, y_pusher_planned = x_pusher_for_plan / action_scale, y_pusher_for_plan / action_scale
                for j in range(horizon):
                    x_pusher_planned += action_sequence[j, 0]
                    y_pusher_planned += action_sequence[j, 1]
                    state_sequence[j, ::2] += x_pusher_planned
                    state_sequence[j, 1::2] += y_pusher_planned

            res["state_seq"] = (state_sequence * state_scale).tolist()
            res["start_step"] = i
            all_res.append(res)
            # if not only_final_cost:
            #     planned_cost_real = step_cost(state_sequence[-1][:state_dim], scaled_target_state, cost_norm)
        
        # calculate the action
        action = action_sequence_copy[n_act_step - n_to_act]
        state_planner = state_sequence_copy[n_act_step - n_to_act]

        dx, dy = action_scale * action
        print(f"{'-'*5}step {i}{'-'*5}  Planned action: {(dx,dy)}")
        next_pusher_pos = (np.clip(x_pusher_for_plan + dx, pusher_pos_lb, pusher_pos_ub),
                            np.clip(y_pusher_for_plan + dy, pusher_pos_lb, pusher_pos_ub))
        if not enable_latent:
            # convert to absolute coordinate
            state_planner[::2] += x_pusher_for_plan / action_scale + action[0]
            state_planner[1::2] += y_pusher_for_plan / action_scale + action[1]
        exec_action_seq.append((action_scale * action).tolist())
        pred_state_seq.append((state_scale * state_planner[:state_dim]).tolist())

        # execute the action
        if enable_real:
            env.update(*next_pusher_pos, wait=False)
            # env.update_waypoint(next_pusher_pos)
        else:
            env.update(next_pusher_pos, rel=False)
            if open_loop and enable_vis:
                time.sleep(0.1)

        n_to_act -= 1
        i += 1
        if enable_real:
            exec_time = time.time() - start_time
        else:
            exec_time = i
    print(f"{'-'*5}Planning time: {planning_time}{'-'*5}")
    if success:
        cost_meter.print_metric()
        if not enable_real and save_result:
            assert vis_file_name_prefix is not None
            env.save_gif(vis_file_name_prefix + f"_ori_{method_type}_{round(actual_final_cost, 3)}.gif")
            # env.save_mp4(vis_file_name_prefix + f"_ori_{method_type}_{round(actual_final_cost, 3)}.mp4")
    if not enable_real and enable_vis:
        env.close()
    if enable_real:
        env.stop()
    # reformat the result,
    all_res = format_all_res(method_type, action_scale, all_res)
    if penalty_type == 0:
        gt_safe = True
        pred_safe = True
    else:
        gt_safe = check_state_obs(gt_states, state_dim, obs_pos_list, obs_size_list, obs_type, obs_enlarge, scale)
        pred_safe = check_state_obs(pred_state_seq, state_dim, obs_pos_list, obs_size_list, obs_type, obs_enlarge, scale)
    print(f"{'-'*5}gt safe: {gt_safe}, pred safe: {pred_safe}{'-'*5}")

    exp_setting = {
        "open_loop": open_loop,
        "method_type": method_type,
        "init_pose": init_pose.tolist(),
        "target_pose": target_pose.tolist(),
        "init_pusher_pos": init_pusher_pos.tolist(),
        "n_sim_step": n_sim_step,
        "horizon": horizon,
        "enable_vis": enable_vis,
        "save_result": save_result,
        "fixed_horizon": fixed_horizon,
    }
    exp_setting.update( {"obs_pos_list": obs_pos_list, "obs_size_list": obs_size_list, "obs_type": obs_type, "obs_enlarge": obs_enlarge, "scale": scale})
    try:
        result_summary = {
            "success": success,
            "final_cost": actual_final_cost,
            "gt_safe": gt_safe, 
            "pred_safe": pred_safe,
            "gt_states": gt_states,
            "exec_action_seq": exec_action_seq,
            "pred_state_seq": pred_state_seq,
            "runtime": planning_time,
        }
        result_summary.update(cost_meter.get_metric())
        if open_loop:
            # # hack for collect data
            # tmp_file = "result_cem/cost_test.txt"
            # with open(tmp_file, "a") as f:
            #     f.write(f"method: {method_type}, planned cost: {planned_final_cost}, runtime: {planning_time}\n")
            planned_cost = cost_meter.planned_cost_seq[-1] if only_final_cost else planned_final_cost
            actual_cost = cost_meter.actual_cost_seq[-1] if only_final_cost else cost_meter.actual_acc_cost
            result_summary["planned_cost"] = planned_cost
            result_summary["actual_cost"] = actual_cost
            result_summary["cost_diff"] = actual_cost - planned_cost
    except:
        result_summary = {"success": success, "runtime": planning_time}
    return {
        "exp_setting": exp_setting,
        "result_summary": result_summary,
        "all_res": all_res,
    }


# # rollout function for mppi
# # [n_his=1, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample, n_look_ahead, state_dim]
# def rollout(state_cur: torch.Tensor, act_seqs: torch.Tensor, model: MLP):
#     # [n_sample, n_his, state_dim]
#     state_cur_mppi = state_cur.unsqueeze(0).repeat(act_seqs.shape[0], 1, 1)
#     res_dict = model.rollout_model({"state_init": state_cur_mppi, "action_seq": act_seqs})
#     return {"state_seqs": res_dict["state_pred"]}  # [n_sample, n_look_ahead, state_dim]

# # reward function for mppi
# # [n_sample, n_look_ahead, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample]
# def reward(state_mppi, action_mppi, reward_config):
#     action_scale, state_dim = reward_config["action_scale"], reward_config["state_dim"]
#     enable_latent, cost_norm, only_final_cost = reward_config["enable_latent"], reward_config["cost_norm"], reward_config["only_final_cost"]
#     target_state, pusher_pos = reward_config["target_state"], reward_config["pusher_pos"]
#     n_sample, n_look_ahead, action_dim = action_mppi.shape
#     state_mppi = state_mppi.clone()
#     pos_pusher_mppi = torch.zeros((n_sample, n_look_ahead + 1, action_dim), device=action_mppi.device)
#     pos_pusher_mppi[:, 0, 0:1] = pos_pusher_mppi[:, 0, 0:1] + (pusher_pos[0] / action_scale)
#     pos_pusher_mppi[:, 0, 1:2] = pos_pusher_mppi[:, 0, 1:2] + (pusher_pos[1] / action_scale)
#     for i in range(n_look_ahead):
#         pos_pusher_mppi[:, i + 1, :] = pos_pusher_mppi[:, i, :] + action_mppi[:, i, :]
#     if not enable_latent:
#         state_mppi[:, :, :state_dim:2] = state_mppi[:, :, :state_dim:2] + pos_pusher_mppi[:, 1:, 0:1]
#         state_mppi[:, :, 1:state_dim:2] = state_mppi[:, :, 1:state_dim:2] + pos_pusher_mppi[:, 1:, 1:2]
#     obs_pos = torch.tensor([2.5,2.5], device=state_mppi.device)
#     max_penalty = 0.01
#     min_penalty = torch.tensor(0.0, device=state_mppi.device)
#     # # add penalty
#     penalty = 0
#     penalty = 100000*torch.max(-(pos_pusher_mppi[:, 1:, :2] - obs_pos)**2+max_penalty, min_penalty)
#     penalty = penalty.sum(dim=-1).sum(dim=-1)
#     # [n_sample, n_look_ahead]
#     cost_seqs = (
#         torch.norm(state_mppi[:, :, :state_dim] - target_state[:state_dim], p=cost_norm, dim=-1)
#         ** cost_norm
#     ) 

#     costs = cost_seqs.sum(dim=-1)+penalty if not only_final_cost else cost_seqs[:, -1] + penalty
#     # print("penalty: ", penalty.max().item(), penalty.min().item())
#     # print("lowest cost: ", costs.min().item(), "highest cost: ", costs.max().item())
#     # costs = cost_seqs.sum(dim=-1) if not only_final_cost else cost_seqs[:, -1]
#     # costs += 10000 * torch.sum(abs(action_mppi[:, -1, :]) > 0.0001, dim=1)
#     # costs += 10000 * torch.sum(abs(action_mppi[:, -1, :]), dim=1)
#     # costs += 10000 * torch.sum(action_mppi[:, -1, :] != torch.tensor([0, 0], device=action_mppi.device), dim=1)
#     # print("lowest cost: ", costs.min().item(), "highest cost: ", costs.max().item())
#     rewards = -costs
#     return {
#         "rewards": rewards,
#         "cost_seqs": cost_seqs,
#     }

# def step_cost(state, target_state, cost_norm):
#     return (np.linalg.norm(target_state - state, cost_norm)) ** cost_norm


