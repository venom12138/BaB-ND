import numpy as np
import torch
from functools import partial
from planning.utils import create_general_planning_config
from base_sim import Base_Sim
import tasks.merging_L.l_sim as l_sim
import tasks.pushing_T.t_sim as t_sim
import tasks.box_pushing.box_sim as box_sim
import tasks.inserting.hp_sim as hp_sim
import tasks.obj_pile.objpile_sim as objpile_sim
from planning.planning_hier_real_sync_for_pushing_T import rollout, reward
from planning.sampling_planner import Sampling_Planner
from others.helper import env_state_to_input, rand_float, _gen_pose_list
import time
import torch.multiprocessing as mp
from tqdm import tqdm
import warnings
import imageio
from planning.utils import get_state

warnings.filterwarnings("ignore")
# only support MPPI planner
def sample_data(root_config, model, num_episodes, episode_length, seed, queue, verbose=False, process_id=0):
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    model.eval()
    config = create_general_planning_config(root_config, {}, {}) # this is config_for_planning
    config["verbose"] = verbose
    verbose = config["verbose"]
    task_name = config["task_name"]
    task_spec_dict = config["task_spec_dict"]
    task_spec_dict["save_img"] = False
    task_spec_dict["enable_vis"] = False
    lo, hi = config["window_size"]*0.05, config["window_size"]*0.95
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    device = config["device"]
    scale = config["scale"]
    only_final_cost = config["only_final_cost"]
    action_bound = config["action_bound"]
    
    device = torch.device(process_id % torch.cuda.device_count())
    model = model.to(device)
    if task_name in ["merging_L", "pushing_T"]:
        action_lb = torch.tensor([-action_bound] * action_dim, device=device)
        action_ub = torch.tensor([action_bound] * action_dim, device=device)
    # elif task_name == "obj_pile":
    #     action_lb = torch.tensor([lo / scale, lo / scale, -action_bound, -action_bound], device=device)
    #     action_ub = torch.tensor([hi / scale, hi / scale, action_bound, action_bound], device=device)
    #     # pusher_lo = config["pusher_lo"] / scale
    #     # pusher_hi = config["pusher_hi"] / scale
    #     # action_lb = torch.tensor([pusher_lo, pusher_lo, -action_bound, -action_bound], device=device)
    #     # action_ub = torch.tensor([pusher_hi, pusher_hi, action_bound, action_bound], device=device)
    #     # action_norm = action_ub - action_lb
    #     # action_lb /= action_norm
    #     # action_ub /= action_norm
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    open_loop = False
    
    config["n_sim_step"] = episode_length - 1 #  + 1

    # open_loop = config["open_loop"]
    # task_name = config["task_name"]
    # config["n_sim_step"] = episode_length + 1
    if open_loop:
        config["horizon"] = config["n_sim_step"]
        config["n_act_step"] = config["horizon"]
    n_sim_step = config["n_sim_step"]
    max_horizon = min(config["horizon"], n_sim_step)
    max_n_act_step = min(config["n_act_step"], max_horizon)
    fixed_horizon = config["fixed_horizon"]
    use_prev_sol = config["use_prev_sol"]
    method_type = "MPPI"
    
    if verbose:
        task_spec_dict["save_img"] = True
        task_spec_dict["enable_vis"] = True
    
    if "merging_L" in task_name:
        generate_init_target_states_func = l_sim.generate_init_target_states
        sim_class = l_sim.L_Sim
        keypoint_offsets_2d = l_sim.get_offests_w_origin(task_spec_dict)
        seed = np.random.randint(0, 1023939234)
        # generate the pose
        init_pusher_pos = _gen_pose_list(1, seed, (50, 100), (50, 100), None)[0]
        init_pose = _gen_pose_list(1, seed, (100, 200), (100, 200), (90, 180), 0)[0]
        target_pose = _gen_pose_list(1, seed, (200, 400), (200, 400), (90, 180), 0)[0]
        # parse the pose
        init_pose = np.array(l_sim.parse_merging_pose(init_pose, task_spec_dict, 20, 3))
        target_pose = np.array(l_sim.parse_merging_pose(target_pose, task_spec_dict, 0, 0))
    elif "pushing_T" in task_name:
        generate_init_target_states_func = t_sim.generate_init_target_states
        sim_class = t_sim.T_Sim
        keypoint_offsets_2d = [t_sim.get_offests_w_origin(task_spec_dict)]
        seed = np.random.randint(0, 1023939234)
        # generate the pose
        init_pusher_pos = _gen_pose_list(1, seed, (50, 100), (50, 100), None)[0]
        init_pose = _gen_pose_list(1, seed, (100, 200), (100, 200), (90, 180), 0)[0]
        target_pose = _gen_pose_list(1, seed, (200, 400), (200, 400), (90, 180), 0)[0]
        # parse the pose
        init_pose = np.array([init_pose])
        target_pose = np.array([target_pose])
    # keypoint_offsets_2d: in pixel or mm in real world
    keypoint_offsets_2d = np.array(keypoint_offsets_2d)
    
    # generate init poses and target poses
    init_state, target_state = generate_init_target_states_func(init_pose, target_pose, task_spec_dict, root_config["train"]["include_com"])
    env: Base_Sim = sim_class(task_spec_dict, init_pose, target_pose, init_pusher_pos)
    # assert np.allclose(target_state[:state_dim], np.array(env.get_all_object_keypoints(True)).flatten())
    # allow the simulator to a resting position
    env_state = None
    for i in range(2):
        env.update((init_pusher_pos[0], init_pusher_pos[1]), rel=False)
        env.wait(1)
    
    config["planner_type"] = "MPPI"
    config["model_rollout_fn"] = partial(rollout, model=model)
    
    # this is obstacles
    obs_pos_list = None
    obs_size_list = None
    obs_type = None
    if "obs_pos_list" in root_config.planning and root_config.planning.penalty_type != 0:
        obs_pos_list = np.array(root_config.planning["obs_pos_list"]).tolist()
        obs_size_list = np.array(root_config.planning["obs_size_list"]).tolist()
        obs_type = root_config.planning.get("obs_type", "circle")
    
    # obj_num = state_dim // 2
    # # define cost function
    # pose_lo, pose_hi = 50, 250
    # init_poses = [[rand_float(pose_lo, pose_hi), rand_float(pose_lo, pose_hi)] for _ in range(obj_num)]
    # if verbose:
    #     task_spec_dict["save_img"] = True
    #     task_spec_dict["enable_vis"] = False
    # sim = objpile_sim.ObjectPileSim(task_spec_dict, init_poses=init_poses)
    # cls_num = sim.classes
    # labels = sim.label_list
    
    # cls_idx = []
    # for i in range(cls_num):
    #     cls_idx.append(np.where(np.array(labels) == i)[0])
    # if cost_mode == "target":
    #     corners = [[100,100], [300,300], [100, 300], [300, 100], [200, 300], [200, 100], [100, 200], [300, 200]]
    #     random.shuffle(corners)
    #     target_pose = [np.array(corners[i]) for i in range(cls_num)]
    # else:
    #     target_pose = None
    # cost_func = partial(cost_function, task_name=task_name, cls_idx=cls_idx)
    # reward_func = partial(reward, target_state=target_pose, \
    #         param_dict={"scale": scale, "state_dim": state_dim, "cost_func": cost_func, \
    #                     "only_final_cost": only_final_cost, "obj_size":task_spec_dict["obj_size"],
    #                     "task_name": task_name})
        
    # config["planner_type"] = "MPPI"
    # config["model_rollout_fn"] = partial(rollout, model=model, state_dim=state_dim, action_norm=action_norm, task_name=task_name, scale=scale)
    # config["evaluate_traj_fn"] = reward_func
    config["action_lower_lim"] = action_lb
    config["action_upper_lim"] = action_ub
    config["n_look_ahead"] = max_horizon
    dataset = []
    for episode_idx in tqdm(range(num_episodes)):
        if "merging_L" in task_name:
            seed = np.random.randint(0, 1023939234)
            # generate the pose
            init_pusher_pos = _gen_pose_list(1, seed, (50, 450), (50, 450), None)[0]
            init_pose = _gen_pose_list(1, seed, (100, 400), (100, 400), (0, 360), 0)[0]
            target_pose = _gen_pose_list(1, seed, (100, 400), (100, 400), (0, 360), 0)[0]
            # parse the pose
            init_pose = np.array(l_sim.parse_merging_pose(init_pose, task_spec_dict, 20, 3))
            target_pose = np.array(l_sim.parse_merging_pose(target_pose, task_spec_dict, 0, 0))
        elif "pushing_T" in task_name:
            seed = np.random.randint(0, 1023939234)
            # generate the pose
            init_pusher_pos = _gen_pose_list(1, seed, (50, 450), (50, 450), None)[0]
            init_pose = _gen_pose_list(1, seed, (100, 400), (100, 400), (0, 360), 0)[0]
            target_pose = _gen_pose_list(1, seed, (100, 400), (100, 400), (0, 360), 0)[0]
            # parse the pose
            init_pose = np.array([init_pose])
            target_pose = np.array([target_pose])
        
        # generate init poses and target poses
        init_state, target_state = generate_init_target_states_func(init_pose, target_pose, task_spec_dict, root_config["train"]["include_com"])
        scaled_target_state = target_state / scale
        scaled_target_state_tensor = torch.tensor(scaled_target_state, dtype=torch.float32, device=device)
        
        reward_config = {
            "action_scale": scale,
            "state_dim": state_dim,
            "enable_latent": False,
            "cost_norm": 1,
            "only_final_cost": only_final_cost,
            "target_state": scaled_target_state_tensor,
            "penalty_type": 0, # root_config.planning.get("penalty_type", 0),
            "obs_pos_list": obs_pos_list,
            "obs_size_list": obs_size_list,
            "obs_type": obs_type,
            "obs_enlarge": root_config.planning.get("obs_enlarge", 0),
        }
        config["evaluate_traj_fn"] = partial(reward, reward_config=reward_config)
        # import pdb; pdb.set_trace()
        # print(f"\nmppi: {config}\n")
        
        planner_mppi = Sampling_Planner(config)
        planner_mppi.device = device

        env.refresh(init_pose)
        for _ in range(2):
            env.update((init_pusher_pos[0], init_pusher_pos[1]), rel=False)
        # init simulation and planner finish
        
        env_state = env.get_env_state(True)[0]
        n_to_act = 0
        episode = []
        # print(f"---------env_state:{env_state/scale}---------")
        episode.append(env_state/scale)
        
        for step in range(n_sim_step):
            if open_loop and step == max_n_act_step:
                break
            # do planning only when there is no action to actuate the pusher
            if n_to_act == 0:
                horizon = max_horizon if fixed_horizon else min(max_horizon, n_sim_step - step)
                n_act_step = min(max_n_act_step, horizon)
                if method_type == "MPPI":
                    if use_prev_sol and (step != 0):
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
                    reward_config["pusher_pos"] = env.get_pusher_position()
                    # import pdb; pdb.set_trace()
                    # TODO: something wrong with mppi_init_act_seq
                    res = planner_mppi.trajectory_optimization(env_state_to_input(env_state[:state_dim], device=device), mppi_init_act_seq, skip=False)
                    mppi_init_act_seq = res["act_seq"]
                    # res["act_seq"] = res["act_seq"] * action_norm
                    # res['act_seq']: [horizon, action_dim]
                    action_sequence = res["act_seq"].detach().cpu().numpy()
                    del res["model_outputs"]
                    del res["eval_outputs"]
                    del res["act_seqs_lst"]
                
                n_to_act = n_act_step
                prev_n_act_step = n_act_step

            # actuate the pusher
            action = action_sequence[n_act_step - n_to_act] * scale

            x_pusher, y_pusher = env.get_pusher_position()
            dx, dy = action[:2]
            x_pusher = np.clip(x_pusher + dx, lo, hi)
            y_pusher = np.clip(y_pusher + dy, lo, hi)
            env.update((x_pusher, y_pusher), rel=False)
            
            # get new state
            env_state = env.get_env_state(True)[0]
            episode[-1][-action_dim:] = env_state[-action_dim:] / scale
            episode.append(env_state/scale)
            # print(f"---------env_state/scale:{env_state/scale}---------")
            if verbose:
                print(f"{'-'*5}step {step}{'-'*5}  Planned action: {(dx,dy)}")
            #     print(f"{'-'*5}step {step}{'-'*5}  Planned action: {(x_pusher_start, y_pusher_start, dx,dy)}")

            # sim.update((x_pusher, y_pusher))
            # new_state = sim.get_current_state()
            # new_state = new_state / scale
            # new_state = new_state.flatten()
            # episode[-1][-2:] = new_state[-2:]
            n_to_act -= 1
        episode = np.array(episode)
        
        dataset.append(episode)
        if verbose:
            imageio.mimsave(f"./outputs/{episode_idx}.gif", env.frames, fps=5)
            
    queue.put(dataset)
    del dataset

def online_traj_sample(root_config, model, num_episodes, num_processes, episode_length, verbose=False):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    if verbose:
        num_processes = 1
        
    process_seeds = [np.random.randint(0, 1023939234) + i for i in range(num_processes)]
    queue = mp.Queue()
    process_args = [(root_config, model, num_episodes//num_processes, episode_length, process_seeds[i], queue, verbose, i) for i in range(num_processes)]
    processes = []
    
    for rank in range(0, num_processes):
        p = mp.Process(target=sample_data, args=process_args[rank])
        p.start()
        processes.append(p)
    
    dataset = []
    for p in processes:      
        dataset.extend(queue.get())
        
    for p in processes:
        time.sleep(0.1)
        p.join()
    
    if mp.get_start_method(allow_none=True) != 'fork':
        mp.set_start_method('fork', force=True)
    
    return dataset
