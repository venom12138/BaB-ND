import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F
import time
import os
from others.helper import reset_seed
from copy import deepcopy
# torch.autograd.set_detect_anomaly(False)

# Tips to tune MPC:
# - When sampling actions, noise_level should be large enough to have enough coverage, but not too large to cause instability
# - Larger n_sample should lead to better performance, but it will also increase the computation cost
# - Properly tune reward_weight, higher reward_weight encourages to 'exploit' the current best action sequence, while lower reward_weight encourages to 'explore' more action sequences
# - Plot reward vs. iteration to see the convergence of the planner


def fps_np(pcd, num, init_idx=-1):
    # pcd: (n, c) numpy array
    # pcd_fps: (num, c) numpy array
    # radius: float
    n, c = pcd.shape
    fps_idx = []
    assert pcd.shape[0] > 0
    if init_idx == -1:
        # rand_idx = np.random.randint(pcd.shape[0])
        # choose the idx with largest motion
        motion_dist = np.linalg.norm(pcd[:, (c // 2) :] - pcd[:, : (c // 2)], axis=1)
        rand_idx = motion_dist.argmax()
    else:
        rand_idx = init_idx
    fps_idx.append(rand_idx)
    pcd_fps_lst = [pcd[rand_idx]]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while len(pcd_fps_lst) < num:
        fps_idx.append(dist.argmax())
        pcd_fps_lst.append(pcd[dist.argmax()])
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    return pcd_fps

class Sampling_Planner(object):

    def __init__(self, config: dict):
        # config contains following keys:

        # REQUIRED
        # - action_dim: the dimension of the action
        # - model_rollout_fn:
        #   - description: the function to rollout the model
        #   - input:
        #     - state_cur (shape: [n_his, state_dim] torch tensor)
        #     - action_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - output: a dict containing the following keys:
        #     - state_seqs: the sequence of the state, shape: [n_sample, n_look_ahead, state_dim] torch tensor
        #     - any other keys that you want to return
        # - evaluate_traj_fn:
        #   - description: the function to evaluate the trajectory
        #   - input:
        #     - state_seqs (shape: [n_sample, n_look_ahead, state_dim] torch tensor)
        #     - action_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - output: a dict containing the following keys:
        #     - rewards (shape: [n_sample] torch tensor)
        #     - any other keys that you want to return
        # - n_sample: the number of action trajectories to sample
        # - n_look_ahead: the number of steps to look ahead
        # - n_update_iter: the number of iterations to update the action sequence
        # - reward_weight: the weight of the reward to aggregate action sequences
        # - action_lower_lim:
        #   - description: the lower limit of the action
        #   - shape: [action_dim]
        #   - type: torch tensor
        # - action_upper_lim: the upper limit of the action
        #   - description: the upper limit of the action
        #   - shape: [action_dim]
        #   - type: torch tensor
        # - planner_type: the type of the planner (options: 'GD', 'MPPI', 'MPPI_GD')
        self.config = config
        self.action_dim = config["action_dim"]
        self.model_rollout = config["model_rollout_fn"]
        self.evaluate_traj = config["evaluate_traj_fn"]
        self.n_look_ahead = config["n_look_ahead"]

        self.action_lower_lim: torch.Tensor = config["action_lower_lim"]
        self.action_upper_lim: torch.Tensor = config["action_upper_lim"]
        self.planner_type = config["planner_type"]
        self.seed = config["seed"]
        # torch.manual_seed(self.seed)
        # np.random.seed(self.seed)
        assert self.planner_type in ["GD", "GD_BF", "MPPI", "MPPI_GD", "CROWN", "MPPI_BF", "CEM", "DecentCEM"]
        self.brute_force = False
        if self.planner_type == "CROWN":
            self.planner_type = "MPPI"
        if self.planner_type == "MPPI_BF":
            self.planner_type = "MPPI"
            self.brute_force = True
        if self.planner_type == "GD_BF":
            self.planner_type = "GD"
            self.brute_force = True
        self.experiment_folder = config.get("experiment_folder", None)
        self.save_planning_result = config.get("save_planning_result", False)
        assert self.action_lower_lim.shape == (self.action_dim,)
        assert self.action_upper_lim.shape == (self.action_dim,)
        assert type(self.action_lower_lim) == torch.Tensor
        assert type(self.action_upper_lim) == torch.Tensor
        
        if "MPPI" in self.planner_type:
            mppi_config = config["mppi_config"]
            self.n_sample = mppi_config["n_sample"]
            self.n_update_iter = mppi_config["n_update_iter"]
            self.reward_weight = mppi_config["reward_weight"]
            self.noise_type = mppi_config.get("noise_type", "normal")
            assert self.noise_type in ["normal", "fps", "general"], "unknown noise type: %s" % self.noise_type
            self.noise_level = mppi_config.get("noise_level", 0.1)
            self.noise_decay = mppi_config.get("noise_decay", 1)
            self.reject_bad = mppi_config.get("reject_bad", False)
            if self.noise_level == "auto":
                self.noise_level = 0.15 * (self.action_upper_lim - self.action_lower_lim).max()
            if "MPPI_GD" in self.planner_type:
                self.lr = mppi_config.get("lr", 1e-3)
        elif "CEM" in self.planner_type:
            cem_config = config["cem_config"]
            self.n_sample = cem_config["n_sample"]
            self.n_update_iter = cem_config["n_update_iter"]
            self.jitter_factor = cem_config.get("jitter_factor", 0.001) # jitter for CEM, DecentCEM, small value to ensure positive definiteness
            self.elite_ratio = cem_config.get("elite_ratio", 0.001) # ratio of elite samples for CEM, DecentCEM
            self.min_n_elites = cem_config.get("min_n_elites", 10) # minimum number of elites for CEM
            if "DecentCEM" in self.planner_type:
                self.n_agents = cem_config["n_agents"]
        elif "GD" in self.planner_type:
            gd_config = config["gd_config"]
            self.n_sample = gd_config["n_sample"]
            self.n_update_iter = gd_config["n_update_iter"]
            self.lr = gd_config.get("lr", 1e-3)
            self.noise_type = gd_config.get("noise_type", "random")
            self.lr_decay = gd_config.get("lr_decay", 1)
            self.reject_bad = gd_config.get("reject_bad", False)
            assert self.noise_type in ["normal", "fps", "general", "random"], "unknown noise type: %s" % self.noise_type
            if self.brute_force:
                self.n_restart_round = gd_config.get("n_restart_round", 2)
        # OPTIONAL
        # - device: 'cpu' or 'cuda', default: 'cuda'
        # - verbose: True or False, default: False
        # - sampling_action_seq_fn:
        #   - description: the function to sample the action sequence
        #   - input: init_act_seq (shape: [n_look_ahead, action_dim] torch tensor)
        #   - output: act_seqs (shape: [n_sample, n_look_ahead, action_dim] torch tensor)
        #   - default: sample action sequences from a normal distribution
        # - noise_type: the type of the noise (options: 'normal', 'fps', 'general'), default: 'normal'
        # - noise_level: the level of the noise, default: 0.1
        # - n_his: the number of history states to use, default: 1
        # - rollout_best: whether rollout the best act_seq and get model prediction and reward. True or False, default: True
        # - lr: the learning rate of the optimizer, default: 1e-3
        self.device = config.get("device", "cuda")
        # if self.device == "cuda":
        #     print("using cuda")
        self.verbose = config.get("verbose", False)
        self.sample_action_sequences = (
            config["sampling_action_seq_fn"]
            if "sampling_action_seq_fn" in config
            else self.sample_action_sequences_default
        )
        self.n_his = config.get("n_his", 1)
        self.rollout_best = config.get("rollout_best", True)
        self.report_interval = config.get("report_interval", 10)

    def sample_action_sequences_default(self, act_seq):
        # init_act_seq: shape: [n_look_ahead, action_dim] torch tensor
        # return: shape: [n_sample, n_look_ahead, action_dim] torch tensor
        # assert act_seq.shape == (self.n_look_ahead, self.action_dim)
        # assert type(act_seq) == torch.Tensor
        n_look_ahead, action_dim = act_seq.shape
        if self.noise_type == "fps":
            action_lower_lim_np = self.action_lower_lim.cpu().numpy()
            action_upper_lim_np = self.action_upper_lim.cpu().numpy()
            grid_size = 0.02
            grid_axis = []
            for i in range(self.action_dim):
                grid_axis.append(np.arange(action_lower_lim_np[i], action_upper_lim_np[i], grid_size))
            grids = np.meshgrid(*grid_axis)
            grids = np.stack(grids, axis=-1).reshape(-1, self.action_dim)
            act_seqs = fps_np(grids, self.n_sample)  # (n_sample, action_dim)
            act_seqs = torch.from_numpy(act_seqs).to(self.device).float()
            act_seqs = act_seqs.unsqueeze(1).repeat(1, n_look_ahead, 1)
            return act_seqs
        elif self.noise_type == "general":
            return self.sample_action_sequences_general(act_seq)
        elif self.noise_type == "random":
            return torch.rand((self.n_sample, n_look_ahead, action_dim), device=self.device) * (self.action_upper_lim - self.action_lower_lim) + self.action_lower_lim
        beta_filter = 0.7

        # [n_sample, n_look_ahead, action_dim]
        act_seqs = act_seq.unsqueeze(0).repeat(self.n_sample, 1, 1)

        # noise_sample = torch.normal(0, self.noise_level, (self.n_sample, n_look_ahead, self.action_dim), device=self.device)
        # return act_seqs + noise_sample

        # [n_sample, action_dim]
        act_residual = torch.zeros((self.n_sample, self.action_dim), dtype=act_seqs.dtype, device=self.device)

        # actions that go as input to the dynamics network
        for i in range(n_look_ahead):
            if self.noise_type == "normal":
                noise_sample = torch.normal(0, self.noise_level, (self.n_sample, self.action_dim), device=self.device)
            else:
                raise ValueError("unknown noise type: %s" % (self.noise_type))

            act_residual = beta_filter * noise_sample + act_residual * (1.0 - beta_filter)

            # add the perturbation to the action sequence
            act_seqs[:, i] += act_residual

            # clip to range
            act_seqs[:, i] = torch.clamp(act_seqs[:, i], self.action_lower_lim, self.action_upper_lim)

        # assert act_seqs.shape == (self.n_sample, n_look_ahead, self.action_dim)
        # assert type(act_seqs) == torch.Tensor
        return act_seqs

    def sample_action_sequences_general(self, act_seq):
        # init_act_seq: shape: [n_look_ahead, action_dim] torch tensor
        # return: shape: [n_sample, n_look_ahead, action_dim] torch tensor

        act_seqs = act_seq.unsqueeze(0).repeat(self.n_sample, 1, 1)
        act_seqs = act_seqs + torch.normal(0, self.noise_level, act_seqs.shape, device=self.device)
        act_seqs.data.clamp_(self.action_lower_lim, self.action_upper_lim)
        return act_seqs

    def optimize_action(self, act_seqs, rewards, optimizer: optim.Adam = None):
        # act_seqs: shape: [n_sample, n_look_ahead, action_dim] torch tensor
        # rewards: shape: [n_sample] torch tensor
        # optimizer: optimizer for GD, default: None
        # assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
        # assert rewards.shape == (self.n_sample,)
        # assert type(act_seqs) == torch.Tensor
        # assert type(rewards) == torch.Tensor

        if self.planner_type == "MPPI":
            return self.optimize_action_mppi(act_seqs, rewards)
        elif self.planner_type == "GD":
            return self.optimize_action_gd(act_seqs, rewards, optimizer)
        elif self.planner_type == "MPPI_GD":
            raise self.optimize_action_mppi_gd(act_seqs, rewards, optimizer)
        else:
            raise ValueError("unknown planner type: %s" % (self.planner_type))

    def trajectory_optimization(self, state_cur, act_seq, skip=False):
        # input:
        # - state_cur: current state, shape: [n_his, state_dim] torch tensor
        # - act_seq: initial action sequence, shape: [n_look_ahead, action_dim] torch tensor
        # - skip: whether to skip the optimization and return the initial action sequence, default: False
        # output:
        # - a dictionary with the following keys:
        #   - 'act_seq': optimized action sequence, shape: [n_look_ahead, action_dim] torch tensor
        #   - 'model_outputs' if verbose is True, otherwise None, might be useful for debugging
        #   - 'eval_outputs' if verbose is True, otherwise None, might be useful for debugging
        #   - 'best_model_output' if rollout_best is True, otherwise None, might be useful for debugging
        #   - 'best_eval_output' if rollout_best is True, otherwise None, might be useful for debugging
        # assert type(state_cur) == torch.Tensor
        # # assert act_seq.shape == (self.n_look_ahead, self.action_dim)
        # assert type(act_seq) == torch.Tensor
        if skip:
            init_reward, init_model_out, init_eval_out = self.calculate_reward(state_cur, act_seq.unsqueeze(0))
            return {
            "act_seq": act_seq,
            "model_outputs": init_model_out if self.verbose else None,
            "eval_outputs": init_eval_out if self.verbose else None,
            "act_seqs_lst": None if self.verbose else None,
            "best_model_output": init_model_out if self.rollout_best else None,
            "best_eval_output": init_eval_out if self.rollout_best else None,
            }
        if self.brute_force and self.planner_type == "MPPI":
            return self.trajectory_optimization_mppi_bf(state_cur, act_seq)
        if self.brute_force and self.planner_type == "GD":
            return self.trajectory_optimization_gd_bf(state_cur, act_seq)
        if self.planner_type == "MPPI":
            return self.trajectory_optimization_mppi(state_cur, act_seq)
        elif self.planner_type == "GD":
            return self.trajectory_optimization_gd(state_cur, act_seq)
        elif self.planner_type == "MPPI_GD":
            return self.trajectory_optimization_mppi_gd(state_cur, act_seq)
        elif self.planner_type == "CEM":
            return self.trajectory_optimization_cem(state_cur, act_seq)
        elif self.planner_type == "DecentCEM":
            return self.trajectory_optimization_decent_cem(state_cur, act_seq)
        else:
            raise ValueError("unknown planner type: %s" % (self.planner_type))

    # [n_sample, n_look_ahead, action_dim], [n_sample] -> [n_look_ahead, action_dim]
    def optimize_action_mppi(self, act_seqs: torch.Tensor, rewards: torch.Tensor):
        weights = F.softmax(rewards * self.reward_weight, dim=0)
        act_seq = torch.sum(act_seqs * weights.unsqueeze(-1).unsqueeze(-1), dim=0)
        return self.clip_actions(act_seq)

    def optimize_action_gd(self, act_seqs: torch.Tensor, rewards: torch.Tensor, optimizer: optim.Adam):
        loss = -torch.sum(rewards)

        loss.backward()
        try:
            assert torch.isnan(act_seqs.grad).sum() == 0
        except:
            print("act_seqs:", act_seqs)
            print("act_seqs.grad:", act_seqs.grad)
            exit()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    def optimize_action_mppi_gd(self, act_seqs: torch.Tensor, state_cur: torch.Tensor):
        model_out_before, eval_out_before = {}, {}
        optimizer = torch.optim.Adam([act_seqs], lr=self.lr, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, verbose=False)
        num_iter = self.n_update_iter // 4
        for j in range(num_iter+1):
            # assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
            rewards, model_out, eval_out = self.calculate_reward(state_cur, act_seqs)
            # [n_sample, n_look_ahead, action_dim], [n_sample] -> update act_seqs
            # self.optimize_action_gd(act_seqs, rewards, optimizer)
            if j > 0:
                bad_update_indices = rewards < rewards_before
                print(f"reject {bad_update_indices.sum().item()} bad updates from {rewards.shape[0]} in iteration {j}")
                act_seqs.data[bad_update_indices] = act_seqs_before[bad_update_indices]
                rewards.data[bad_update_indices] = rewards_before[bad_update_indices]
                model_out["state_seqs"].data[bad_update_indices] = model_out_before["state_seqs"][bad_update_indices]
                eval_out["rewards"].data[bad_update_indices] = eval_out_before["rewards"][bad_update_indices]
            if j == num_iter:
                break
            rewards_before = rewards.clone()
            model_out_before["state_seqs"] = model_out["state_seqs"].clone()
            eval_out_before["rewards"] = eval_out["rewards"].clone()
            act_seqs_before = act_seqs.clone()
            loss = -torch.mean(rewards)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            self.clip_actions(act_seqs)
        del model_out_before["state_seqs"], eval_out_before["rewards"]
        del model_out_before, eval_out_before, act_seqs_before, rewards_before
        del model_out, eval_out, rewards
        del optimizer, scheduler


    def clip_actions(self, act_seqs):
        # act_seqs: shape: [**dim, action_dim] torch tensor
        # return: shape: [**dim, action_dim] torch tensor
        act_seqs.data.clamp_(self.action_lower_lim, self.action_upper_lim)
        return act_seqs

    # launch multiple mppis with different hyperparameters and return the best one
    def trajectory_optimization_mppi_bf(self, state_cur, act_seq):
        # n_sample, n_update_iter, reward_weight, noise_level
        # self.n_update_iter = 50
        # reward_weight_list = [1, 5, 10, 50, 100]
        # self.n_sample = 320000
        # self.n_update_iter = 15 # 50
        # # self.n_update_iter = 2
        # reward_weight_list = [10, 100, 500, ] # 50 1000
        base_reward_weight = self.reward_weight
        base_noise_level = self.noise_level
        if isinstance(base_reward_weight, torch.Tensor):
            base_reward_weight = base_reward_weight.cpu().detach().numpy()
        if isinstance(base_noise_level, torch.Tensor):
            base_noise_level = base_noise_level.cpu().detach().numpy()
        raw_base_noise_level = deepcopy(base_noise_level)
        raw_base_reward_weight = deepcopy(base_reward_weight)
        
        noise_level_list = np.array([0.1, 0.5, 1, 1.5, 2]) * base_noise_level
        reward_weight_list = np.array([0.1, 0.5, 1, 1.5, 2]) * base_reward_weight
        # # T, L, rope
        # noise_level_list = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5]) * base_noise_level
        # reward_weight_list = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5]) * base_reward_weight
        # # rope
        # noise_level_list = np.array([0.1, 0.5, 1, 2, 5]) * base_noise_level
        # reward_weight_list = np.array([0.1, 0.5, 1, 5]) * base_reward_weight
        # pile
        # noise_level_list = np.array([0.5, 1, 1.5]) * base_noise_level
        # reward_weight_list = np.array([0.1, 1, 5]) * base_reward_weight
        # noise_level_list = np.array([1] * 10) * base_noise_level
        # reward_weight_list = np.array([1] * 10) * base_reward_weight
        n_restarts = len(noise_level_list) * len(reward_weight_list)
        # f = open(os.path.join(self.experiment_folder, "mppi_bf.txt"), "a")
        # print(f"{'noise':<8} {'weight':<8} {'cost':<8}", file=f)
        best_reward = -np.inf
        best_result = None
        best_reward_weight = None
        best_noise_level = None
        global_reward_list = []

        for reward_weight in reward_weight_list:
            for noise_level in noise_level_list:
                self.reward_weight = reward_weight
                self.noise_level = noise_level
                result = self.trajectory_optimization_mppi(state_cur.clone(), act_seq.clone())

                # print(f"MPPI_BF: noise: {noise_level}, weight: {reward_weight} reward: {result['best_eval_output']['rewards'].item()}")
                # print(f"{noise_level:<8.3f} {reward_weight:<8} {-result['best_eval_output']['rewards'].item():<8.5f}", file=f)
                
                global_reward_list.append(result["reward_list"])
                self.seed += 1

                if result["best_eval_output"]["rewards"] > best_reward:
                    best_result = result
                    best_reward = result["best_eval_output"]["rewards"]
                    best_reward_weight = reward_weight
                    best_noise_level = noise_level
        # print("--------best result--------", file=f)
        # print(f"{best_noise_level:<8.3f} {best_reward_weight:<8} {-best_result['best_eval_output']['rewards'].item():<8.5f}", file=f)
        
        # [n_restarts, n_update_iter]
        global_rewards = torch.stack(global_reward_list, dim=0)
        self.noise_level = raw_base_noise_level
        self.reward_weight = raw_base_reward_weight
        if self.save_planning_result:
            report_stat(global_rewards, os.path.join(self.experiment_folder, "mppi_bf.txt"), self.report_interval)

        return best_result


    # [n_his, state_dim], [n_look_ahead, action_dim]
    def trajectory_optimization_mppi(self, state_cur, act_seq, init_pusher_pos=None):
        reset_seed(self.seed)
        if self.verbose:
            model_outputs = []
            eval_outputs = []
            act_seqs_lst = []
        reward_list = []
        best_reward = torch.tensor(-float('inf'), device=self.device)
        best_act_seq = act_seq
        reject_bad = getattr(self, "reject_bad", False)
        rewards_prev = None
        act_seqs_prev = None
        model_out_prev = None
        eval_out_prev = None

        with torch.no_grad():
            for i in range(self.n_update_iter+1):
                if not reject_bad and i == self.n_update_iter:
                    break
                act_seqs = self.sample_action_sequences(act_seq)
                self.noise_level *= getattr(self, "noise_decay", 1)
                # assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
                rewards, model_out, eval_out = self.calculate_reward(state_cur, act_seqs)
                if reject_bad:
                    if i > 0:
                        bad_update_indices = rewards < rewards_prev
                        # print(f"reject {bad_update_indices.sum().item()} bad updates from {rewards.shape[0]} in iteration {i}")
                        act_seqs.data[bad_update_indices] = act_seqs_prev[bad_update_indices]
                        rewards.data[bad_update_indices] = rewards_prev[bad_update_indices]
                        model_out["state_seqs"].data[bad_update_indices] = model_out_prev["state_seqs"][bad_update_indices]
                        eval_out["rewards"].data[bad_update_indices] = eval_out_prev["rewards"][bad_update_indices]
                    if i == self.n_update_iter:
                        break
                    rewards_prev = rewards.clone()
                    act_seqs_prev = act_seqs.clone()
                    model_out_prev = model_out
                    eval_out_prev = eval_out
                
                reward_list.append(rewards.max())
                if rewards.max() > best_reward:
                    best_id = torch.argmax(rewards)
                    best_act_seq = act_seqs[best_id].clone()
                    best_reward = rewards.max()
                
                # [n_sample, n_look_ahead, action_dim], [n_sample] -> [n_look_ahead, action_dim]
                act_seq = self.optimize_action(act_seqs, rewards)
                if self.verbose and i == self.n_update_iter - 1:
                    model_outputs.append(model_out)
                    eval_outputs.append(eval_out)
                    act_seqs_lst.append(act_seqs)
                # # print
                # best_model_out = self.model_rollout(state_cur, act_seq.unsqueeze(0))
                # best_eval_out = self.evaluate_traj(best_model_out["state_seqs"], act_seq.unsqueeze(0))
                # print(f"iter {i}, action: {act_seq}, cost: {-best_eval_out['rewards'].item()}")        if self.rollout_best:
            
            act_seq = best_act_seq
            # with torch.no_grad():
            # [n_his, state_dim], [1, n_look_ahead, action_dim] -> [1, n_look_ahead, state_dim]
            best_model_out = self.model_rollout(state_cur, act_seq.unsqueeze(0))
            # [1, n_look_ahead, state_dim], [1, n_look_ahead, action_dim] -> [1]
            best_eval_out = self.evaluate_traj(best_model_out["state_seqs"], act_seq.unsqueeze(0))
            # if best_eval_out["rewards"] < init_reward:
            #     act_seq = init_act_seq
            #     best_eval_out = init_eval_out
            #     best_model_out = init_model_out
            # print(f"mppi cost: {-best_eval_out['rewards'].item()}")
        return {
            "act_seq": act_seq,
            "reward_list": torch.stack(reward_list).view(-1),
            "model_outputs": model_outputs if self.verbose else None,
            "eval_outputs": eval_outputs if self.verbose else None,
            "act_seqs_lst": act_seqs_lst if self.verbose else None,
            "best_model_output": best_model_out if self.rollout_best else None,
            "best_eval_output": best_eval_out if self.rollout_best else None,
        }
        
    def trajectory_optimization_cem(self, state_cur, act_seq, init_pusher_pos=None):
        reset_seed(self.seed)
        if self.verbose:
            model_outputs = []
            eval_outputs = []
            act_seqs_lst = []

        horizon = act_seq.shape[0]
        action_dim = act_seq.shape[1]
        n_params = horizon * action_dim
        n_elite = max(int(self.elite_ratio * self.n_sample), self.min_n_elites)
        sigma = (self.action_upper_lim - self.action_lower_lim) / 2

        # mean = torch.zeros(n_params, device=self.device)
        mean = ((self.action_upper_lim + self.action_lower_lim) / 2).repeat(horizon)
        cov = torch.eye(n_params, device=self.device)
        alpha = 0
        var = torch.diag(cov)

        lower_lim = self.action_lower_lim.view(1, 1, -1)
        upper_lim = self.action_upper_lim.view(1, 1, -1)

        for i, sig in enumerate(sigma):
            cov[i::action_dim] *= sig
        # Small jitter to ensure positive definiteness
        jitter = self.jitter_factor
        cov += jitter * torch.eye(n_params, device=self.device)
        with torch.no_grad():
            for i in range(self.n_update_iter):
                act_seqs = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample((self.n_sample,))
                act_seqs = act_seqs.view(self.n_sample, horizon, action_dim).to(self.device)
                act_seqs = act_seqs.clamp(min=lower_lim, max=upper_lim)
                rewards, model_out, eval_out = self.calculate_reward(state_cur, act_seqs)

                elite_idxs = torch.argsort(rewards, descending=True)[:n_elite]
                elite_action_sequences = act_seqs[elite_idxs]

                if alpha > 0 and alpha < 1:
                    new_mean = elite_action_sequences.view(n_elite, -1).mean(dim=0)
                    new_var = torch.var(elite_action_sequences.view(n_elite, -1), dim=0)

                    # Update with momentum
                    mean = alpha * mean + (1 - alpha) * new_mean
                    var = alpha * var + (1 - alpha) * new_var
                    cov = torch.diag(var)
                else:
                    mean = elite_action_sequences.view(n_elite, -1).mean(dim=0)
                    cov = torch.cov(elite_action_sequences.view(n_elite, -1).T)

                # Ensure positive definiteness after updating covariance matrix
                cov += jitter * torch.eye(n_params, device=self.device)
                if self.verbose and i == self.n_update_iter - 1:
                    model_outputs.append(model_out)
                    eval_outputs.append(eval_out)
                    act_seqs_lst.append(act_seqs)
            
            act_seq = elite_action_sequences[0].to(self.device)
            best_model_out = self.model_rollout(state_cur, act_seq.unsqueeze(0))
            best_eval_out = self.evaluate_traj(best_model_out["state_seqs"], act_seq.unsqueeze(0))
        return {
            "act_seq": act_seq,
            "model_outputs": model_outputs if self.verbose else None,
            "eval_outputs": eval_outputs if self.verbose else None,
            "act_seqs_lst": act_seqs_lst if self.verbose else None,
            "best_model_output": best_model_out if self.rollout_best else None,
            "best_eval_output": best_eval_out if self.rollout_best else None,
        }

    def trajectory_optimization_decent_cem(self, state_cur, act_seq, init_pusher_pos=None):
        reset_seed(self.seed)
        if self.verbose:
            model_outputs = []
            eval_outputs = []
            act_seqs_lst = []
        
        horizon = act_seq.shape[0]
        action_dim = act_seq.shape[1]
        n_params = horizon * action_dim
        n_elite = max(int(self.elite_ratio * self.n_sample), self.min_n_elites)
        sigma = (self.action_upper_lim - self.action_lower_lim) / 2
        n_agents = self.n_agents
        means = [((self.action_upper_lim + self.action_lower_lim) / 2).repeat(horizon) for _ in range(n_agents)]
        covs = [torch.eye(n_params, device=self.device) for _ in range(n_agents)]

        lower_lim = self.action_lower_lim.view(1, 1, -1)
        upper_lim = self.action_upper_lim.view(1, 1, -1)

        # Initialize covariance matrices for each agent
        for j in range(n_agents):
            for i, sig in enumerate(sigma):
                covs[j][i::action_dim] *= sig
        
        # Small jitter to ensure positive definiteness
        jitter = self.jitter_factor
        global_best_action_sequence = None
        global_best_fitness = -float('inf')
        global_fitness = []

        with torch.no_grad():
            for i in range(self.n_update_iter):
                agent_best_action_sequences = []
                agent_best_fitnesses = []
                
                for j in range(n_agents):
                    mean = means[j]
                    cov = covs[j] + jitter * torch.eye(n_params, device=self.device)  # Ensure positive definiteness
                    
                    act_seqs = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).sample((self.n_sample,))
                    act_seqs = act_seqs.view(self.n_sample, horizon, action_dim).to(self.device)
                    act_seqs = act_seqs.clamp(min=lower_lim, max=upper_lim)
                    rewards, model_out, eval_out = self.calculate_reward(state_cur, act_seqs)
                    
                    elite_idxs = torch.argsort(rewards, descending=True)[:n_elite]
                    elite_action_sequences = act_seqs[elite_idxs]
                    
                    means[j] = elite_action_sequences.view(n_elite, -1).mean(dim=0)
                    covs[j] = torch.cov(elite_action_sequences.view(n_elite, -1).T)
                    
                    # Track the best action sequence of this agent
                    best_agent_fitness = rewards.max()
                    best_agent_action_sequence = act_seqs[rewards.argmax()]
                    
                    agent_best_fitnesses.append(best_agent_fitness)
                    agent_best_action_sequences.append(best_agent_action_sequence)
                
                global_fitness.append(torch.stack(agent_best_fitnesses))

                # Share the best samples among agents to form the global best
                best_agent_idx = torch.argmax(torch.tensor(agent_best_fitnesses))
                if agent_best_fitnesses[best_agent_idx] > global_best_fitness:
                    global_best_fitness = agent_best_fitnesses[best_agent_idx]
                    global_best_action_sequence = agent_best_action_sequences[best_agent_idx]
                
                if self.verbose and i == self.n_update_iter - 1 and j == n_agents - 1:
                    model_outputs.append(model_out)
                    eval_outputs.append(eval_out)
                    act_seqs_lst.append(act_seqs)
            
            act_seq = global_best_action_sequence.to(self.device)
            best_model_out = self.model_rollout(state_cur, act_seq.unsqueeze(0))
            best_eval_out = self.evaluate_traj(best_model_out["state_seqs"], act_seq.unsqueeze(0))
        
        # [n_agents, n_update_iter]
        global_rewards = torch.stack(global_fitness, dim=0).view(self.n_update_iter, n_agents).T

        if self.save_planning_result:
            report_stat(global_rewards, os.path.join(self.experiment_folder, "decentCEM.txt"), self.report_interval)
        
        return {
            "act_seq": act_seq,
            "model_outputs": model_outputs if self.verbose else None,
            "eval_outputs": eval_outputs if self.verbose else None,
            "act_seqs_lst": act_seqs_lst if self.verbose else None,
            "best_model_output": best_model_out if self.rollout_best else None,
            "best_eval_output": best_eval_out if self.rollout_best else None,
        }

    def trajectory_optimization_gd(self, state_cur, act_seq):
        reset_seed(self.seed)
        if self.verbose:
            model_outputs = []
            eval_outputs = []
            act_seqs_lst = []
        reward_list = []
        best_reward = torch.tensor(-float('inf'), device=self.device)
        best_act_seq = act_seq
        reject_bad = getattr(self, "reject_bad", False)
        rewards_prev = None
        act_seqs_prev = None
        model_out_prev = None
        eval_out_prev = None

        act_seqs = self.sample_action_sequences(act_seq).requires_grad_()  # (n_sample, n_look_ahead, action_dim)
        optimizer = torch.optim.Adam([act_seqs], lr=self.lr, betas=(0.9, 0.999))
        for i in range(self.n_update_iter):
            # assert act_seqs.shape == (self.n_sample, self.n_look_ahead, self.action_dim)
            rewards, model_out, eval_out = self.calculate_reward(state_cur, act_seqs)
            # print(f"iter {i}, cost: {-rewards.max()}, {-rewards.median()}, {-rewards.min()}")

            self.lr *= getattr(self, "lr_decay", 1)
            # print(f"iter {i}, lr: {self.lr}")
            if reject_bad:
                if i > 0:
                    bad_update_indices = rewards < rewards_prev
                    # print(f"reject {bad_update_indices.sum().item()} bad updates from {rewards.shape[0]} in iteration {i}")
                    act_seqs.data[bad_update_indices] = act_seqs_prev[bad_update_indices]
                    rewards.data[bad_update_indices] = rewards_prev[bad_update_indices]
                    model_out["state_seqs"].data[bad_update_indices] = model_out_prev["state_seqs"][bad_update_indices]
                    eval_out["rewards"].data[bad_update_indices] = eval_out_prev["rewards"][bad_update_indices]
                if i == self.n_update_iter:
                    break
                rewards_prev = rewards.clone()
                act_seqs_prev = act_seqs.clone()
                model_out_prev = model_out
                eval_out_prev = eval_out

            reward_list.append(rewards.max().detach())
            if rewards.max() > best_reward:
                best_id = torch.argmax(rewards)
                best_act_seq = act_seqs[best_id].clone()
                best_reward = rewards.max().detach()

            # [n_sample, n_look_ahead, action_dim], [n_sample] -> update act_seqs
            self.optimize_action(act_seqs, rewards, optimizer)
            self.clip_actions(act_seqs)
            if self.verbose and i == self.n_update_iter - 1:
                model_outputs.append(model_out)
                eval_outputs.append(eval_out)
                act_seqs_lst.append(act_seqs)
        # [n_look_ahead, action_dim]
        # act_seq = act_seqs[torch.argmax(rewards)]
        act_seq = best_act_seq
        with torch.no_grad():
            # [n_his, state_dim], [1, n_look_ahead, action_dim] -> [1, n_look_ahead, state_dim]
            best_model_out = self.model_rollout(state_cur, act_seq.unsqueeze(0))
            # [1, n_look_ahead, state_dim], [1, n_look_ahead, action_dim] -> [1]
            best_eval_out = self.evaluate_traj(best_model_out["state_seqs"], act_seq.unsqueeze(0))

        del act_seqs, optimizer, rewards

        return {
            "act_seq": act_seq,
            "reward_list": torch.stack(reward_list).view(-1),
            "model_outputs": model_outputs if self.verbose else None,
            "eval_outputs": eval_outputs if self.verbose else None,
            "act_seqs_lst": act_seqs_lst if self.verbose else None,
            "best_model_output": best_model_out if self.rollout_best else None,
            "best_eval_output": best_eval_out if self.rollout_best else None,
        }

    def trajectory_optimization_gd_bf(self, state_cur, act_seq):
        base_lr = self.lr
        # lr_list = np.array([0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 5, 10, 20]) * base_lr
        lr_list = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 5, 10]) * base_lr
        best_reward = -np.inf
        best_result = None
        best_lr = None
        global_reward_list = []

        for _ in range(self.n_restart_round):
            for lr in lr_list:
                self.lr = lr
                result = self.trajectory_optimization_gd(state_cur.clone(), act_seq.clone())
                torch.cuda.empty_cache()
                global_reward_list.append(result["reward_list"])
                self.seed += 1

                if result["best_eval_output"]["rewards"] > best_reward:
                    best_result = result
                    best_reward = result["best_eval_output"]["rewards"]
                    best_lr = lr
                else:
                    del result

        global_rewards = torch.stack(global_reward_list, dim=0)

        if self.save_planning_result:
            report_stat(global_rewards, os.path.join(self.experiment_folder, "gd_bf.txt"), self.report_interval)

        return best_result


    def trajectory_optimization_mppi_gd(self, state_cur, act_seq=None):
        if self.verbose:
            model_outputs = []
            eval_outputs = []
            act_seqs_lst = []
        for i in range(self.n_update_iter):
            act_seqs = self.sample_action_sequences(act_seq.clone().detach()).requires_grad_() # (n_sample, n_look_ahead, action_dim)
            self.optimize_action_mppi_gd(act_seqs, state_cur)
            with torch.no_grad():
                rewards, model_out, eval_out = self.calculate_reward(state_cur, act_seqs)
                # [n_sample, n_look_ahead, action_dim], [n_sample] -> [n_look_ahead, action_dim]
                act_seq = self.optimize_action_mppi(act_seqs, rewards)
                if self.verbose and i == self.n_update_iter - 1:
                    model_outputs.append(model_out)
                    eval_outputs.append(eval_out)
                    act_seqs_lst.append(act_seqs)
        act_seq = act_seq.unsqueeze(0).requires_grad_()
        self.optimize_action_mppi_gd(act_seq, state_cur)

        if self.rollout_best:
            # [n_his, state_dim], [1, n_look_ahead, action_dim] -> [1, n_look_ahead, state_dim]
            best_model_out = self.model_rollout(state_cur, act_seq)
            # [1, n_look_ahead, state_dim], [1, n_look_ahead, action_dim] -> [1]
            best_eval_out = self.evaluate_traj(best_model_out["state_seqs"], act_seq)

        return {
            "act_seq": act_seq.squeeze(0),
            "model_outputs": model_outputs if self.verbose else None,
            "eval_outputs": eval_outputs if self.verbose else None,
            "act_seqs_lst": act_seqs_lst if self.verbose else None,
            "best_model_output": best_model_out if self.rollout_best else None,
            "best_eval_output": best_eval_out if self.rollout_best else None,
        }

    def calculate_reward(self, state_cur, act_seqs):
        # [n_his, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample, n_look_ahead, state_dim]
        model_out = self.model_rollout(state_cur, act_seqs)
        state_seqs = model_out["state_seqs"]
        
        # [n_sample, n_look_ahead, state_dim], [n_sample, n_look_ahead, action_dim] -> [n_sample]
        eval_out = self.evaluate_traj(state_seqs, act_seqs)
        
        rewards = eval_out["rewards"] # (n_sample)
        return rewards, model_out, eval_out
    
def report_stat(global_rewards: torch.Tensor, output_file, report_interval = 10):
    # global_rewards: [n_restarts, n_update_iter]
    n_restarts, n_update_iter = global_rewards.shape
    # assume output_file is a txt file
    assert output_file.endswith(".txt")

    global_cost = -global_rewards

    with open(output_file, "a") as f:
        print("--------summary--------", file=f)
        iter_fitness = global_cost.min(axis=0)[0]
        for iter_id in range(report_interval, n_update_iter + 1, report_interval):
            best_cost = iter_fitness[:iter_id].min()
            print(f"n_iter:{iter_id}, best cost: {best_cost.item():<8.5f}", file=f)

        agent_fitness = global_cost.min(axis=1)[0]
        for restart_id in range(report_interval, n_restarts + 1, report_interval):
            best_cost = agent_fitness[:restart_id].min()
            print(f"n_restarts:{restart_id}, best cost: {best_cost.item():<8.5f}", file=f)

    with open(output_file.replace(".txt", "_full.txt"), "a") as f:
        print("--------full--------", file=f)
        for i in range(n_restarts):
            print(f"restart {i}: {global_cost[i].tolist()}", file=f)
