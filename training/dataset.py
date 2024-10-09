import pickle
import os, sys
import numpy as np
from torch.utils.data import Dataset
sys.path.append(os.getcwd())
from others.helper import rotate_state, reset_seed

class DynamicsDataset(Dataset):
    def __init__(self, config, phase):
        self.config = config
        # dynamics data expected to be a single pickle file
        # should be a list of numpy matrices
        # each matrix represents an episode in step order
        # and is of shape (# steps, state_dim + action_dim) representing (s, a)
        data_file_name = config["train"].get("data_path", None)
        if data_file_name is None:
            data_file_name = os.path.join(config["data_root"], config["task_name"], "data.p")
        with open(data_file_name, "rb") as fp:
            data_load = pickle.load(fp)

        data_config = config["data"]
        train_config = config["train"]
        state_dim = data_config["state_dim"]
        action_dim = data_config["action_dim"]
        self.phase = phase
        n_his = train_config["n_history"]
        n_roll = train_config["n_rollout"]
        if phase == "valid":
            n_roll = (
                train_config["n_rollout_valid"] if "n_rollout_valid" in train_config else n_roll * 2
            )
        n_sample = n_his + n_roll

        num_train = int(len(data_load) * train_config["train_valid_ratio"])
        if phase == "train":
            data_load = data_load[:num_train]
        elif phase == "valid":
            data_load = data_load[num_train:]
        else:
            raise AssertionError("Unknown phase %s" % phase)
        n_roll = min(data_load[0].shape[0]-n_his, n_roll)
        n_sample = n_his + n_roll
        self.n_roll = n_roll
        self.n_sample = n_sample
        self.n_his = n_his
        self.episode_length = len(data_load[0]) #  - 1
        self.obs = []
        self.act = []
        self.weights = []
        # import pdb; pdb.set_trace()
        for ep in data_load:
            for i in range(len(ep) - n_sample + 1):
                self.obs.append(ep[i : i + n_sample, :-action_dim])
                self.act.append(ep[i : i + n_sample, -action_dim:])
        self.obs = np.array(self.obs)
        self.act = np.array(self.act)
        # if data_config["augment"] and phase == "train":
        #     self.obs = rotate_state(self.obs, config["seed"])
        #     self.act = rotate_state(self.act, config["seed"])
        self.weights = np.ones((self.obs.shape[0], n_roll))
        print('weights percentile', [round(np.percentile(self.weights, 5*i),5) for i in range(21)])

        print(f"state shape {self.obs.shape}")
        # print percentile for every dim of state
        # for j in range(state_dim):
        #     print(f"state dim {j} percentile", [round(np.percentile(self.obs[:, :, j], 5 * i), 5) for i in range(21)])

        print("x vel percentile", [round(np.percentile(self.act[:, :, -2], 5 * i), 5) for i in range(21)])
        print("y vel percentile", [round(np.percentile(self.act[:, :, -1], 5 * i), 5) for i in range(21)])

        if train_config["include_com"]:
            pusher_pos_idx = state_dim+2 * config["data"]["obj_num"]
        else:
            pusher_pos_idx = state_dim
        self.pusher_pos = self.obs[:, :, pusher_pos_idx : pusher_pos_idx + 2]
        self.other = self.obs[:, :, pusher_pos_idx+2:]
        self.obs = self.obs[:, :, :state_dim]
        # add noise
        if phase == "train":
            eps = train_config["noise"] if "noise" in train_config else 0.003
            np.random.seed(config["seed"])
            noise = np.random.normal(0, eps, size=self.obs.shape)
            self.obs = self.obs + noise
        
        # # sdf use relative position to pusher
        # Nsamples, Nsteps, Ndim = self.obs.shape
        # self.obs = self.obs.reshape(Nsamples, Nsteps, Ndim//2, 2)
        # self.obs = self.obs - self.pusher_pos[:, :, np.newaxis, :]
        # self.obs = self.obs.reshape(Nsamples, Nsteps, Ndim)
        # print(f"obs shape: {self.obs.shape}")
        # print(f"act shape: {self.act.shape}")
        # print(f"pusher_shape: {self.pusher_pos.shape}")

        # shuffle together
        if phase == "train" and data_config["augment"]:
            self.augment()
        else:
            self.shuffle()  

    def augment(self):
        seed = np.random.randint(0, 100000)
        self.obs = rotate_state(self.obs, seed)
        self.act = rotate_state(self.act, seed)
        self.pusher_pos = rotate_state(self.pusher_pos, seed)
        if self.other.size != 0:
            self.other = rotate_state(self.other, seed)
        reset_seed(self.config["seed"])
        self.shuffle()

    def shuffle(self):
        # np.random.seed(self.config["seed"])
        idx = np.random.permutation(range(len(self.obs)))
        # np.random.seed(self.config["seed"])
        self.obs = self.obs[idx]
        self.act = self.act[idx]
        self.weights = self.weights[idx]
        self.pusher_pos = self.pusher_pos[idx]
        self.other = self.other[idx]

    # only called when data.enable_hnm is True
    def update_weights(self, indices, new_weight):
        if len(indices.shape) == 1:
            self.weights[indices] *= new_weight.reshape(len(new_weight), 1)
            self.weights[indices] = np.clip(self.weights[indices], 1, self.config["data"]["weight_ub"])
        elif len(indices.shape) == 2:
            self.weights[indices[:, 0], indices[:, 1]] *= new_weight
            self.weights[indices[:, 0], indices[:, 1]] = np.clip(
                self.weights[indices[:, 0], indices[:, 1], 0], 1, self.config["data"]["weight_ub"]
            )
        else:
            raise AssertionError("Unknown indices shape")
        print("weights percentile", [round(np.percentile(self.weights, 5 * i), 5) for i in range(21)])
        self.shuffle()


    def add_online_data(self, data_load):
        state_dim = self.config["data"]["state_dim"]
        action_dim = self.config["data"]["action_dim"]
        n_sample = self.n_sample
        n_his = self.n_his
        n_roll = self.n_roll
        self.new_obs = []
        self.new_act = []
        self.new_weights = []
        # import pdb; pdb.set_trace()
        for ep in data_load:
            for i in range(len(ep) - n_sample + 1):
                self.new_obs.append(ep[i : i + n_sample, :-action_dim])
                self.new_act.append(ep[i : i + n_sample, -action_dim:])
                self.new_weights.append(
                    np.square(ep[i + n_his : i + n_sample, :state_dim] - ep[i + n_his - 1, :state_dim]).mean(
                        axis=1, keepdims=True
                    )
                )
        self.new_obs = np.array(self.new_obs)
        self.new_act = np.array(self.new_act)
        self.new_weights = np.ones((self.new_obs.shape[0], n_roll))
        print('weights percentile', [round(np.percentile(self.new_weights, 5*i),5) for i in range(21)])
        
        # hard code pusher position in 2D
        self.new_pusher_pos = self.new_obs[:, :, state_dim : state_dim + 2]
        # should be empty for now
        self.new_other = self.new_obs[:, :, state_dim+2:]
        self.new_obs = self.new_obs[:, :, :state_dim]
        
        # add noise
        if self.phase == "train":
            eps = self.config["train"]["noise"] if "noise" in self.config["train"] else 0.003
            # np.random.seed(self.config["seed"])
            noise = np.random.normal(0, eps, size=self.new_obs.shape)
            self.new_obs = self.new_obs + noise
        self.obs = np.concatenate((self.obs, self.new_obs))
        self.act = np.concatenate((self.act, self.new_act))
        self.weights = np.concatenate((self.weights, self.new_weights))
        self.pusher_pos = np.concatenate((self.pusher_pos, self.new_pusher_pos))
        self.other = np.concatenate((self.other, self.new_other))
        self.shuffle()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if self.phase == "valid" or self.config["task_name"] != 'obj_pile':
            return {
            "observations": self.obs[idx],
            "actions": self.act[idx],
            "weights": self.weights[idx],
            "pusher_pos": self.pusher_pos[idx],
            "others": self.other[idx],
        }
        theta = np.random.rand() * 2 * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        steps_lookahead, obs_dim = self.obs[idx].shape
        obs = self.obs[idx].reshape(steps_lookahead, obs_dim//2, 2)
        # rotate the data
        obs = np.matmul(obs, rot)
        
        obs = obs.reshape(steps_lookahead, obs_dim)
        
        action = np.concatenate((np.matmul(self.act[idx][:, :2], rot), \
            np.matmul(self.act[idx][:, 2:], rot)), axis=-1)
        pusher_pos = np.matmul(self.pusher_pos[idx], rot)
        
        return {
            "observations": obs,
            "actions": action,
            "weights": self.weights[idx],
            "pusher_pos": pusher_pos,
            "others": self.other[idx],
        }

class ImageDataset(Dataset):
    def __init__(self, config, phase):
        self.config = config
        # encoder image data expected to be a single pickle file
        # should be a list(num_ep) of array(len_ep) of dict of image state (n,n,3) and action(2)
        data_file_name = os.path.join(config["data_root"], config["task_name"], "data_img.p")
        with open(data_file_name, "rb") as fp:
            data_load = pickle.load(fp)

        data_config = config["data"]
        train_config = config["train"]

        num_train = int(len(data_load) * train_config["train_valid_ratio"])
        if phase == "train":
            data_load = data_load[:num_train]
        elif phase == "valid":
            data_load = data_load[num_train:]
        else:
            raise AssertionError("Unknown phase %s" % phase)

        self.obs = []
        self.explicit_states = []
        for ep in data_load:
            for i in range(len(ep)):
                self.obs.append(ep[i]["image"])
                self.explicit_states.append(np.concatenate([ep[i]['state'], ep[i]['pusher_pos']]))
        
        self.obs = np.array(self.obs)
        self.explicit_states = np.array(self.explicit_states)
        # import pdb; pdb.set_trace()
        self.explicit_states = self.explicit_states / 100
        self.episode_length = len(ep)
        
        # for pytorch, the channel should be the second dimension
        self.obs = np.transpose(self.obs, (0 ,3, 1, 2))
        self.weights = np.ones((self.obs.shape[0], 1))

        self.shuffle()

    def augment(self):
        pass
        # seed = np.random.randint(0, 100000)
        # reset_seed(self.config["seed"])
        # self.shuffle()

    def shuffle(self):
        pass
        # np.random.seed(self.config["seed"])
        # idx = np.random.permutation(range(len(self.obs)))
        # np.random.seed(self.config["seed"])
        # self.obs = self.obs[idx]
        # self.explicit_states = self.explicit_states[idx]

    # only called when data.enable_hnm is True
    def update_weights(self, indices, new_weight):
        pass
        # if len(indices.shape) == 1:
        #     self.weights[indices] *= new_weight.reshape(len(new_weight), 1)
        #     self.weights[indices] = np.clip(self.weights[indices], 1, self.config["data"]["weight_ub"])
        # elif len(indices.shape) == 2:
        #     self.weights[indices[:, 0], indices[:, 1]] *= new_weight
        #     self.weights[indices[:, 0], indices[:, 1]] = np.clip(
        #         self.weights[indices[:, 0], indices[:, 1], 0], 1, self.config["data"]["weight_ub"]
        #     )
        # else:
        #     raise AssertionError("Unknown indices shape")
        # print("weights percentile", [round(np.percentile(self.weights, 5 * i), 5) for i in range(21)])
        # self.shuffle()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        episode_idx = idx // self.episode_length
        step = idx % self.episode_length
        other_idx = np.random.choice(list(range(self.episode_length)))
        
        if other_idx == step:
            other_idx = max(self.episode_length-1, step+1)
        if other_idx == step:
            other_idx = min(0, step-1)
        
        return {
            "observations": self.obs[idx],
            "positive": self.obs[other_idx + episode_idx * self.episode_length],
            "explicit_state": self.explicit_states[idx]
        }

class DynamicsDataset_IMG(Dataset):
    def __init__(self, config, phase):
        self.config = config
        data_file_name = os.path.join(config["data_root"], config["task_name"], "data_latent.p")
        with open(data_file_name, "rb") as fp:
            data_load = pickle.load(fp)

        data_config = config["data"]
        train_config = config["train"]
        latent_config = config["latent"]

        latent_dim = latent_config["latent_dim"]
        num_episodes = data_config["num_episodes"]
        episode_length = data_config["episode_length"]
        n_his = train_config["n_history"]
        n_roll = train_config["n_rollout"]
        
        if phase == "valid":
            n_roll = (
                train_config["n_rollout_valid"] if "n_rollout_valid" in train_config else n_roll * 2
            )
        n_sample = n_his + n_roll

        latent_states = data_load["latent_states"]
        
        # assert latent_states.shape[0] == num_episodes and latent_states.shape[1] == episode_length
        num_episodes = latent_states.shape[0]
        actions = data_load["actions"]
        states = data_load["states"]
        pusher_pos = data_load["pusher_pos"]

        num_train = int(num_episodes * train_config["train_valid_ratio"])
        if phase == "train":
            latent_states = latent_states[:num_train]
            actions = actions[:num_train]
            states = states[:num_train]
            pusher_pos = pusher_pos[:num_train]
        elif phase == "valid":
            latent_states = latent_states[num_train:]
            actions = actions[num_train:]
            states = states[num_train:]
            pusher_pos = pusher_pos[num_train:]
        else:
            raise AssertionError("Unknown phase %s" % phase)

        self.n_his = n_his
        self.n_roll = n_roll
        self.n_sample = n_sample
        self.phase = phase
        # import pdb; pdb.set_trace()
        self.episode_length = len(states[0])
        self.latent_state = []
        self.act = []
        self.state = []
        self.pusher_pos = []
        self.weights = []

        for ep in range(len(latent_states)):
            for step in range(episode_length - n_sample + 1):
                self.latent_state.append(latent_states[ep][step : step + n_sample])
                self.act.append(actions[ep][step : step + n_sample])
                self.state.append(states[ep][step : step + n_sample])
                self.pusher_pos.append(pusher_pos[ep][step : step + n_sample])

        self.latent_state = np.array(self.latent_state)
        self.act = np.array(self.act)
        self.state = np.array(self.state)
        self.pusher_pos = np.array(self.pusher_pos)
        self.weights = np.ones((self.latent_state.shape[0], n_roll))
        print('weights percentile', [round(np.percentile(self.weights, 5*i),5) for i in range(21)])

        print(f"state shape {self.latent_state.shape}")
        # print percentile for every dim of state
        # for j in range(latent_dim):
        #     print(f"state dim {j} percentile", [round(np.percentile(self.latent_state[:, :, j], 5 * i), 5) for i in range(21)])

        print("x vel percentile", [round(np.percentile(self.act[:, :, -2], 5 * i), 5) for i in range(21)])
        print("y vel percentile", [round(np.percentile(self.act[:, :, -1], 5 * i), 5) for i in range(21)])

        # add noise
        if phase == "train" and train_config["noise"] > 0:
            np.random.seed(config["seed"])
            noise = np.random.normal(0, train_config["noise"], size=self.latent_state.shape)
            self.latent_state = self.latent_state + noise

        self.shuffle()

    def augment(self):
        pass
    
    def shuffle(self):
        np.random.seed(self.config["seed"])
        idx = np.random.permutation(range(len(self.latent_state)))
        np.random.seed(self.config["seed"])
        self.latent_state = self.latent_state[idx]
        self.act = self.act[idx]
        self.state = self.state[idx]
        self.pusher_pos = self.pusher_pos[idx]
        self.weights = self.weights[idx]

    # only called when data.enable_hnm is True
    def update_weights(self, indices, new_weight):
        if len(indices.shape) == 1:
            self.weights[indices] *= new_weight.reshape(len(new_weight), 1)
            self.weights[indices] = np.clip(self.weights[indices], 1, self.config["data"]["weight_ub"])
        elif len(indices.shape) == 2:
            self.weights[indices[:, 0], indices[:, 1]] *= new_weight
            self.weights[indices[:, 0], indices[:, 1]] = np.clip(
                self.weights[indices[:, 0], indices[:, 1], 0], 1, self.config["data"]["weight_ub"]
            )
        else:
            raise AssertionError("Unknown indices shape")
        print("weights percentile", [round(np.percentile(self.weights, 5 * i), 5) for i in range(21)])
        self.shuffle()

    def __len__(self):
        return len(self.latent_state)

    def __getitem__(self, idx):
        if self.phase == 'train':
            latent = self.latent_state[idx] + np.random.normal(0, 0.003, size=self.latent_state[idx].shape)
        else:
            latent = self.latent_state[idx]
        return {
            "observations": latent,
            "actions": self.act[idx],
            "weights": self.weights[idx],
            "pusher_pos": self.pusher_pos[idx],
            "states": self.state[idx],
        }

class ContrastiveImageDataset(Dataset):
    def __init__(self, config, phase):
        self.config = config
        # encoder image data expected to be a single pickle file
        # should be a list(num_ep) of array(len_ep) of dict of image state (n,n,3) and action(2)
        data_file_name = os.path.join(config["data_root"], config["task_name"], "data_img.p")
        with open(data_file_name, "rb") as fp:
            data_load = pickle.load(fp)

        data_config = config["data"]
        train_config = config["train"]

        num_train = int(len(data_load) * train_config["train_valid_ratio"])
        if phase == "train":
            data_load = data_load[:num_train]
        elif phase == "valid":
            data_load = data_load[num_train:]
        else:
            raise AssertionError("Unknown phase %s" % phase)

        self.obs = []
        self.explicit_states = []
        for ep in data_load:
            ep_obs = []
            ep_explicit = []
            for i in range(len(ep)):
                ep_obs.append(ep[i]["image"])
                ep_explicit.append(np.concatenate([ep[i]['state'], ep[i]['pusher_pos']]))
            self.obs.append(ep_obs)
            self.explicit_states.append(ep_explicit)
        
        self.obs = np.array(self.obs)
        self.explicit_states = np.array(self.explicit_states)
        # import pdb; pdb.set_trace()
        self.explicit_states = self.explicit_states / 100
        
        
        # for pytorch, the channel should be the second dimension
        self.obs = np.transpose(self.obs, (0, 1, 4, 2, 3))
        self.weights = np.ones((self.obs.shape[0], 1))

        self.shuffle()

    def augment(self):
        pass
        # seed = np.random.randint(0, 100000)
        # reset_seed(self.config["seed"])
        # self.shuffle()

    def shuffle(self):
        pass

    # only called when data.enable_hnm is True
    def update_weights(self, indices, new_weight):
        pass

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        num_obs = len(self.obs)
        epi_len = self.obs.shape[1]
        
        sub_idx1 = np.random.choice(list(range(epi_len)))
        sub_idx2 = np.random.choice(list(range(epi_len)))
        neg_idx = np.random.choice(list(range(num_obs)))
        if idx == neg_idx:
            neg_idx = max(num_obs-1, neg_idx+1)
        if idx == neg_idx:
            neg_idx = min(0, neg_idx-1)
            
        return {
            "observations": self.obs[idx][sub_idx1],
            "positive": self.obs[idx][sub_idx2],
            "negative": self.obs[neg_idx][sub_idx1],
            "explicit_state": self.explicit_states[idx]
        }

# if __name__ == "__main__":
#     data_file_name = f"{os.getcwd()}/data/box_pushing/data_img.p"
#     with open(data_file_name, "rb") as fp:
#         data_load = pickle.load(fp)

#     print("load")

