from tasks.pushing_rope.flex_env import FlexEnv
import torch
import numpy as np
from scipy.spatial.distance import cdist
from dgl.geometry import farthest_point_sampler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class RopeSim(FlexEnv):
    def __init__(self, param_dict, init_poses=None, target_poses=None, pusher_pos=None):
        super().__init__({'dataset': param_dict})
        # redundent definitions
        self.classes = None
        self.label_list = None
        max_nobj = param_dict['max_nobj']
        state_dim = param_dict['state_dim']
        action_dim = param_dict['action_dim']
        self.dim_of_work_space = state_dim // max_nobj
        assert self.dim_of_work_space == action_dim // 2
        assert self.dim_of_work_space == 2 # currently only support 2D
        self.reset()
        obj_kp = self.particle_pos_list[-1]
        # farthest point sampling
        particle_tensor = torch.from_numpy(obj_kp).float()[None, ...]
        obj_kp_num = min(max_nobj, particle_tensor.shape[1])
        fps_idx_tensor = farthest_point_sampler(particle_tensor, obj_kp_num, 
                            start_idx=np.random.randint(0, obj_kp.shape[0]))[0]
        self.fps_idx = fps_idx_tensor.numpy().astype(np.int32)
        self.fps_idx = self.reorder_particles(self.particle_pos_list[-1][self.fps_idx][:, [0, 2]], self.fps_idx)
        
        # self.step([-self.wkspc_w, -self.wkspc_w, -self.wkspc_w+0.1, -self.wkspc_w+0.1])
        self.step([-3, -3, -3+0.1, -3+0.1])
        # self.step([2, self.wkspc_w, 2, -self.wkspc_w])
        # whatever push it first
        
        # self.step(u)
        
    def random_reset(self,):
        iters = 3 # np.random.choice(list(range(1,3)))
        for _ in range(iters):
            u = self.sample_action_new()
            self.step(u)
    
    def update(self, u):
        self.particle_pos_list = []
        self.eef_states_list = []
        self.step_list = []
        self.contact_list = []
        # our model predicts the eef_states, so we need to reverse the y axis manually
        # self.step takes in the [x_s, -y_s, x_e, -y_e]
        u[1] *= -1
        u[3] *= -1 # reverse y, because it is x, z, -y
        _, self.particle_pos_list, self.eef_states_list, self.step_list, self.contact_list = self.step(u)
        
    def get_current_state(self,):
        # import pdb; pdb.set_trace()
        if self.dim_of_work_space == 3:
            particle_pos = self.particle_pos_list[-2][self.fps_idx]
            # particle_pos[:, 2] *= -1
            eef_pos_final = self.eef_states_list[-2][:3] # the last state is reset 
            eef_pos_start = self.eef_states_list[0][:3]
            length = eef_pos_final - eef_pos_start
            return np.concatenate([particle_pos, [eef_pos_final], [length]])
        else:
            particle_pos = self.particle_pos_list[-2][self.fps_idx][:, [0, 2]]
            
            # particle_pos[:, 1] *= -1
            eef_pos_final = self.eef_states_list[-2][[0, 2]]
            eef_pos_start = self.eef_states_list[0][[0, 2]]
            length = eef_pos_final - eef_pos_start
            # import pdb; pdb.set_trace()
            return np.concatenate([particle_pos, [eef_pos_final], [length]])
    
    def get_pusher_start_position(self,):
        if self.dim_of_work_space == 3:
            return self.eef_states_list[0][:3] # the last state is reset
        else:
            return self.eef_states_list[0][[0, 2]]
        
    def get_pusher_position(self,):
        if self.dim_of_work_space == 3:
            return self.eef_states_list[-2][:3] # the last state is reset
        else:
            return self.eef_states_list[-2][[0, 2]]
    
    def sample_action_new(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1 # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]
        
        pos_x, pos_z = positions[:, 0], positions[:, 2]
        center_x, center_z = np.median(pos_x), np.median(pos_z)
        chosen_points = []
        for idx, (x, z) in enumerate(zip(pos_x, pos_z)):
            if np.sqrt((x-center_x)**2 + (z-center_z)**2) < 2.0:
                chosen_points.append(idx)
        # print(f'chosen points {len(chosen_points)} out of {num_points}.')
        if len(chosen_points) == 0:
            print('no chosen points')
            chosen_points = np.arange(num_points)
        
        # random choose a start point which can not be overlapped with the object
        valid = False
        for _ in range(1000):
            startpoint_pos_origin = np.random.uniform(-self.wkspc_w, self.wkspc_w, size=(1, 2))
            startpoint_pos = startpoint_pos_origin.copy()
            startpoint_pos = startpoint_pos.reshape(-1)

            # choose end points which is the expolation of the start point and obj point
            pickpoint = np.random.choice(chosen_points)
            obj_pos = positions[pickpoint, [0, 2]]
            slope = (obj_pos[1] - startpoint_pos[1]) / (obj_pos[0] - startpoint_pos[0])
            if obj_pos[0] < startpoint_pos[0]:
                # 1.0 for planning
                # (1.5, 2.0) for data collection
                x_end = obj_pos[0] - 1.5# rand_float(1.5, 1.5)
            else:
                x_end = obj_pos[0] + 1.5 #rand_float(0.5, 1.5)
            y_end = slope * (x_end - startpoint_pos[0]) + startpoint_pos[1]
            
            endpoint_pos = np.array([x_end, y_end])
            # add max pushing length
            
            if obj_pos[0] != startpoint_pos[0] and np.abs(x_end) < 1.5 and np.abs(y_end) < 1.5 \
                and np.min(cdist(startpoint_pos_origin, pos_xz)) > 0.2:
                valid = True
                break
        
        if valid:
            action = np.concatenate([startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0)
        else:
            action = None
        
        return action
    
    def reorder_particles(self, particle_pos, fps_idx):
        # reorder the particles based on the distance to the end effector
        particle_pos = particle_pos.reshape(-1, 2)
        # scaler = StandardScaler()
        # particle_pos_scaled = scaler.fit_transform(particle_pos)
        particle_pos_scaled = particle_pos - np.mean(particle_pos, axis=0)
        pca = PCA(n_components=1)  # We want to reduce the data to 2 principal components
        particle_pos_pca = pca.fit_transform(particle_pos_scaled).reshape(-1)
        
        sorted_idx = np.argsort(particle_pos_pca)
        fps_idx = fps_idx[sorted_idx]
        # particle_pos_pca = particle_pos_pca[sorted_idx]
        # COLOR_LIST = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
        # for i, particle in enumerate(particle_pos_pca):
        #     plt.plot(particle, 0, 'o', color=COLOR_LIST[i], label=f"{fps_idx[i]}_{i}")
        # plt.xlim(-self.wkspc_w, self.wkspc_w)
        # plt.ylim(-self.wkspc_w, self.wkspc_w)
        # plt.legend()
        # plt.savefig('pca_particle_pos.png')
        # plt.cla()
        
        # particle_pos = particle_pos[sorted_idx]
        # COLOR_LIST = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
        # for i, particle in enumerate(particle_pos):
        #     plt.plot(particle[0], particle[1], 'o', color=COLOR_LIST[i], label=f"{fps_idx[i]}_{i}")
        # plt.xlim(-self.wkspc_w, self.wkspc_w)
        # plt.ylim(-self.wkspc_w, self.wkspc_w)
        # plt.legend()
        # plt.savefig('particle_pos.png')
        # plt.cla()
        
        # COLOR_LIST = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink']
        # for i, particle in enumerate(particle_pos_scaled):
        #     plt.plot(particle[0], particle[1], 'o', color=COLOR_LIST[i], label=f"{fps_idx[i]}_{i}")
        # plt.xlim(-self.wkspc_w, self.wkspc_w)
        # plt.ylim(-self.wkspc_w, self.wkspc_w)
        # plt.legend()
        # plt.savefig('scaled_particle_pos.png')
        # plt.cla()
        
        return fps_idx