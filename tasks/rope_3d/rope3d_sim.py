from tasks.rope_3d.flex_env_rope_3D import FlexEnvRope3D
import torch
import numpy as np
from scipy.spatial.distance import cdist
from dgl.geometry import farthest_point_sampler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random


class Rope3dSim(FlexEnvRope3D):
    def __init__(self, param_dict, init_poses=None, target_poses=None, pusher_pos=None):
        seed = param_dict['seed']
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        super().__init__({'dataset': param_dict})
        # redundent definitions
        self.classes = None
        self.label_list = None
        max_nobj = param_dict['max_nobj']
        state_dim = param_dict['state_dim']
        action_dim = param_dict['action_dim']
        self.dim_of_work_space = state_dim // max_nobj
        assert self.dim_of_work_space == action_dim
        assert self.dim_of_work_space == 3 # currently only support 2D
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.reset()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        obj_kp = self.particle_pos_list[-1]
        # farthest point sampling
        particle_tensor = torch.from_numpy(obj_kp).float()[None, ...]
        obj_kp_num = min(max_nobj, particle_tensor.shape[1])
        fps_idx_tensor = farthest_point_sampler(particle_tensor, obj_kp_num, 
                            start_idx=0)[0]
        self.fps_idx = fps_idx_tensor.numpy().astype(np.int32)
        # the first three particles are most near the grasper
        self.fps_idx = self.reorder_particles(self.particle_pos_list[-1][self.fps_idx][:, [0, 2]], self.fps_idx)
        # import pdb; pdb.set_trace()
        # self.step([-self.wkspc_w, -self.wkspc_w, -self.wkspc_w+0.1, -self.wkspc_w+0.1])
        # self.step([-3, -3, -3+0.1, -3+0.1])
        self.particle_pos_list = []
        self.eef_states_list = []
        self.step_list = []
        self.contact_list = []
        _, self.particle_pos_list, self.eef_states_list, self.step_list, self.contact_list = self.grasp_rope_end()
        
        # correct the fps_idx_list
        tool_kp_start_pos = np.mean(self.eef_states_list[-1], axis=0)[[0,2]]
        obj_kp_start_pos = self.particle_pos_list[0][self.fps_idx][:, [0,2]]
        tool_obj_dis = np.linalg.norm(tool_kp_start_pos - obj_kp_start_pos, axis=1)
        Is_reverse_fps_idx = (np.argmin(tool_obj_dis) > len(self.fps_idx) // 2)
        if Is_reverse_fps_idx:
            self.fps_idx = self.fps_idx[::-1]
        
        self.left_forbidden_area, self.right_forbidden_area, self.target_pose = None, None, None
        self.target_offest = param_dict.get('target_offest', 0.05)
        self.obs_gap = param_dict.get('obs_gap', 0.2)
        # self.step([2, self.wkspc_w, 2, -self.wkspc_w])
        # whatever push it first
        # self.step(u)
        
    def get_target_pose(self, ):
        # import pdb; pdb.set_trace()
        if self.target_pose is not None:
            return self.target_pose
        self.x_obj_radius = 0.3
        self.y_obj_radius = 0.3
        # if np.random.rand() < 0.5:
        #     target_center = np.array([self.rope_fixed_end_coord[0], np.random.uniform(-1.2, -0.9)]) # np.array([np.random.uniform(-1, -0.5), np.random.uniform(-0.9, -0.5)]) # (x, y, z)
        # else:
        #     target_center = np.array([self.rope_fixed_end_coord[0], np.random.uniform(-1.2, -0.9)]) # np.array([np.random.uniform(0.4, 1), np.random.uniform(-0.9, -0.5)]) # (x, y, z)

        target_center = np.array([self.rope_fixed_end_coord[0]+self.target_offest[0], -self.grasp_end[2]+self.target_offest[1]])
        # set the obstacle's orientation to be vertical to the line of rope fixed end and target
        fixed_end = self.rope_fixed_end_coord[0], -self.rope_fixed_end_coord[2]
        vector_dir_2D = (np.array(target_center) - np.array(fixed_end)) / np.linalg.norm((np.array(target_center) - np.array(fixed_end)))
        orn_dir_2D = np.array([-vector_dir_2D[1], vector_dir_2D[0]])
        self.left_obj_center = target_center - orn_dir_2D * (self.obs_gap + self.x_obj_radius) # [-0.5, 0.3] # (x, y)
        self.right_obj_center = target_center + orn_dir_2D * (self.obs_gap + self.x_obj_radius)
        obs_height = 0.45
        
        self.left_forbidden_area = np.array([[self.left_obj_center[0]-self.x_obj_radius*orn_dir_2D[0]+self.y_obj_radius*vector_dir_2D[0], \
                                            self.left_obj_center[1]-self.x_obj_radius*orn_dir_2D[1]+self.y_obj_radius*vector_dir_2D[1], \
                                            0.5], \
                                            [self.left_obj_center[0]+self.x_obj_radius*orn_dir_2D[0]-self.y_obj_radius*vector_dir_2D[0], \
                                            self.left_obj_center[1]+self.x_obj_radius*orn_dir_2D[1]-self.y_obj_radius*vector_dir_2D[1], \
                                            0.5+obs_height], \
                                            ]) # [vector_dir_2D[0], vector_dir_2D[1], 0.0]
        self.right_forbidden_area = np.array([[self.right_obj_center[0]-self.x_obj_radius*orn_dir_2D[0]+self.y_obj_radius*vector_dir_2D[0], \
                                            self.right_obj_center[1]-self.x_obj_radius*orn_dir_2D[1]+self.y_obj_radius*vector_dir_2D[1], \
                                            0.5], \
                                            [self.right_obj_center[0]+self.x_obj_radius*orn_dir_2D[0]-self.y_obj_radius*vector_dir_2D[0], \
                                            self.right_obj_center[1]+self.x_obj_radius*orn_dir_2D[1]-self.y_obj_radius*vector_dir_2D[1], \
                                            0.5+obs_height], \
                                            ]) # [vector_dir_2D[0], vector_dir_2D[1], 0.0]
        # np.array([[left_obj_center[0]+0.2, left_obj_center[1], 0.5]])
        # import pdb; pdb.set_trace()
        # target_pose = np.concatenate((target_center, [0.5]))[np.newaxis, :] # (x, y, z)
        # return target_pose
        rope_fixed_coord_in_normal_frame = self.rope_fixed_end_coord[[0,2,1]] * np.array([1,-1,1])
        rope_indices_y = np.linspace(0, self.get_rope_length(), num=10)
        target_rope = rope_fixed_coord_in_normal_frame[np.newaxis, :] + np.array([[0, -dy, 0] for dy in rope_indices_y])
        self.target_pose = target_rope[::-1] # (x, y, z)
        # self.get_fixed_action_sequence()
        return self.target_pose # (x, y, z)
    
    def get_fixed_action_sequence(self,):
        last_ee = self.last_ee
        way_points = []
        way_point0 = np.array([last_ee[0], last_ee[1], last_ee[2]])
        way_point1 = np.array([last_ee[0], self.left_obj_center[1]+0.4, 3.5])
        way_point2 = np.array([self.left_obj_center[0]+0.2, self.left_obj_center[1]+0.2, 3.0])
        way_point3 = np.array([self.left_obj_center[0]+0.2, last_ee[1], last_ee[2]])
        way_points = [way_point0, way_point1, way_point2, way_point3]
        actions = []
        for idx in range(len(way_points)-1):
            steps = np.linalg.norm(way_points[idx+1] - way_points[idx]) / 0.2 + 1
            t = np.linspace(0, 1, int(steps))
            actions.extend([way_points[idx] + (way_points[idx+1] - way_points[idx]) * i for i in t])
        self.fixed_action_sequence = actions
        
    def update(self, u):
        self.particle_pos_list = []
        self.eef_states_list = []
        self.step_list = []
        self.contact_list = []
        # if self.fix_rope_end:
        #     fixed_end_coord = self.rope_fixed_end_coord[[0,2,1]]*np.array([1,-1,1])
        #     if np.linalg.norm(fixed_end_coord - u - 0.5 ) > self.max_action_radius:
        
        # our model predicts the eef_states, so we need to reverse the y axis manually
        # self.step takes in the [x_e, y_e, z_e]
        # import pdb; pdb.set_trace()
        
        _, self.particle_pos_list, self.eef_states_list, self.step_list, self.contact_list = self.step(u)
    
    def get_rope_fixed_end_coord(self,):
        return self.rope_fixed_end_coord[[0,2,1]]*np.array([1,-1,1])
    
    def get_rope_length(self,):
        return self.rope_length
    
    # coordinate is tranformed to x, y, z
    def get_current_state(self,):
        particle_pos = self.particle_pos_list[-1][self.fps_idx]
        particle_pos = particle_pos[:, [0,2,1]] * np.array([[1,-1,1]])
        # x, y, z in fact this is finger joint position
        eef_pos_final = np.mean(self.eef_states_list[-1], axis=0)[[0,2,1]] * np.array([1,-1,1])
        eef_pos_start = np.mean(self.eef_states_list[0], axis=0)[[0,2,1]] * np.array([1,-1,1])
        length = eef_pos_final - eef_pos_start
        return np.concatenate([particle_pos, [eef_pos_final], [length]])
    
    # coordinate is tranformed to x, y, z
    def get_pusher_start_position(self,):
        return np.mean(self.eef_states_list[0], axis=0)[[0,2,1]] * np.array([1,-1,1]) # the last state is reset
    
    # coordinate is tranformed to x, y, z
    def get_pusher_position(self,):
        # end_effector_position is different from eef_states_list[-1](this is finger_position+0.5)
        return np.mean(self.eef_states_list[-1], axis=0)[[0,2,1]] * np.array([1,-1,1])# the last state is reset
    
    # this is the end effector position used for self.update
    def get_end_effector_position(self,):
        return np.mean(self.eef_states_list[-1], axis=0)[[0,2,1]] * np.array([1,-1,1]) + np.array([0,0,1.616-0.5]) # the last state is reset
    
    def reorder_particles(self, particle_pos, fps_idx):
        # reorder the particles based on the distance to the end effector
        particle_pos = particle_pos.reshape(-1, 2)
        # scaler = StandardScaler()
        # particle_pos_scaled = scaler.fit_transform(particle_pos)
        particle_pos_scaled = particle_pos - np.mean(particle_pos, axis=0)
        pca = PCA(n_components=1)  # We want to reduce the data to 2 principal components
        particle_pos_pca = pca.fit_transform(particle_pos_scaled).reshape(-1)
        # import pdb; pdb.set_trace()
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
