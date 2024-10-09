import numpy as np
import sapien.core as sapien
import transforms3d
from pathlib import Path
from tasks.reorientation.sapien_env.rl_env.para import ARM_INIT
from urdfpy import URDF

class TeleopRobot():
    def __init__(self, robot_name="xarm6_with_gripper"):
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        current_dir = Path(__file__).parent
        package_dir = (current_dir.parent / "assets").resolve()
        assert "xarm" in robot_name
        xarm_path = f"{package_dir}/robot/xarm6_description/xarm6_with_gripper.urdf"
        self.xarm = loader.load(xarm_path)
        self.robot_model = self.xarm.create_pinocchio_model()
        self.teleop_ik = self.ik_xarm

        self.joint_names = [joint.get_name() for joint in self.xarm.get_active_joints()]
        self.link_name2id = {self.xarm.get_links()[i].get_name(): i for i in range(len(self.xarm.get_links()))}

        # self.urdf_robot = URDF.load(urdf_path)

    # copute fk in sapien in robot frame with qpos
    def compute_fk(self, qpos): 
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose = self.robot_model.get_link_pose(8)
        print("fk",link_pose)
        return link_pose
    
    def compute_fk_links(self, qpos, link_idx):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i))
        return link_pose_ls
    
    def init_cartisen_action(self,qpos):
        # fk = self.robot_model.compute_forward_kinematics(qpos)
        self.ee_link_pose = self.robot_model.get_link_pose(8)
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        p = self.ee_link_pose.p
        q=self.ee_link_pose.q
        eluer = transforms3d.euler.quat2euler(q,axes='sxyz')
        cartisen_action[0:3] = p
        cartisen_action[3:6] = eluer
        cartisen_action[6] = 0.021
        return cartisen_action

    def init_cartisen_action_franka(self,qpos):
        fk = self.robot_model.compute_forward_kinematics(qpos)
        ee_link_pose = self.robot_model.get_link_pose(11)
        cartisen_action_dim =6
        grip_dim = 1
        cartisen_action = np.zeros(cartisen_action_dim+grip_dim)
        p = ee_link_pose.p
        q=ee_link_pose.q
        eluer = transforms3d.euler.quat2euler(q,axes='sxyz')
        cartisen_action[0:3] = p
        cartisen_action[3:6] = eluer
        cartisen_action[6] = 0.021
        return cartisen_action

    def ik_xarm(self,initial_qpos,cartesian):
        p = cartesian[0:3]
        q = transforms3d.euler.euler2quat(ai=cartesian[3],aj=cartesian[4],ak=cartesian[5],axes='sxyz')
        pose = sapien.Pose(p=p, q=q)
        active_qmask= np.array([True,True,True,True,True,True,False,False]) # only have 6 joints
        # link index ranges from 0 to 14
        # [Actor(name="world", id="4"), Actor(name="link_base", id="5"), Actor(name="link1", id="6"), 
        # Actor(name="link2", id="7"), Actor(name="link3", id="8"), Actor(name="link4", id="9"), 
        # Actor(name="link5", id="10"), Actor(name="link6", id="11"), Actor(name="xarm_gripper_base_link", id="12"), 
        # Actor(name="left_outer_knuckle", id="17"), Actor(name="left_finger", id="18"), Actor(name="left_inner_knuckle", id="16"), 
        # Actor(name="right_outer_knuckle", id="14"), Actor(name="right_finger", id="15"), Actor(name="right_inner_knuckle", id="13")]
        # link_index =8 is xarm_gripper_base_link, rahter than link6
        # but from my observation, the position and orn of link6 is the same as xarm_gripper_base_link
        # this can be verified by line 314 in xarm6_with_gripper.urdf
        qpos = self.robot_model.compute_inverse_kinematics(link_index=8, pose=pose,\
                                                        initial_qpos=initial_qpos,active_qmask=active_qmask,damp=1e-1) # 8
        # print(qpos) 
        return qpos[0][0:6]

    def keyboard_control(self, viewer, cartesian):
        delta_position = 0.002
        constant_x = 1
        constant_y = 1
        constant_z = 1
        delta_orientation = 0.005
        # x
        if viewer.window.key_down("i"):
            cartesian[0]+=delta_position*constant_x
        if viewer.window.key_down("k"):
            cartesian[0]-=delta_position*constant_x
        # y
        if viewer.window.key_down("j"):
            cartesian[1]+=delta_position*constant_y
        if viewer.window.key_down("l"):
            cartesian[1]-=delta_position*constant_y
        # z
        if viewer.window.key_down("u"):
            cartesian[2]+=delta_position*constant_z
        if viewer.window.key_down("o"):
            cartesian[2]-=delta_position*constant_z
        # roll
        if viewer.window.key_down("r"):
            cartesian[3]+=delta_orientation
        if viewer.window.key_down("f"):
            cartesian[3]-=delta_orientation
        # pitch
        if viewer.window.key_down("t"):
            cartesian[4]+=delta_orientation
        if viewer.window.key_down("g"):
            cartesian[4]-=delta_orientation
        # yaw
        if viewer.window.key_down("y"):
            cartesian[5]+=delta_orientation
        if viewer.window.key_down("h"):
            cartesian[5]-=delta_orientation

        # gripper open or close
        if viewer.window.key_down("z"):
            cartesian[6]+=0.001
        if viewer.window.key_down("x"):
            cartesian[6]-=0.001
        cartesian[6] = np.clip(cartesian[6],0.021,0.057)
        return cartesian

    def keyboard_control_2arm(self, viewer, cartesian,cartesian_2):
        delta_position = 0.001
        constant_x = 1
        constant_y = 1
        constant_z = 1
        delta_orientation = 0.01
        # x
        if viewer.window.key_down("i"):
            cartesian[0]+=delta_position*constant_x
        if viewer.window.key_down("k"):
            cartesian[0]-=delta_position*constant_x
        # y
        if viewer.window.key_down("j"):
            cartesian[1]+=delta_position*constant_y
        if viewer.window.key_down("l"):
            cartesian[1]-=delta_position*constant_y
        # z
        if viewer.window.key_down("u"):
            cartesian[2]+=delta_position*constant_z
        if viewer.window.key_down("o"):
            cartesian[2]-=delta_position*constant_z
        # roll
        if viewer.window.key_down("r"):
            cartesian[3]+=delta_orientation
        if viewer.window.key_down("f"):
            cartesian[3]-=delta_orientation
        # pitch
        if viewer.window.key_down("t"):
            cartesian[4]+=delta_orientation
        if viewer.window.key_down("g"):
            cartesian[4]-=delta_orientation
        # yaw
        if viewer.window.key_down("y"):
            cartesian[5]+=delta_orientation
        if viewer.window.key_down("h"):
            cartesian[5]-=delta_orientation

        # gripper open or close
        if viewer.window.key_down("z"):
            cartesian[6]+=0.0005
        if viewer.window.key_down("x"):
            cartesian[6]-=0.0005
        cartesian[6] = np.clip(cartesian[6],0.01,0.09)

        # x
        if viewer.window.key_down("up"):
            cartesian_2[0]+=delta_position*constant_x
        if viewer.window.key_down("down"):
            cartesian_2[0]-=delta_position*constant_x
        # y
        if viewer.window.key_down("left"):
            cartesian_2[1]+=delta_position*constant_y
        if viewer.window.key_down("right"):
            cartesian_2[1]-=delta_position*constant_y
        # z
        if viewer.window.key_down("0"):
            cartesian_2[2]+=delta_position*constant_z
        if viewer.window.key_down("p"):
            cartesian_2[2]-=delta_position*constant_z
        # # roll
        # if viewer.window.key_down("r"):
        #     cartesian_2[3]+=delta_orientation
        # if viewer.window.key_down("f"):
        #     cartesian_2[3]-=delta_orientation
        # # pitch
        # if viewer.window.key_down("t"):
        #     cartesian_2[4]+=delta_orientation
        # if viewer.window.key_down("g"):
        #     cartesian_2[4]-=delta_orientation
        # # yaw
        # if viewer.window.key_down("y"):
        #     cartesian_2[5]+=delta_orientation
        # if viewer.window.key_down("h"):
        #     cartesian_2[5]-=delta_orientation

        # gripper open or close
        if viewer.window.key_down("n"):
            cartesian_2[6]+=0.01
        if viewer.window.key_down("m"):
            cartesian_2[6]-=0.01
        cartesian_2[6] = np.clip(cartesian_2[6],0.01,0.09)
        return cartesian, cartesian_2