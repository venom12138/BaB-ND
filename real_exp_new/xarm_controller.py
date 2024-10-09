import threading
import multiprocessing as mp
import queue
import time
from enum import Enum


from pynput import keyboard, mouse
from xarm.wrapper import XArmAPI

import numpy as np
# import pybullet as p

# test sapien
import sapien.core as sapien
import transforms3d

from common.kinematics_utils import KinHelper


# np.set_printoptions(precision=2,suppress=True)
UPDATE_INTERVAL = 0.01


class ControllerState(Enum):
    INIT = 0
    RUNNING = 1
    STOP = 2

class XarmController(mp.Process):
    kin_helper = KinHelper(robot_name='xarm6')
    XY_MIN, XY_MAX = 50, 700
    X_MIN,X_MAX = 200, 600
    Y_MIN,Y_MAX = -250, 250
    Z_MIN, Z_MAX =  40, 650
    
    MOVE_RATE = 1000 #hz
    MOVE_SLEEP = 1/MOVE_RATE
    XYZ_VELOCITY = 0.1
    ANGLE_VELOCITY_MAX = 5 # degree
    name="xarm_controller"
    
    def log(self,msg):
        if self.verbose:
            print(f"\033[92m{msg}\033[0m")

    def __init__(self, start_time, init_pose, ip="192.168.1.209", command_hz=None, verbose=False):
        verbose = True
        self.INIT_POS = init_pose
        super().__init__()
        self.start_time = start_time
        self._ip = ip
        # input(f"Connecting to {self._ip}, Press Enter to continue...")
        self.verbose = verbose
        # self.arm = None
        self.arm = XArmAPI(self._ip)
        self.arm.motion_enable(enable=True)
        self.exe_lock = mp.Lock()
        if command_hz is not None:
            self.max_command_time = 1 / command_hz
        else:
            self.max_command_time = None

        self.state = mp.Value('i', ControllerState.INIT.value)

        # NOTE: because the arm can only be interacted with the process created the API class
        self.command_q = mp.Queue()
        self.cur_position_q = mp.Queue(maxsize=1)

    def reset(self):
        # init pose
        if not self.exe_lock.acquire(block=True,timeout=1):
            self.log("xarm reset failed! exe_lock not acquired!")
            return
        next_position = self.INIT_POS
        self.move(next_position, steps=500, wait=True, clean=True)
        
        self.state.value = ControllerState.RUNNING.value
        self.exe_lock.release()


    def update_cur_position(self):
        # tic = time.time()
        while self.state.value in [ControllerState.RUNNING.value, ControllerState.INIT.value]:
            # toc = time.time()
            # tic = toc
            cur_qpos = np.array(self.arm.get_servo_angle()[1][0:6]) / 180. * np.pi
            fk_trans_mat = self.kin_helper.compute_fk_sapien_links(cur_qpos, [self.kin_helper.sapien_eef_idx])[0]
            cur_position = np.zeros(6)
            cur_position[:3] = fk_trans_mat[:3,3]*1000
            cur_position[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3,:3],)
            cur_position[3:] = cur_position[3:] / np.pi * 180
            
            # self.log(f"d_pos: {self.arm.get_position()[1][0:6]-cur_position}")

            if not self.cur_position_q.full():
                # print(f"put agent: {time.time()}")
                self.cur_position_q.put(fk_trans_mat)

            self.cur_position = cur_position
            # self.log(self.cur_position)
            # if self.max_command_time is not None:
            #     time.sleep(self.max_command_time - (time.time()-toc))
            # else:
            #     time.sleep(UPDATE_INTERVAL)
            time.sleep(UPDATE_INTERVAL)

    def run(self):
        # the arm initialization must be invoked in the same process
        self.arm = XArmAPI(self._ip)
        if self.verbose:
            self.log(f"Connected to xarm at {self._ip}")
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(state=0)

        self.update_pos_t = threading.Thread(target=self.update_cur_position, name="update_cur_position")
        self.update_pos_t.start()

        time.sleep(1)

        self.reset()
        # time.sleep(10)
        
        while self.state.value == ControllerState.RUNNING.value:
            # self.log(self.cur_position)
            # self.log("df")
            # time.sleep(0.1)
            if self.command_q.empty():
                # self.log(f"empty comand q: {self.cur_position}")
                time.sleep(0.01)
                continue
            with self.exe_lock:
                self.log("xarm controller get lock")
                self.log(f"new commands at {time.time()-self.start_time:.2f}")
                while not self.command_q.empty():
                    command = self.command_q.get()
                    self.log(f"get command: {command}")
                    command, wait = command
                    if isinstance(command,str) and command == "quit":
                        self.state.value = ControllerState.STOP.value
                        break
                    self.move(command,steps=125, wait=wait, clean=True)
                    # time.sleep(0.01)
            
        self.update_pos_t.join()
        self.log(f"{'='*20} xarm exit! state: {ControllerState(self.state.value)}")

    def check_valid_move(self, next_position, steps):
        # absolute position
        x,y,z,roll,pitch,yaw = next_position
        if x**2+y**2 > self.XY_MAX**2 or x**2+y**2 < self.XY_MIN**2\
            or x<self.X_MIN or x>self.X_MAX or y<self.Y_MIN or y>self.Y_MAX:
            self.log(f"invalid move command {next_position}! x,y out of range!")
            return False
        elif z > self.Z_MAX or z < self.Z_MIN:
                self.log(f"invalid move command {next_position}! z out of range!")
                return False
        
        # relative position
        # init_position = self.arm.get_position()[1]
        # d_pos = np.array(next_position) - np.array(init_position)
        # if np.linalg.norm(d_pos[0:3]) > self.VELOCITY_MAX*steps:
        #     self.log(f"invalid move command {next_position}! translate too fast!")
        #     return False
        # elif (abs(d_pos[3:])%360 > self.ANGLE_VELOCITY_MAX).any():
        #     self.log(f"invalid move command {next_position}! rotation too fast!")
        #     return False
        
        return True

    def move(self,next_position, steps=50, wait=True, clean=True):
        assert next_position is not None, "next_position is not set!"
        next_position = np.array(next_position)
        # if not self.check_valid_move(next_position,steps):
        #     print(f"invalid move command {next_position}!")
        #     return
        
        self.log(f'move start: {next_position}')

        # initial_qpos = self.arm.get_servo_angle()[1][0:6]
        initial_qpos = np.array(self.arm.get_servo_angle()[1][0:6]) / 180. * np.pi
        # log('init', initial_qpos)
        next_position_unit = np.zeros_like(np.array(next_position))
        next_position_unit[0:3] = np.array(next_position)[0:3] / 1000.
        next_position_unit[3:] = np.array(next_position)[3:] / 180. * np.pi
        next_servo_angle = self.kin_helper.compute_ik_sapien(initial_qpos, next_position_unit)
        # log('next_servo',next_servo_angle)
        joints = next_servo_angle

        # NOTE: In the servo mode: state(1), the actual speed is contorlled by the
        #       rate of command sending and the the distance between the current and target position 
        # The steps of each move is decided by the distance of moving to keep speed constant
        init_position = self.kin_helper.compute_fk_sapien_links(initial_qpos, [self.kin_helper.sapien_eef_idx])[0][:3,3]*1000
        d_pos = np.array(next_position[:3]) - np.array(init_position)
        distance = np.linalg.norm(d_pos)
        min_steps = int(distance / self.XYZ_VELOCITY)
        self.log(f"distance: {distance}, min_steps: {min_steps}")
        steps = max(min_steps, steps)
        # steps *= 5

        # init_position_fk = self.kin_helper.compute_fk_sapien_links(initial_qpos, [self.kin_helper.sapien_eef_idx])[0]

        # self.log(next_servo_angle - initial_qpos)
        # self.log(d_pos)
        # self.log(np.array(next_position) - np.array(init_position_fk))

        tic = time.time()
        for i in range(steps): 
            # start_time = time.time()
            angle = initial_qpos + (next_servo_angle - initial_qpos) * (i+1) / steps
            # self.log(f"inter move: {self.cur_position}")
            # if i % 10 == 0:
            #     f = open('log/keyboard_action_1.txt', 'a')
            #     f.write(self.arm.get_position()[1].__str__())
            #     f.write('\n')
            self.arm.set_servo_angle_j(angles=angle,is_radian=True, speed=1)
            time.sleep(self.MOVE_SLEEP)
            if (not wait) and self.max_command_time is not None and time.time()-tic > self.max_command_time:
                self.log(f"move timeout: {time.time()-tic:.2f}, executed steps: {i}/{steps}")
                break
            # elapsed_time = time.time() - start_time
            # loop_time = 0.01
            # if elapsed_time < loop_time:
            #     time.sleep(loop_time - elapsed_time)
        self.log(f"move end: volecity: {distance/(time.time()-tic):.2f} mm/s")

        # self.log(f"move done, {self.cur_position}")
        # self.log(f"q error: {np.array(self.arm.get_servo_angle()[1][0:6]) - next_servo_angle/np.pi*180} degree")
        # self.log(f"p error: {self.cur_position - next_position}")
        if clean:
            self.arm.clean_error()
            self.arm.clean_warn()

    def start(self) -> None:
        return super().start()
    
    def stop(self):
        self.state.value = ControllerState.STOP.value


class KeyboardTeleop(mp.Process):
    
    XYZ_STEP = 5
    ANGLE_STEP = 0.1
    name="keyboard_teleop"

    def __init__(self, start_time, controller=None) -> None:
        super().__init__()
        self.controller = controller
        self.start_time = start_time
        self.key_states = {
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "p": False,
            "up": False,    # Up arrow key for moving up
            "down": False,  # Down arrow key for moving down
            "z": False,     # Key for adjusting roll
            "x": False,     # Key for adjusting pitch
            "c": False,      # Key for adjusting yaw
            "space": False, # Key for send current position
            "shift": False, # Key for slow down the speed
        }
        self.init = True
        self.pause = False
        self.cur_position = None

    @staticmethod
    def log(msg):
        print(f"\033[94m{msg}\033[0m")

    def on_press(self,key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = True
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_states["up"] = True
            elif key == keyboard.Key.down:
                self.key_states["down"] = True
            elif key == keyboard.Key.space:
                self.key_states["space"] = True
            elif key == keyboard.Key.shift:
                self.key_states["shift"] = True

    def on_release(self,key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = False
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_states["up"] = False
            elif key == keyboard.Key.down:
                self.key_states["down"] = False
            elif key == keyboard.Key.space:
                self.key_states["space"] = False
            elif key == keyboard.Key.shift:
                self.key_states["shift"] = False
            
            
            if key == keyboard.Key.esc:
                return False

    def set_controller(self, controller):
        assert isinstance(controller, XarmController), "controller must be an instance of XarmController"
        self.controller = controller
    
    def update_xarm_pos(self):
        while self.keyboard_listener.is_alive():
            try:
                self.cur_position = self.controller.cur_position_q.get()
                time.sleep(0.005)
            except queue.Empty:
                pass
        self.log(f"update_xarm_pos exit!")

    def get_command(self):
        if self.cur_position is None:
            return
        fk_trans_mat = self.cur_position
        
        cur_position = np.zeros(6)
        cur_position[:3] = fk_trans_mat[:3,3]*1000
        cur_position[3:] = transforms3d.euler.mat2euler(fk_trans_mat[:3,:3],)
        cur_position[3:] = cur_position[3:] / np.pi * 180
            
        x, y, z, roll, pitch, yaw = cur_position
        
        if self.key_states["p"]:
            # abandon all other keyinputs
            self.pause = not self.pause
            self.log(f"keyboard teleop pause: {self.pause}")
            time.sleep(0.5)

        if self.pause:
            self.command = []
            return
        else:
            xyz_step = 1 if self.key_states["shift"] else self.XYZ_STEP
            if self.key_states["w"]:
                x += xyz_step
            if self.key_states["s"]:
                x -= xyz_step
            if self.key_states["a"]:
                y += xyz_step
            if self.key_states["d"]:
                y -= xyz_step
            if self.key_states["up"]:
                z += xyz_step   # Move up
            if self.key_states["down"]:
                z -= xyz_step   # Move down
            # if self.key_states["z"]:
            #     roll += self.ANGLE_STEP
            # if self.key_states["x"]:
            #     pitch += self.ANGLE_STEP
            # if self.key_states["c"]:
            #     yaw += self.ANGLE_STEP
            if self.key_states["space"]:
                # if space key is pressed, send current position
                pass
            # roll, pitch, yaw = 180,0,0

            next_position = [x, y, z, roll, pitch, yaw]
            assert abs(roll-180)<5 or abs(roll)<5
            assert abs(pitch-180)<5 or abs(pitch)<5
            assert abs(yaw-180)<5 or abs(yaw)<5
            
            if any([self.key_states[x] for x in ["w","s","a","d","up","down","space"]]):
                # self.init = False
                # self.log(f"get command: {next_position}")
                self.command = [next_position]
            else:
                self.command = []
            return

    def run(self) -> None:
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()
        
        self.update_pos_t = threading.Thread(target=self.update_xarm_pos)
        self.update_pos_t.start()
        time.sleep(1)

        while self.keyboard_listener.is_alive():
            # always update action from perception
            # self.log(self.cur_position)

            self.get_command()
            if self.controller.exe_lock.acquire(block=False):
                if not self.controller.command_q.empty():
                    continue
                for c in self.command:
                    self.log(f"put command: {c}")
                    self.controller.command_q.put(c)
                    self.cur_position = c
                self.controller.exe_lock.release()
                time.sleep(0.001)
        self.controller.command_q.put("quit")
        self.log(f"{'='*20} keyboard teleop exit!")
        # self.update_pos_t.stop()
        # self.keyboard_listener.stop()
            

if __name__ == "__main__":
    # mp.set_start_method('fork')
    start_time = time.time()
    controller = XarmController(start_time)
    controller.start()
    
    time.sleep(2)
    keyboard_teleop = KeyboardTeleop(start_time)
    keyboard_teleop.set_controller(controller)
    keyboard_teleop.start()

    keyboard_teleop.join()
    controller.join()