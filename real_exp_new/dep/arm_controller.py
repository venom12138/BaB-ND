import multiprocessing as mp
import time
from queue import Empty
from multiprocessing.managers import SharedMemoryManager
from shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from xarm.wrapper import XArmAPI
import numpy as np

class ArmController(mp.Process):
    def __init__(self, shm_manager: SharedMemoryManager, \
                ):
        super().__init__()
        ip = "192.168.1.209"
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(1)
        self.arm.set_state(state=0)

        self.alive = False
        # self.arm = arm
        
        example = {
            'angle': np.zeros(6),
            'sleep_time':0.01
        }

        pose_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=512
        )
        self.pose_queue = pose_queue
        self.stop_event = mp.Event()

    def start(self,):
        self.alive = True
        super().start()

    def stop(self):
        self.alive = False
        self.join()
    
    def run(self,):
        print(f"Arm controller run")
        while True:
            if self.alive:
                # print(f"pose queue length: {self.pose_queue.qsize()}")
                if not self.pose_queue.empty():
                    commands = self.pose_queue.get_all()
                    angles = commands['angle']
                    sleep_times = commands['sleep_time']
                    n_cmds = len(sleep_times)
                    print(f"arm controller angles: {angles[0]} {angles[-1]}")
                    for i in range(n_cmds):
                        # print(f"angle: {angles[i]}, sleep_time: {sleep_times[i]}")
                        self.arm.set_servo_angle_j(angles=angles[i], is_radian=True)
                        time.sleep(sleep_times[i])
                else:
                    time.sleep(1)
            else:
                print("pose thread not alive")
                time.sleep(5)
    
    def put(self, pose):
        message = {
            'angle': pose[0],
            'sleep_time': pose[1]
        }
        self.pose_queue.put(message)
        # print(f"putting pose: {pose}, pose queue length: {self.pose_queue.qsize()}")
    
    def clear(self,):
        self.pose_queue.get_all()
    
    def empty(self,):
        return self.pose_queue.empty()
    
    def get_position(self,):
        return self.arm.get_position()
    
    def get_servo_angle(self,):
        return self.arm.get_servo_angle()
    
    def reset(self,):
        self.arm.reset(wait=True)


if __name__ == "__main__":
    from xarm.wrapper import XArmAPI
    ip = "192.168.1.209"
    arm = XArmAPI(ip)
    multi_controllers = multiArmController([XArmAPI(ip), XArmAPI(ip)])
    # controller = ArmController(arm)
    
    multi_controllers.start()

    # # Simulate adding some poses to the queue
    # for i in range(5):
    #     controller.pose_queue.put(([i, i + 1, i + 2], 2))  # Example angles and sleep time

    # # Run for a while then stop
    # time.sleep(10)
    # controller.stop()
    # print("Processing complete.")
