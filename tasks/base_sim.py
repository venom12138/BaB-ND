import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from pygame.color import *
import time, os

import pymunk
from pymunk import Vec2d
# import pymunk.pyglet_util
import pymunk.pygame_util
import math
import numpy as np
import random
import cv2
import PIL
from PIL import Image, ImageSequence
COLOR_LISTS = [(0, 0, 255, 255), (255, 255, 0, 255), (0, 255, 0, 255), (255, 0, 255, 255), (255, 255, 255, 255), (0, 0, 0, 255), (255, 0, 0, 255), (0, 255, 255, 255)]

class Base_Sim(object):
    def __init__(self, SAVE_IMG=False, ENABLE_VIS=False, pusher_size=5):
        # Sim window parameters. These also define the resolution of the image
        self.width = 500 # 800
        self.height = 500 # 800
        self.elasticity = 0.1
        self.friction = 0.6
        self.obj_mass = 100
        self.velocity = np.array([0, 0])
        self.pusher_body = None
        self.pusher_shape = None
        self.pusher_size = pusher_size # 2 dimension
        self.global_time = 0.0
        self.obj_num = 0
        self.obj_list = []
        self.image_list = []
        self.colors = COLOR_LISTS
        self.SAVE_IMG = SAVE_IMG
        self.ENABLE_VIS = ENABLE_VIS
        if self.ENABLE_VIS or self.SAVE_IMG:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        else:
            self.screen = None
            self.draw_options = None
        self.space = pymunk.Space()

        # User flags.
        # If this flag is enabled, then the rendering will be done at every
        # simulation timestep. This makes the sim much slower, but is better
        # for visualizing the pusher continuously.
        self.RENDER_EVERY_TIMESTEP = False
        self.frames = []

    def create_world(self, init_poses, pusher_pos):
        self.space.gravity = Vec2d(0, 0)  # planar setting
        self.space.damping = 0.0001  # quasi-static. low value is higher damping.
        self.space.iterations = 5  # TODO(terry-suh): re-check. what does this do?
        self.space.color = pygame.color.THECOLORS["white"]
        self.add_objects(self.obj_num, init_poses)
        self.add_pusher(pusher_pos)
        self.wait(1.0)
        self.render()

    def add_objects(self, obj_num, poses=None):
        """
        Create and add multiple L-shaped Blocks to sim.
        """
        if poses is None:
            for i in range(obj_num):
                self.add_object(i)
        else:
            for i in range(obj_num):
                self.add_object(i, poses[i])

    def add_object(self, id, pose=None):
        raise NotImplementedError

    def create_object(self, id, poses=None):
        raise NotImplementedError

    def remove_objects(self):
        raise NotImplementedError

    def update_object_pose(self, index, new_pose):
        raise NotImplementedError

    def get_object_positions(self):
        raise NotImplementedError

    def get_object_angles(self):
        raise NotImplementedError

    def get_object_keypoints(self, index, target=False, **kwargs):
        raise NotImplementedError

    def get_all_object_keypoints(self, target=False, **kwargs):
        all_keypoints = []
        for i in range(len(self.obj_list)):
            all_keypoints.append(self.get_object_keypoints(i, target, **kwargs))

        return all_keypoints

    def get_current_state(self):
        raise NotImplementedError

    def create_pusher(self, position):
        """
        Create a single pusher by defining its shape, mass, etc.
        """
        body = pymunk.Body(1e7, float("inf"))
        if position is None:
            body.position = Vec2d(
                random.randint(int(self.width * 0.25), int(self.width * 0.75)),
                random.randint(int(self.height * 0.25), int(self.height * 0.75)),
            )
        else:
            body.position = Vec2d(position[0], position[1])
        shape = pymunk.Circle(body, radius=self.pusher_size)
        shape.elasticity = self.elasticity
        shape.friction = self.friction
        shape.color = (255, 0, 0, 255)
        return body, shape

    def add_pusher(self, position):
        """
        Create and add a single pusher to the sim.
        """
        self.pusher_body, self.pusher_shape = self.create_pusher(position)
        self.space.add(self.pusher_body, self.pusher_shape)

    def remove_pusher(self):
        """
        Remove pusher from simulation.
        """
        if hasattr(self, "pusher_body") and self.pusher_body is not None:
            self.space.remove(self.pusher_body, self.pusher_shape)

    def get_pusher_position(self):
        """
        Return the position of the pusher.
        """
        if self.pusher_body is None:
            return None, None
        return self.pusher_body.position

    def update(self, u):
        """
        Once given a control action, run the simulation forward and return.
        """
        # Parse into integer coordinates
        uxf = u[0]
        uyf = u[1]

        # add the pusher if not added
        if self.pusher_body is None:
            self.add_pusher((uxf, uyf))
            # self.render()
            return None

        uxi, uyi = self.pusher_body.position

        # transform into angular coordinates
        theta = np.arctan2(uyf - uyi, uxf - uxi)
        length = np.linalg.norm(np.array([uxf - uxi, uyf - uyi]), ord=2)

        n_sim_step = 60
        step_dt = 1.0 / n_sim_step
        self.velocity = np.array([np.cos(theta), np.sin(theta)]) * length
        self.pusher_body.velocity = self.velocity.tolist()

        for i in range(n_sim_step):
            # make sure that pos_next = pos_curr + vel * dt (in pixel space)
            self.space.step(step_dt)
            self.global_time += step_dt
            # print(self.pusher_body.velocity )
        current_state = self.get_current_state()

        # Wait 1 second in sim time to slow down moving pieces, and render.
        # self.wait(1.0)
        self.render()
        return current_state

    def wait(self, time):
        """
        Wait for some time in the simulation. Gives some time to stabilize bodies in collision.
        """
        t = 0
        step_dt = 1 / 60.0
        while t < time:
            self.space.step(step_dt)
            t += step_dt

    """
    2.2 Methods related to rendering
    """

    def render(self):
        """
        Render the simulation.
        """
        raise NotImplementedError

    """
    2.3 Methods related to image publishing
    """

    def save_mp4(self, filename="output_video.mp4", fps=10):
        """
        Save the list of images as a video.

        Parameters:
            filename (str): Video filename
            fps (int): Frames per second
        """
        if not self.SAVE_IMG:
            print("no save")
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, fps, (self.width, self.height))

        try:
            for frame in self.image_list:
                out.write(frame[:, :, ::-1])
            out.release()
            print(f"Video saved as {filename}")
        except Exception as e:
            out.release()  # Make sure to release the video writer object
            print(f"Failed to save video. Error: {e}")
        # self.image_list = []

    def play_mp4(self, filename="output_video.mp4"):
        """
        Play the video using OpenCV (or do other processing)

        Parameters:
            filename (str): Video filename
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return
        cap = cv2.VideoCapture(filename)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow(filename, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def save_gif(self, filename="output_video.gif", fps=5):
        """
        Save the list of images as a gif.

        Parameters:
            filename (str): Gif filename
            fps (int): Frames per second
        """
        if not self.SAVE_IMG:
            print("no save")
            return
        images = []
        for frame in self.image_list:
            images.append(Image.fromarray(frame))
        images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=1000 / fps, loop=0)
        print(f"Gif saved as {filename}")
        # self.image_list = []

    def play_gif(self, filename="output_video.gif"):
        """
        Play the gif using OpenCV (or do other processing)

        Parameters:
            filename (str): Gif filename
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return
        # Read the gif from the file
        img = Image.open(filename)
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]

        for frame in frames:
            # Convert the PIL image to an OpenCV frame
            # opencv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            opencv_image = np.array(frame)

            # Display the frame
            cv2.imshow(filename, opencv_image)

            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

    """
    3. Methods for External Commands
    """

    def refresh(self, new_poses=None):
        self.remove_objects()
        self.remove_pusher()
        self.pusher_body = None
        self.pusher_shape = None
        self.add_objects(self.obj_num, new_poses)

        self.wait(1.0) # Give some time for collision pieces to stabilize.
        self.render()
        self.image_list = []
    
    def close(self, ):
        self.remove_objects()
        self.remove_pusher()
