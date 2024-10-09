import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.locals import *
from pygame.color import *
import os

import pymunk
from pymunk import Vec2d
# import pymunk.pyglet_util
from scipy.spatial import ConvexHull
import time

import numpy as np
import random
import cv2
import PIL
from tasks.base_sim import Base_Sim
from sklearn.cluster import KMeans

"""Main Simulation class for ObjectPileSim.

The attributes and methods are quite straightforward, see ObjectPileSim.py
for basic usage.
"""
class ObjectPileSim(Base_Sim):
    def __init__(self, param_dict, init_poses=None, target_poses=None, pusher_pos=None):
        super().__init__(param_dict["save_img"], param_dict["enable_vis"], param_dict["pusher_size"])
        self.env_type = param_dict['env_type']

        # Simulation parameters.
        self.obj_num = param_dict["obj_num"]
        if init_poses is not None:
            assert len(init_poses) == self.obj_num
        self.obj_size = param_dict["obj_size"]
        self.label_list = []
        self.classes = param_dict["classes"]
        self.push_single = param_dict["push_single"]

        self.bin_a_pos = np.array([150, 100])
        self.bin_b_pos = np.array([350, 100])
        self.bin_side_length = 150
        self.bin_width = 6
        # if init_poses is not None:
        self.create_world(init_poses, pusher_pos)

    def create_world(self, init_poses, pusher_pos):
        self.space.gravity = Vec2d(0,0) # planar setting 
        self.space.damping = 0.0001 # quasi-static. low value is higher damping.
        self.space.iterations = 5 # TODO(terry-suh): re-check. what does this do? 
        self.space.color = pygame.color.THECOLORS["white"]
        if init_poses is not None:
            self.add_objects(init_poses)
        if pusher_pos is not None:
            self.add_pusher(pusher_pos)
        if self.env_type == 'bin':
            self.add_bins()

        self.wait(1.0) # give some time for colliding pieces to stabilize.
        
        if self.ENABLE_VIS or self.SAVE_IMG:
            self.render()

    def generate_random_poly(self, center, radius):
        """
        Generates random polygon by random sampling a circle at "center" with
        radius "radius". Its convex hull is taken to be the final shape.
        """
        k = 12
        r = radius # TODO: allow more possible radius * (np.random.rand(k, 1) / 2. + 0.5)
        
        theta = 2 * np.pi * np.arange(0, 1, step=1/k)# np.random.uniform(0,1,size=(k, 1))
        theta = np.expand_dims(theta, axis=1)
        p = np.hstack((r * np.cos(theta), r * np.sin(theta)))
        
        return np.take(p, ConvexHull(p).vertices, axis=0)

    def add_objects(self, init_poses):
        """
        Create and add multiple obj pieces to sim.
        """
        assert self.obj_num % self.classes == 0
        num_labels = self.obj_num // self.classes # 6 // 2 = 3
        for i in range(self.obj_num):
            self.add_object(list(init_poses[i]), i // num_labels)
        if self.push_single:
            for i in range(1, self.obj_num):
                self.label_list[i] = 1
            
    def add_object(self, pose, label):
        """
        Create and add a single object piece to the sim.
        """
        color = self.colors[label]
        object_body, object_shape = self.create_object(pose, color)
        self.space.add(object_shape.body, object_shape)
        self.obj_list.append([object_body, object_shape,])
        self.label_list.append(label)

    def create_object(self, pose, color):
        """
        Create a single object piece by defining its shape, mass, etc.
        """
        points = self.generate_random_poly((0, 0), self.obj_size)
        inertia = pymunk.moment_for_poly(self.obj_mass, points.tolist(), (0, 0))
        body = pymunk.Body(self.obj_mass, inertia)
        # TODO(terry-suh): is this a reasonable distribution?

        if self.env_type == 'bin':
            body.position = pose # Vec2d(random.randint(200, 300), random.randint(300, 400))
        elif self.env_type == 'plain':
            body.position = pose # Vec2d(random.randint(200, 300), random.randint(200, 300))
        #body.position = Vec2d(random.randint(0, 500), random.randint(0, 500))

        shape = pymunk.Poly(body, points.tolist())
        shape.color = color
        shape.elasticity = self.elasticity
        shape.friction = self.friction
        return body, shape
    
    def remove_objects(self):
        """
        Remove all box pieces in sim.
        """
        for i in range(len(self.obj_list)):
            box_body = self.obj_list[i][0]
            box_shape = self.obj_list[i][1]
            self.space.remove(box_shape, box_shape.body)
        self.obj_list = []
        self.label_list = []

    def refresh(self, new_poses=None):
        self.remove_objects()
        self.remove_pusher()
        self.pusher_body = None
        self.pusher_shape = None
        self.add_objects(new_poses)
        if new_poses is not None:
            assert len(new_poses) == self.obj_num
        self.wait(1.0)  # Give some time for collision pieces to stabilize.
        # self.render()
        self.frames = []
    
    def get_current_state(self):
        state = np.zeros((self.obj_num + 2, 2))
        for i in range(self.obj_num):
            state[i] = np.array(self.obj_list[i][0].position)
        state[-2] = np.array(self.pusher_body.position)
        state[-1] = np.array(self.length)
        return state

    def get_current_color(self):
        colors = np.zeros((self.obj_num + 1, 4))
        colors[0] = np.array(self.pusher_shape.color)
        for i in range(self.obj_num):
            colors[i + 1] = np.array(self.obj_list[i][1].color)
        return colors

    def get_obj_center(self, lst_center=None):
        obj_positions = np.array([body.position for body, shape in self.obj_list])
        center_x_obj = np.mean(obj_positions[:, 0])
        center_y_obj = np.mean(obj_positions[:, 1])
        
        if lst_center == None:
            return center_x_obj, center_y_obj
        sse = []
        centers = []
        for k in range(1, 3):
            kmeans = KMeans(n_clusters=k, n_init=2)
            kmeans.fit(obj_positions)
            centers.append(kmeans.cluster_centers_)
            sse.append(kmeans.inertia_)
        sse = np.array(sse)
        sse_diff = sse[:-1] - sse[1:]
        
        if max(sse_diff) < 10000:
            return lst_center
        else:
            idx = list(sse_diff).index(max(sse_diff)) + 1
            centers = centers[idx]
            dist = np.sum((centers - np.array(lst_center))**2, axis=1)
            closest_idx = np.argmin(dist)
            if np.random.rand() < 0.3:
                return random.choice(centers)
            else:
                return list(centers[closest_idx])
        
    # def set_pusher_pose(self, position, angle):
    #     uxf, uyf = position
    #     if self.pusher_body is None:
    #         self.add_pusher((uxf, uyf))
    #     else:
    #         self.pusher_body.position = (uxf, uyf)
    #         self.pusher_body.angle = np.arctan2(angle[1], angle[0]) + np.pi / 2 

    def set_pusher_position(self, uxf, uyf):
        if self.pusher_body is None:
            self.add_pusher((uxf, uyf))
        else:
            self.pusher_body.position = (uxf, uyf)
    
    # Update the sim by applying action, progressing sim, then stopping. 
    def update(self, u, steps=None):
        """
        Once given a control action, run the simulation forward and return.
        """
        # Parse into integer coordinates
        uxf = u[0]
        uyf = u[1]
        # add the bar if not added
        if self.pusher_body is None:
            self.add_pusher((uxf, uyf))
            # if (self.ENABLE_VIS or self.SAVE_IMG):
            #     self.render()
            return None

        uxi, uyi = self.pusher_body.position
        self.pusher_body.angle = np.arctan2(uyf - uyi, uxf - uxi) + np.pi / 2 # np.pi / 2 + np.pi/4 #
        
        # transform into angular coordinates
        theta = np.arctan2(uyf - uyi, uxf - uxi) # np.arctan2(uxf - uxi, uyf - uyi)
        length = np.linalg.norm(np.array([uxf - uxi, uyf - uyi]), ord=2)

        # print(uxi, uyi, uxf, uyf, length)
        if steps == None:
            n_sim_step = 60
            total_time = 2
            step_dt = total_time / n_sim_step
        else:
            n_sim_step = int(steps)
            step_dt = 1 / 60
            total_time = n_sim_step * step_dt

        # print(length)
        self.length = np.array([np.cos(theta), np.sin(theta)]) * length
        self.velocity = np.array([np.cos(theta), np.sin(theta)]) * length / total_time
        self.pusher_body.velocity = self.velocity.tolist()
        # before_position = self.pusher_body.position
        for i in range(n_sim_step):
            self.pusher_body.velocity = self.velocity.tolist()
            self.space.step(step_dt)
            self.global_time += step_dt
            if i %15 == 0:
                if (self.ENABLE_VIS or self.SAVE_IMG):
                    self.render()
        # wait until the obj stops moving
        for _ in range(5):
            self.pusher_body.velocity = np.zeros_like(self.velocity).tolist()
            self.space.step(step_dt)
            self.global_time += step_dt
        if (self.ENABLE_VIS or self.SAVE_IMG):
            self.render()
        return None

    def render(self):
        self.screen.fill((220, 220, 220))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        frame_image = pygame.surfarray.array3d(self.screen)
        frame_image = frame_image.transpose([1, 0, 2])
        frame_image = np.flip(frame_image, axis=0)
        
        self.frames.append(frame_image)

    def create_pusher(self, position):
        """
        Create a single bar by defining its shape, mass, etc.
        """
        body = pymunk.Body(1e7, float('inf'))
        body.position = Vec2d(position[0], position[1])
        # shape = pymunk.Circle(body, radius=5)
        shape = pymunk.Poly.create_box(body, self.pusher_size)
        # pymunk.Poly.create_box(body, (50, 50))
        shape.elasticity = self.elasticity
        shape.friction = self.friction
        shape.color = (255, 0, 0, 255)
        return body, shape




    # for other environments
    def add_bin(self, start, end, color):
        body = pymunk.Body(1e7, float('inf'))
        shape = pymunk.Segment(body, start.tolist(), end.tolist(), self.bin_width)
        shape.elasticity = 0.1
        shape.friction = 0.6
        shape.color = color
        self.space.add(body, shape)
        self.bins.append([body, shape])

    def add_bins(self):
        color = (255, 255, 255, 255)

        self.bins = []

        ### bin a

        center = np.array([self.bin_a_pos[0] - self.bin_side_length / 2., self.bin_a_pos[1]])
        start = np.array([center[0], center[1] + self.bin_side_length / 2.])
        end = np.array([center[0], center[1] - self.bin_side_length / 2.])
        self.add_bin(start, end, color)

        center = np.array([self.bin_a_pos[0] + self.bin_side_length / 2., self.bin_a_pos[1]])
        start = np.array([center[0], center[1] + self.bin_side_length / 2.])
        end = np.array([center[0], center[1] - self.bin_side_length / 2.])
        self.add_bin(start, end, color)

        center = np.array([self.bin_a_pos[0], self.bin_a_pos[1] - self.bin_side_length / 2.])
        start = np.array([center[0] - self.bin_side_length / 2. + self.bin_width * 2, center[1]])
        end = np.array([center[0] + self.bin_side_length / 2. - self.bin_width * 2, center[1]])
        self.add_bin(start, end, color)

        ### bin b

        center = np.array([self.bin_b_pos[0] - self.bin_side_length / 2., self.bin_b_pos[1]])
        start = np.array([center[0], center[1] + self.bin_side_length / 2.])
        end = np.array([center[0], center[1] - self.bin_side_length / 2.])
        self.add_bin(start, end, color)

        center = np.array([self.bin_b_pos[0] + self.bin_side_length / 2., self.bin_b_pos[1]])
        start = np.array([center[0], center[1] + self.bin_side_length / 2.])
        end = np.array([center[0], center[1] - self.bin_side_length / 2.])
        self.add_bin(start, end, color)

        center = np.array([self.bin_b_pos[0], self.bin_b_pos[1] - self.bin_side_length / 2.])
        start = np.array([center[0] - self.bin_side_length / 2. + self.bin_width * 2, center[1]])
        end = np.array([center[0] + self.bin_side_length / 2. - self.bin_width * 2, center[1]])
        self.add_bin(start, end, color)

    def remove_bins(self):
        for i in range(len(self.bins)):
            bin_body = self.bins[i][0]
            bin_shape = self.bins[i][1]
            self.space.remove(bin_shape, bin_shape.body)
        self.bins = []



