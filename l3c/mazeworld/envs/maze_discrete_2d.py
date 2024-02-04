"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
from pygame import font
from numpy import random as npyrnd
from numpy.linalg import norm
from l3c.mazeworld.envs.maze_base import MazeBase
from .ray_caster_utils import landmarks_rgb,landmarks_color

class MazeCoreDiscrete2D(MazeBase):
    def __init__(self, view_grid=1, task_type="SURVIVAL", max_steps=5000):
        super(MazeCoreDiscrete2D, self).__init__(
                view_grid=view_grid,
                task_type=task_type,
                max_steps=max_steps
                )

    def do_action(self, action):
        assert numpy.shape(action) == (2,)
        assert abs(action[0]) < 2 and abs(action[1]) < 2
        tmp_grid_i = self._agent_grid[0] + action[0]
        tmp_grid_j = self._agent_grid[1] + action[1]

        if(self._cell_walls[tmp_grid_i, tmp_grid_j] < 1):
            self._agent_grid[0] = tmp_grid_i
            self._agent_grid[1] = tmp_grid_j
        self._agent_loc = self.get_cell_center(self._agent_grid)

        reward, done = self.evaluation_rule()
        self.update_observation()
        return reward, done

    def render_observation(self):
        #Paint Observation
        empty_range = 40
        obs_surf = pygame.surfarray.make_surface(self._observation)
        obs_surf = pygame.transform.scale(obs_surf, (self._view_size - 2 * empty_range, self._view_size - 2 * empty_range))
        self._screen.blit(self._obs_logo,(5, 5))
        self._screen.blit(obs_surf, (empty_range, empty_range))

        # Paint the blue edge for observation
        pygame.draw.rect(self._screen, pygame.Color("blue"), 
                (empty_range, empty_range,
                self._view_size - 2 * empty_range, self._view_size - 2 * empty_range), width=1)
        # Paint agent in god view map
        pygame.draw.circle(self._screen, pygame.Color("gray"), 
                ((self._agent_grid[0] + 0.5) * self._render_cell_size + self._view_size, (self._agent_grid[1] + 0.5) * self._render_cell_size),
                int(0.40 * self._render_cell_size), width=0)

    def movement_control(self, keys):
        #Keyboard control cases
        if keys[pygame.K_LEFT]:
            return (-1, 0)
        if keys[pygame.K_RIGHT]:
            return (1, 0)
        if keys[pygame.K_UP]:
            return (0, 1)
        if keys[pygame.K_DOWN]:
            return (0, -1)
        if keys[pygame.K_SPACE]:
            return (0, 0)
        return None

    def update_observation(self):
        #Add the ground first
        #Find Relative Cells
        obs_raw = - numpy.ones(shape=(2 * self.view_grid + 1, 2 * self.view_grid + 1), dtype="float32")
        x_s = self._agent_grid[0] - self.view_grid
        x_e = self._agent_grid[0] + self.view_grid + 1
        y_s = self._agent_grid[1] - self.view_grid
        y_e = self._agent_grid[1] + self.view_grid + 1
        i_s = 0
        i_e = 2 * self.view_grid + 1
        j_s = 0
        j_e = 2 * self.view_grid + 1
        if(x_s < 0):
            i_s = -x_s
            x_s = 0
        if(x_e > self._n):
            i_e -= x_e - self._n
            x_e = self._n
        if(y_s < 0):
            j_s = -y_s
            y_s = 0
        if(y_e > self._n):
            j_e -= y_e - self._n
            y_e = self._n
        # Observation: -1 for walls, > 0 for landmarks, = 0 for grounds
        obs_raw[i_s:i_e, j_s:j_e] = - self._cell_walls[x_s:x_e, y_s:y_e]
        if(self.task_type == "SURVIVAL"):
            obs_raw[i_s:i_e, j_s:j_e] += self._cell_active_landmarks[x_s:x_e, y_s:y_e] + 1 # +1 for cell_active_landmarks in [-1, 0~n]
            obs_raw[self.view_grid, self.view_grid] = self._life
        elif(self.task_type == "NAVIGATION"):
            obs_raw[i_s:i_e, j_s:j_e] += self._cell_landmarks[x_s:x_e, y_s:y_e] + 1

        w,h = obs_raw.shape
        obs_raw = numpy.expand_dims(obs_raw, axis=-1)
        c_w = w // 2
        c_h = h // 2
        wall_rgb = numpy.array([0, 0, 0], dtype="int32")
        empty_rgb = numpy.array([255, 255, 255], dtype="int32")

        self._observation = ((obs_raw == -1).astype("int32") * wall_rgb + 
                (obs_raw == 0).astype("int32") * empty_rgb)
        for i in landmarks_rgb:
            self._observation += (obs_raw == (i + 1)).astype("int32") * landmarks_rgb[i].astype("int32")

        if(self.task_type == "SURVIVAL"):
            # For survival task, the color of the center represents its life value
            f = max(0, int(255 - 128 * self._life))
            self._observation[c_w, c_h] = numpy.asarray([255, f, f], dtype="int32")
        elif(self.task_type == "NAVIGATION"):
            # For navigation task, the color of the center represents the navigation target
            self._observation[c_w, c_h] = landmarks_rgb[self._command]

        # reverse the y axis
        self._observation = self._observation[:,::-1]
