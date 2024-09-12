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
    def __init__(self, visibility_2D=5, task_type="SURVIVAL", resolution=(128, 128), max_steps=5000):
        super(MazeCoreDiscrete2D, self).__init__(
                visibility_2D=visibility_2D,
                task_type=task_type,
                max_steps=max_steps
                )
        self.resolution = resolution

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
        obs_surf, obs_arr = self.get_local_map(map_range=self.visibility_2D, resolution=self.resolution)

        if(self.task_type == "SURVIVAL"):
            f = max(0, int(255 - 128 * self._life))
        else:
            f = 0

        center_size = 0.04 * (self.resolution[0] + self.resolution[1])
        pygame.draw.circle(obs_surf, pygame.Color(255, f, f), (self.resolution[0]/2, self.resolution[1]/2), center_size)
        self._observation = pygame.surfarray.array3d(obs_surf)

        vis_grids = int(self.visibility_2D / self._cell_size)
        self._cell_exposed = numpy.zeros_like(self._cell_walls).astype(bool)
        self._cell_exposed[(self._agent_grid[0] - vis_grids) : (self._agent_grid[0] + vis_grids + 1), \
                (self._agent_grid[1] - vis_grids) : (self._agent_grid[1] + vis_grids + 1)] = True
