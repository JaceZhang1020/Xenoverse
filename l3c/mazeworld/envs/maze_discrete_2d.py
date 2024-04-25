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
    def __init__(self, visibility_2D=1, task_type="SURVIVAL", max_steps=5000):
        super(MazeCoreDiscrete2D, self).__init__(
                visibility_2D=visibility_2D,
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
        self._observation = self.get_loc_map(self.visibility_2D)

        if(self.task_type == "SURVIVAL"):
            # For survival task, the color of the center represents its life value
            f = max(0, int(255 - 128 * self._life))
            self._observation[self.visibility_2D, self.visibility_2D] = numpy.asarray([255, f, f], dtype="int32")
        elif(self.task_type == "NAVIGATION"):
            # For navigation task, the color of the center represents the navigation target
            self._observation[self.visibility_2D, self.visibility_2D] = landmarks_rgb[self._command]

        self._cell_exposed = numpy.zeros_like(self._cell_walls).astype(numpy.bool)
        self._cell_exposed[(self._agent_grid[0] - self.visibility_2D) : (self._agent_grid[0] + self.visibility_2D + 1), \
                (self._agent_grid[1] - self.visibility_2D) : (self._agent_grid[1] + self.visibility_2D + 1)] = True
