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
from l3c.mazeworld.envs.dynamics import PI, PI_2, PI_4, PI2d, vector_move_with_collision
from l3c.mazeworld.envs.ray_caster_utils import maze_view
from l3c.mazeworld.envs.maze_task import MAZE_TASK_MANAGER
from l3c.mazeworld.envs.maze_base import MazeBase
from l3c.mazeworld.envs.ray_caster_utils import landmarks_rgb, landmarks_rgb_arr, paint_agent_arrow

class MazeCoreContinuous3D(MazeBase):
    #Read Configurations
    def __init__(
            self,
            collision_dist=0.20, #collision distance
            visibility_3D=12.0, #agent vision range
            fol_angle = 0.6 * PI, #Field of View
            resolution_horizon = 320, #resolution in horizontal
            resolution_vertical = 320, #resolution in vertical
            max_steps = 5000,
            task_type = "SURVIVAL"
        ):
        super(MazeCoreContinuous3D, self).__init__(
                collision_dist = collision_dist,
                visibility_3D = visibility_3D,
                fol_angle = fol_angle,
                resolution_horizon = resolution_horizon,
                resolution_vertical = resolution_vertical,
                task_type = task_type,
                max_steps = max_steps
                )

    def reset(self):
        if(self.task_type == "SURVIVAL"):
            #add the life bar
            self._lifebar_start_x = 0.10 * self.resolution_vertical
            self._lifebar_start_y = 0.10 * self.resolution_vertical
            self._lifebar_l = 0.80 * self.resolution_vertical
            self._lifebar_w = 0.05 * self.resolution_horizon
        elif(self.task_type == "NAVIGATION"):
            #add the navigation guidance bar
            self._navbar_start_x = 0.25 * self.resolution_vertical
            self._navbar_start_y = 0.10 * self.resolution_vertical
            self._navbar_l = 0.50 * self.resolution_vertical
            self._navbar_w = 0.05 * self.resolution_horizon
        else:
            raise Exception("No such task type: %s" % task_type)
        return super(MazeCoreContinuous3D, self).reset()

    def do_action(self, turn_rate, walk_speed, dt=0.10):
        turn_rate = numpy.clip(turn_rate, -1, 1) * PI
        walk_speed = numpy.clip(walk_speed, -1, 1)
        self._agent_ori, self._agent_loc = vector_move_with_collision(
                self._agent_ori, self._agent_loc, turn_rate, walk_speed, dt, 
                self._cell_walls, self._cell_size, self.collision_dist)
        self._agent_grid = self.get_loc_grid(self._agent_loc)
        reward, done = self.evaluation_rule()
        self.update_observation()
        return reward, done

    def render_init(self, view_size):
        super(MazeCoreContinuous3D, self).render_init(view_size)
        self._pos_conversion = self._render_cell_size / self._cell_size
        self._ori_size = 0.60 * self._pos_conversion

    def render_observation(self):
        # Paint Observation
        view_obs_surf = pygame.transform.scale(self._obs_surf, (self._view_size, self._view_size))
        self._screen.blit(view_obs_surf, (0, 0))

        # Paint God-view
        agent_pos = numpy.array(self._agent_loc) * self._pos_conversion
        paint_agent_arrow(self._screen, pygame.Color("gray"), (self._view_size, 0), (agent_pos[0], agent_pos[1]), self._agent_ori, 
                0.4 * self._pos_conversion, 0.5 * self._pos_conversion)

    def movement_control(self, keys):
        #Keyboard control cases
        turn_rate = None
        walk_speed = None
        if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN]:
            turn_rate = 0.0
            walk_speed = 0.0
            if keys[pygame.K_LEFT]:
                turn_rate -= 0.3
            if keys[pygame.K_RIGHT]:
                turn_rate += 0.3
            if keys[pygame.K_UP]:
                walk_speed += 1.0
            if keys[pygame.K_DOWN]:
                walk_speed -= 1.0
        if keys[pygame.K_SPACE]:
            turn_rate = 0.0
            walk_speed = 0.0
        return turn_rate, walk_speed

    def update_observation(self):
        if(self.task_type == "SURVIVAL"):
            self._observation, cell_exposed = maze_view(self._agent_loc, self._agent_ori, self._agent_height, 
                    self._cell_walls, self._cell_active_landmarks, self._cell_texts, self._cell_size, MAZE_TASK_MANAGER.grounds,
                    MAZE_TASK_MANAGER.ceil, self._wall_height, 1.0, self.visibility_3D, 0.20, 
                    self.fol_angle, self.resolution_horizon, self.resolution_vertical, landmarks_rgb_arr)
            lifebar_l = self._life / self._max_life * self._lifebar_l
            start_x = int(self._lifebar_start_x)
            start_y = int(self._lifebar_start_y)
            end_x = int(self._lifebar_start_x + lifebar_l)
            end_y = int(self._lifebar_start_y + self._lifebar_w)
            self._observation[start_x:end_x, start_y:end_y, 0] = 255
            self._observation[start_x:end_x, start_y:end_y, 1] = 0
            self._observation[start_x:end_x, start_y:end_y, 2] = 0
        elif(self.task_type == "NAVIGATION"):
            self._observation, cell_exposed = maze_view(self._agent_loc, self._agent_ori, self._agent_height, 
                    self._cell_walls, self._cell_landmarks, self._cell_texts, self._cell_size, MAZE_TASK_MANAGER.grounds,
                    MAZE_TASK_MANAGER.ceil, self._wall_height, 1.0, self.visibility_3D, 0.20, 
                    self.fol_angle, self.resolution_horizon, self.resolution_vertical, landmarks_rgb_arr)
            start_x = int(self._navbar_start_x)
            start_y = int(self._navbar_start_y)
            end_x = int(self._navbar_start_x + self._navbar_l)
            end_y = int(self._navbar_start_y + self._navbar_w)
            self._observation[start_x:end_x, start_y:end_y] = landmarks_rgb[self._command]
        self._obs_surf = pygame.surfarray.make_surface(self._observation)
        self._cell_exposed = numpy.logical_or(self._cell_exposed, cell_exposed)

    def get_observation(self):
        return numpy.copy(self._observation)
