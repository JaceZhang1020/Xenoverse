"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
import time
from pygame import font
from numpy import random as npyrnd
from numpy.linalg import norm
from l3c.mazeworld.envs.ray_caster_utils import landmarks_color

class MazeBase(object):
    def __init__(self, **kw_args):
        for k in kw_args:
            self.__dict__[k] = kw_args[k]
        pygame.init()

    def set_task(self, task_config):
        # initialize textures
        self._cell_walls = numpy.copy(task_config.cell_walls)
        self._cell_texts = task_config.cell_texts
        self._start = task_config.start
        self._n = numpy.shape(self._cell_walls)[0]
        self._cell_landmarks = task_config.cell_landmarks
        self._cell_size = task_config.cell_size
        self._wall_height = task_config.wall_height
        self._agent_height = task_config.agent_height
        self._step_reward = task_config.step_reward
        self._goal_reward = task_config.goal_reward
        self._landmarks_rewards = task_config.landmarks_rewards
        self._landmarks_coordinates = task_config.landmarks_coordinates
        self._landmarks_refresh_interval = task_config.landmarks_refresh_interval
        self._commands_sequence = task_config.commands_sequence
        self._max_life = task_config.max_life
        self._initial_life = task_config.initial_life
        self._int_max = 100000000

        assert self._agent_height < self._wall_height and self._agent_height > 0, "the agent height must be > 0 and < wall height"
        assert self._cell_walls.shape == self._cell_texts.shape, "the dimension of walls must be equal to textures"
        assert self._cell_walls.shape[0] == self._cell_walls.shape[1], "only support square shape"

    def refresh_command(self):
        """
        Update the command for selecting the target to navigate
        At the same time, update the instant_rewards
        Valid only for ``NAVIGATION`` mode
        """
        if(self.task_type is not "NAVIGATION"):
            return
        if(self._command is not None):
            x,y = self._landmarks_coordinates[self._command]
            self._instant_rewards[x, y] = 0.0

        self._commands_sequence_idx += 1
        if(self._commands_sequence_idx > len(self._commands_sequence) - 1):
            return True
        self._command = self._commands_sequence[self._commands_sequence_idx]
        x,y = self._landmarks_coordinates[self._command]
        self._instant_rewards[x,y] = self._goal_reward
        return False

    def reach_goal(self):
        g_x, g_y = self._landmarks_coordinates[self._command]
        goal = ((g_x == self._agent_grid[0]) and (g_y == self._agent_grid[1]))
        return goal

    def refresh_landmark_attr(self):
        """
        Refresh the landmarks
            refresh the instant rewards in SURVIVAL mode
            refresh the view in SURVIVAL mode
            No need to refresh for NAVIGATION mode
        """
        if(self.task_type is not "SURVIVAL"):
            return
        self._instant_rewards = numpy.zeros_like(self._cell_landmarks, dtype="float32")
        self._cell_active_landmarks = numpy.copy(self._cell_landmarks)
        idxes = numpy.argwhere(self._landmarks_refresh_countdown <= self._landmarks_refresh_interval)
        for idx, in idxes:
            x,y = self._landmarks_coordinates[idx]
            self._cell_active_landmarks[x,y] = -1
        for i, (x,y) in enumerate(self._landmarks_coordinates):
            if(self._cell_active_landmarks[x,y] > -1):
                self._instant_rewards[(x,y)] = self._landmarks_rewards[i]

    def reset(self):
        self._agent_grid = numpy.copy(self._start)
        self._agent_loc = self.get_cell_center(self._start)
        self._agent_trajectory = [numpy.copy(self._agent_grid)]

        # Maximum w and h in the space
        self._size = self._n * self._cell_size

        # Valid in 3D
        self._agent_ori = 0.0
        self._instant_rewards = numpy.zeros_like(self._cell_landmarks, dtype="float32")
        self._landmarks_refresh_countdown = numpy.full(self._landmarks_rewards.shape, self._int_max)

        # Initialization related to tasks
        if(self.task_type == "SURVIVAL"):
            self._life = self._initial_life
            for i, (x,y) in enumerate(self._landmarks_coordinates):
                self._instant_rewards[(x,y)] = self._landmarks_rewards[i]
            self.refresh_landmark_attr()
        elif(self.task_type == "NAVIGATION"):
            self._commands_sequence_idx = -1
            self._command = None
            self.refresh_command()
        else:
            raise Exception("No such task type: %s" % self.task_type)

        self.update_observation()
        self.steps = 0
        return self.get_observation()

    def evaluation_rule(self):
        self.steps += 1
        self._agent_trajectory.append(numpy.copy(self._agent_grid))
        agent_grid_idx = tuple(self._agent_grid)

        # Landmarks refresh countdown update
        self._landmarks_refresh_countdown -= 1
        idxes = numpy.argwhere(self._landmarks_refresh_countdown <= 0)
        for idx, in idxes:
            self._landmarks_refresh_countdown[idx] = self._int_max

        # Refresh landmarks in SURVIVAL mode, including call back those have been resumed
        if(self.task_type == "SURVIVAL"):
            self.refresh_landmark_attr()

        reward = self._instant_rewards[agent_grid_idx] + self._step_reward

        if(self.task_type == "SURVIVAL"):
            self._life = min(reward + self._life, self._max_life)
            landmark_id = self._cell_landmarks[agent_grid_idx]
            if(landmark_id >= 0 and self._landmarks_refresh_countdown[landmark_id] > self._landmarks_refresh_interval):
                 self._landmarks_refresh_countdown[landmark_id] = self._landmarks_refresh_interval
            done = self._life < 0.0 or self.episode_steps_limit()
        elif(self.task_type == "NAVIGATION"):
            done = False
            if(self.reach_goal()):
                done = self.refresh_command()
            done = done or self.episode_steps_limit()

        return reward, done

    def do_action(self, action):
        raise NotImplementedError()

    def render_init(self, view_size):
        """
        Initialize a God View With Landmarks
        """
        font.init()
        self._font = font.SysFont("Arial", 18)

        #Initialize the agent drawing
        self._render_cell_size = view_size / self._n
        self._view_size = view_size

        self._obs_logo = self._font.render("Observation", 0, pygame.Color("red"))

        self._screen = pygame.Surface((2 * view_size, view_size))
        self._screen = pygame.display.set_mode((2 * view_size, view_size))
        pygame.display.set_caption("RandomMazeRender - GodView")
        self._surf_god = pygame.Surface((view_size, view_size))
        self._surf_god.fill(pygame.Color("white"))
        it = numpy.nditer(self._cell_walls, flags=["multi_index"])
        for _ in it:
            x,y = it.multi_index
            landmarks_id = self._cell_landmarks[x,y]
            if(self._cell_walls[x,y] > 0):
                pygame.draw.rect(self._surf_god, pygame.Color("black"), (x * self._render_cell_size, y * self._render_cell_size,
                        self._render_cell_size, self._render_cell_size), width=0)
        logo_god = self._font.render("GodView", 0, pygame.Color("red"))
        self._surf_god.blit(logo_god,(view_size - 90, 5))

    def render_dynamic_map(self, scr, offset):
        """
        Cover landmarks with white in case it is not refreshed
        """
        for landmarks_id, (x,y) in enumerate(self._landmarks_coordinates):
            if(self._landmarks_refresh_countdown[landmarks_id] > self._landmarks_refresh_interval):
                pygame.draw.rect(scr, landmarks_color(landmarks_id), 
                        (x * self._render_cell_size + offset[0], y * self._render_cell_size + offset[1],
                        self._render_cell_size, self._render_cell_size), width=0)
        if(self.task_type is "SURVIVAL"):
            txt_life = self._font.render("Life: %f"%self._life, 0, pygame.Color("green"))
            scr.blit(txt_life,(offset[0] + 90, offset[1] + 10))

    def render_observation(self):
        """
        Need to implement the logic for observation painting
        """
        raise NotImplementedError()

    def render_update(self):
        #Paint God View
        self._screen.blit(self._surf_god, (self._view_size, 0))
        self.render_dynamic_map(self._screen, (self._view_size, 0))

        #Paint Agent and Observation
        self.render_observation()

        pygame.display.update()
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        keys = pygame.key.get_pressed()
        return done, keys

    def render_trajectory(self, file_name, additional=None):
        # Render god view with record on the trajectory
        if(additional is not None):
            aw, ah = additional["surfaces"][0].get_width(),additional["surfaces"][0].get_height()
        else:
            aw, ah = (0, 0)

        traj_screen = pygame.Surface((self._view_size + aw, max(self._view_size, ah)))
        traj_screen.fill(pygame.Color("white"))
        traj_screen.blit(self._surf_god, (0, 0))

        pygame.draw.rect(traj_screen, pygame.Color("red"), 
                (self._agent_grid[0] * self._render_cell_size, self._agent_grid[1] * self._render_cell_size,
                self._render_cell_size, self._render_cell_size), width=0)
        if(self.task_type == "SURVIVAL"):
            self.render_dynamic_map(traj_screen, (0, 0))

        for i in range(len(self._agent_trajectory)-1):
            p = self._agent_trajectory[i]
            n = self._agent_trajectory[i+1]
            p = [(p[0] + 0.5) * self._render_cell_size, (p[1] + 0.5) *  self._render_cell_size]
            n = [(n[0] + 0.5) * self._render_cell_size, (n[1] + 0.5) *  self._render_cell_size]
            pygame.draw.line(traj_screen, pygame.Color("red"), p, n, width=3)

        # paint some additional surfaces where necessary
        if(additional != None):
            for i in range(len(additional["surfaces"])):
                traj_screen.blit(additional["surfaces"][i], (self._view_size, 0))
                pygame.image.save(traj_screen, file_name.split(".")[0] + additional["file_names"][i] + ".png")
        else:
            pygame.image.save(traj_screen, file_name)

    def episode_steps_limit(self):
        return self.steps > self.max_steps-1

    def get_cell_center(self, cell):
        p_x = cell[0] * self._cell_size + 0.5 * self._cell_size
        p_y = cell[1] * self._cell_size + 0.5 * self._cell_size
        return [p_x, p_y]

    def get_loc_grid(self, loc):
        p_x = int(loc[0] / self._cell_size)
        p_y = int(loc[1] / self._cell_size)
        return [p_x, p_y]

    def movement_control(self, keys):
        """
        Implement the movement control logic, or ''agent dynamics''
        """
        raise NotImplementedError()

    def update_observation(self):
        """
        Update the observation, which is used for returning the state when ''get_observation''
        """
        raise NotImplementedError()

    def get_observation(self):
        return numpy.copy(self._observation)
