import sys
import numpy
from queue import Queue
from copy import deepcopy
from l3c.mazeworld.envs.dynamics import PI
from l3c.mazeworld.envs.maze_env import MazeWorldDiscrete2D, MazeWorldDiscrete3D, MazeWorldContinuous3D

class AgentBase(object):
    """
    Base class for agents
    Use this as parent to create new rule based agents
    """
    def __init__(self, **kwargs):
        self.render = False
        for k in kwargs:
            self.__dict__[k] = kwargs[k]
        if("maze_env" not in kwargs):
            raise Exception("Must use maze_env as arguments")
        self.task_type = self.maze_env.maze_core.task_type
        self.maze_type = self.maze_env.maze_type

        # Initialize information
        self._god_info = 1 - self.maze_env.maze_core._cell_walls + self.maze_env.maze_core._cell_landmarks
        if(self.task_type == "SURVIVAL"):
            self._landmarks_rewards = self.maze_env.maze_core._landmarks_rewards
        self._landmarks_coordinates = self.maze_env.maze_core._landmarks_coordinates
        self._step_reward = self.maze_env.maze_core._step_reward
        self._nx, self._ny = self._god_info.shape
        self.neighbors = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self._landmarks_visit = dict()

        # Render
        if(self.render):
            self.render_init()

    def render_init(self):
        raise NotImplementedError()

    def update_common_info(self):
        if(self.task_type == "NAVIGATION"):
            self._command = self.maze_env.maze_core._command
        if(self.task_type == "SURVIVAL"):
            self._life = self.maze_env.maze_core._life
        self._mask_info = self.maze_env.maze_core._cell_exposed
        self._agent_ori = (2.0 * self.maze_env.maze_core._agent_ori / PI)
        self._agent_loc = self.maze_env.maze_core._agent_loc
        self._cur_grid = deepcopy(self.maze_env.maze_core._agent_grid)
        self._cur_grid_float = deepcopy(self.maze_env.maze_core.get_loc_grid_float(self.maze_env.maze_core._agent_loc))
        self._landmarks_cd = []
        for cd in self.maze_env.maze_core._landmarks_refresh_countdown:
            if(cd < self.maze_env.maze_core._landmarks_refresh_interval):
                self._landmarks_cd.append(cd)
            else:
                self._landmarks_cd.append(0)
        self._exposed_info = self._god_info + self._mask_info - 1
        lid = self._god_info[self._cur_grid[0], self._cur_grid[1]]
        if(lid > 0):
            self._landmarks_visit[lid - 1] = 0

    def policy(self, observation, r):
        raise NotImplementedError()

    def render_update(self):
        raise NotImplementedError()

    def step(self, observation, r):
        self.update_common_info()
        action = self.policy(observation, r)
        if(self.render):
            self.render_update()
        return action
