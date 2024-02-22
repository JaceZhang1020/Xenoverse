import sys
import numpy
from queue import Queue
from copy import deepcopy
from l3c.mazeworld.envs.dynamics import PI

class AgentBase(object):
    def __init__(self, **kwargs):
        self._render = False
        for k in kwargs:
            self.__dict__[k] = kwargs[k]
        if("maze_type" not in kwargs):
            raise Exception("Must specify maze type: Discrete2D/Discrete3D/Continuous3D")
        if(self._render):
            self.render_init()
        self.neighbors = [(-1, 0), (1, 0), (0, 1), (0, -1)]
        self._landmarks_visit = dict()

    def render_init(self):
        raise NotImplementedError()

    def update_common_info(self, env):
        self.task_type = env.maze_core.task_type
        self._god_info = 1 - env.maze_core._cell_walls + env.maze_core._cell_landmarks
        self._mask_info = env.maze_core._cell_exposed
        self._landmarks_pos = env.maze_core._landmarks_coordinates
        if(self.task_type == "NAVIGATION"):
            self._command = env.maze_core._command
        if(self.task_type == "SURVIVAL"):
            self._landmarks_rewards = env.maze_core._landmarks_rewards
        self._agent_ori = (2.0 * env.maze_core._agent_ori / PI)
        self._agent_ori = int(self._agent_ori) % 4 + self._agent_ori - int(self._agent_ori)
        self._step_reward = env.maze_core._step_reward
        self._cur_grid = deepcopy(env.maze_core._agent_grid)
        self._cur_grid_float = deepcopy(env.maze_core.get_loc_grid_float(env.maze_core._agent_loc))
        self._nx, self._ny = self._god_info.shape
        self._landmarks_cd = []
        for cd in env.maze_core._landmarks_refresh_countdown:
            if(cd < env.maze_core._landmarks_refresh_interval):
                self._landmarks_cd.append(cd)
            else:
                self._landmarks_cd.append(0)

        self._exposed_info = self._god_info + self._mask_info - 1

        lid = self._god_info[self._cur_grid[0], self._cur_grid[1]]
        if(lid > 0):
            self._landmarks_visit[lid - 1] = 0

    def policy(self, observation, r, env):
        raise NotImplementedError()

    def render_update(self):
        raise NotImplementedError()

    def step(self, observation, r, env):
        self.update_common_info(env)
        action = self.policy(observation, r, env)
        if(self._render):
            self.render_update()
        return action
