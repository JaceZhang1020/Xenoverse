"""
Gym Environment For Any MDP
"""
import numpy
import gym
import pygame
import random as rnd
from numpy import random

from gym import error, spaces, utils
from gym.utils import seeding
from l3c.utils import pseudo_random_seed

class AnyMDPEnv(gym.Env):
    def __init__(self, max_steps):
        """
        Pay Attention max_steps might be reseted by task settings
        """
        self.observation_space = spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(1,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.max_steps = max_steps
        self.task_set = False

    def set_task(self, task):
        for key in task:
            setattr(self, key, task[key])
        # 定义无界的 observation_space
        self.observation_space = gym.spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(self.state_dim,), dtype=numpy.float32)
        # 定义 action_space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=numpy.float32)

        self.task_set = True
        self.need_reset = True

    def reset(self):
        if(not self.task_set):
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        random.seed(pseudo_random_seed())

        loc, noise = rnd.choice(self.born_loc)
        self._inner_state = loc + noise * random.normal(size=self.ndim)
        self._state = self.observation_map(self._inner_state)
        return self._state, {"steps": self.steps}

    def step(self, action):
        if(self.need_reset or not self.task_set):
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        assert numpy.shape(action) == (self.action_dim,)

        ### update inner state
        inner_action = self.action_map(numpy.array(action))
        next_inner_state = self._inner_state + inner_action * self.action_weight

        ### basic reward
        reward = self.average_cost
        done = False

        if(self.mode == 'sgoal'):
            # Static Goal (with reset)
            for gs, d, gr in self.sgoal_loc:
                if(numpy.linalg.norm(next_inner_state - gs) < d):
                    done = True
                    reward += gr
                    break
        else:
            # Dynamic Goal
            dgoal_loc = self.calculate_loc(self.dgoal_loc, self.steps)
            dgoal_dist = numpy.linalg.norm(next_inner_state - dgoal_loc)
            if(dgoal_dist < self.dgoal_potential[0]):
                reward += self.dgoal_potential[1] * (1 - dgoal_dist / self.dgoal_potential[0])

        for gs, d, gr in self.pitfalls_loc:
            if(numpy.linalg.norm(next_inner_state - gs) < d):
                done = True
                reward += gr
                break
        for dir, gr in self.line_potential_energy:
            deta_s = next_inner_state - self._inner_state
            reward += gr * numpy.dot(deta_s, dir)
        for pt, d, gr in self.point_potential_energy:
            pre_s = numpy.linalg.norm(self._inner_state - pt)
            next_s = numpy.linalg.norm(next_inner_state - pt)
            reward += gr * (next_s - pre_s)

        self.steps += 1
        info = {"steps": self.steps}

        self._inner_state = next_inner_state
        self._state = self.observation_map(self._inner_state)

        done = (self.steps >= self.max_steps or done)
        if(done):
            self.need_reset = True
        return self._state, reward, done, info
    
    def calculate_loc(self, loc, steps):
        # Sample a cos nx + b cos ny
        for n, k in loc:
            loc += k[0] * numpy.cos(n * self.steps) + k[1] * numpy.sin(n * self.steps)
        return loc / len(loc)
    
    @property
    def state(self):
        return numpy.copy(self._state)
    
    @property
    def inner_state(self):
        return numpy.copy(self._inner_state)