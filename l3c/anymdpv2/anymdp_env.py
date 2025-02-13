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
from copy import deepcopy

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
        self.observation_space = gym.spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(self.state_dim,), dtype=float)
        # 定义 action_space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=float)

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
        if(self.mode == 'multi'):
            self.available_goal = deepcopy(self.goal_loc)
        return self._state, {"steps": self.steps}
    
    def near_born_loc(self):
        for loc, noise in self.born_loc:
            dist = numpy.linalg.norm(self._inner_state - loc)
            if(dist < noise * 3):
                return True
        return False

    def calculate_loc(self, loc, steps):
        # Sample a cos nx + b cos ny
        g_loc = numpy.zeros(self.ndim)
        for n, k in loc:
            g_loc += k[:, 0] * numpy.cos(0.01 * n * self.steps) + k[:, 1] * numpy.sin(0.01 * n * self.steps)
        return g_loc / len(loc)
    
    def goal_reward_static(self, ns):
        min_dist = numpy.inf
        reward = 0
        done = False
        for gs, d, gr in self.goal_loc:
            dist = numpy.linalg.norm(ns - gs)
            if(dist < d):
                reward += gr
                done = True
                break
        return reward, done
    
    def goal_reward_multi(self, ns):
        min_dist = numpy.inf
        reward = 0
        for gs, d, gr in self.available_goal:
            dist = numpy.linalg.norm(ns - gs)
            if(dist < d):
                reward += gr
                self.available_goal.remove((gs, d, gr))
                break
        return reward, False
    
    def goal_reward_dynamic(self, ns):
        goal_loc = self.calculate_loc(self.goal_loc, self.steps)
        goal_dist = numpy.linalg.norm(ns - goal_loc)
        reward = 0
        if(goal_dist < self.goal_potential[0]):
            reward = self.goal_potential[1] * (1 - goal_dist / self.goal_potential[0])
        goal_loc = goal_loc
        return reward, False
    
    def goal_reward_consist(self, ns):
        reward = self.goal_reward(numpy.concat([self.inner_state, ns], dtype=numpy.float32))[0]
        return reward, False

    def step(self, action):
        if(self.need_reset or not self.task_set):
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        assert numpy.shape(action) == (self.action_dim,)

        ### update inner state
        inner_deta = self.action_map(numpy.concat([self._inner_state, numpy.array(action)], axis=-1))
        next_inner_state = (self._inner_state + 
            inner_deta * self.action_weight + 
            self.transition_noise * random.normal(size=(self.ndim,)))

        ### basic reward
        reward = self.average_cost + self.reward_noise * random.normal()
        done = False
        if(self.mode == 'static'):
            reward, done = self.goal_reward_static(next_inner_state)
        elif(self.mode == 'multi'):
            reward, done = self.goal_reward_multi(next_inner_state)
            if(len(self.available_goal) == 0):
                done = True
        elif(self.mode == 'dynamic'):
            reward, done = self.goal_reward_dynamic(next_inner_state)
        elif(self.mode == 'consis'):
            reward, done = self.goal_reward_consist(next_inner_state)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        ### Calculate Pitfalls
        if(done is not True):
            pitfall_penalty = 0
            switch = self.pitfalls_switch(next_inner_state)
            risk = numpy.sum((switch < 0.0).astype('float32')) / numpy.size(switch)
            if(risk > self.risk_limit and not self.near_born_loc()):
                # Can not have pitfalls near born loc
                reward += self.pitfalls_penalty
                done = True

        ### Calculate Potential Energy
        if(self.use_potential):
            reward += (self.potential_energy(next_inner_state)[0] - self.potential_energy(self._inner_state)[0])

        self.steps += 1
        info = {"steps": self.steps}

        self._inner_state = next_inner_state

        self._state = self.observation_map(self._inner_state)

        done = (self.steps >= self.max_steps or done)
        if(done):
            self.need_reset = True
        return self._state, reward, done, info
    
    @property
    def state(self):
        return numpy.copy(self._state)
    
    @property
    def inner_state(self):
        # 复制内部状态
        return numpy.copy(self._inner_state)