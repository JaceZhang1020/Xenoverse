"""
Gym Environment For Maze3D
"""
import numpy
import gym
import pygame
from numpy import random

from gym import error, spaces, utils
from gym.utils import seeding

class AnyMDPEnv(gym.Env):
    def __init__(self, max_steps:int=5000):
        self.max_steps = max_steps
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.task_set = False

    def set_task(self, task_config):
        self.transition_matrix = task_config["transition"]
        self.reward_matrix = task_config["reward"]

        ns1, na1, ns2 = self.transition_matrix.shape
        ns3, na2 = self.reward_matrix.shape

        assert ns1 == ns2 == ns3, "Transition matrix and reward matrix must have the same number of states"
        assert na1 == na2, "Transition matrix and reward matrix must have the same number of actions"
        assert ns1 > 2, "State space must be at least 2"
        assert na1 > 2, "Action space must be at least 2"

        self.observation_space = spaces.Discrete(ns1)
        self.action_space = spaces.Discrete(na1)

        self.n_states = ns1
        self.n_actions = na1
        self.task_set = True
        self.need_reset = True

    def reset(self):
        if(not self.task_set):
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        self._state = random.randint(0, self.n_states - 1)
        return self._state, {"steps": self.steps}

    def step(self, action):
        if(self.need_reset or not self.task_set):
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        assert action < self.n_actions, "Action must be less than the number of actions"
        transition_gt = self.transition_matrix[self._state, action]
        reward_gt = self.reward_matrix[self._state, action]
        next_state = random.choice(self.n_states, p=transition_gt)
        if(reward_gt > 0):
            reward = random.binomial(1, reward_gt)
        else:
            reward = - random.binomial(1, abs(reward_gt))

        info = {"steps": self.steps, "transition_gt": transition_gt, "reward_gt": reward_gt}
        self.steps += 1
        self._state = next_state
        done = (self.steps >= self.max_steps)
        if(done):
            self.need_reset = True

        return next_state, reward, done, info
