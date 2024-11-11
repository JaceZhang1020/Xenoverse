import gym
import numpy
from numpy import random
from numba import njit
from l3c.anymdp.solver import update_value_matrix


class AnyMDPSolverOpt(object):
    """
    Solver for AnyMDPEnv with Bellman Equation and Value Iteration
    Suppose to know the ground truth of the environment
    """
    def __init__(self, env, gamma=0.98):
        if(not env.task_set):
            raise Exception("AnyMDPEnv is not initialized by 'set_task', must call set_task first")
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.transition_matrix = env.transition_matrix
        self.reward_matrix = env.reward_matrix
        self.value_matrix = numpy.zeros((self.n_states, self.n_actions))
        self.gamma = gamma
        self.q_solver()

    def q_solver(self, gamma=0.99):
        self.value_matrix = update_value_matrix(self.reward_matrix, self.transition_matrix, gamma, self.value_matrix)

    def policy(self, state):
        return numpy.argmax(self.value_matrix[state])