import gym
import numpy
from numpy import random

class AnyMDPSolver(object):
    """
    Solver for AnyMDPEnv with Bellman Equation and Value Iteration
    """
    def __init__(self, env):
        if(not env.task_set):
            raise Exception("AnyMDPEnv is not initialized by 'set_task', must call set_task first")
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.transition_matrix = env.transition_matrix
        self.reward_matrix = env.reward_matrix
        self.value_matrix = numpy.zeros((self.n_states, self.n_actions))
        self.q_solver()

    def q_solver(self, gamma=0.99):
        diff = 1.0
        while diff > 1.0e-4:
            old_value_matrix = numpy.copy(self.value_matrix)
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    self.value_matrix[s,a] = self.reward_matrix[s,a] \
                        + gamma * numpy.sum(self.transition_matrix[s,a] * numpy.max(self.value_matrix, axis=-1))
            diff = numpy.sqrt(numpy.mean((old_value_matrix - self.value_matrix)**2))

    def policy(self, state):
        return numpy.argmax(self.value_matrix[state])