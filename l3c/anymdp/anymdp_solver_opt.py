import gym
import numpy
from numpy import random
from numba import njit


@njit(cache=True)
def update_value_matrix(r_mat, t_mat, gamma, vm):
    diff = 1.0
    cur_vm = numpy.copy(vm)
    ns, na = r_mat.shape
    while diff > 1.0e-4:
        old_vm = numpy.copy(cur_vm)
        for s in range(ns):
            for a in range(na):
                exp_q = 0.0
                for sn in range(ns):
                    exp_q += t_mat[s,a,sn] * numpy.max(cur_vm[sn])
                cur_vm[s,a] = r_mat[s,a] + gamma * exp_q
        diff = numpy.sqrt(numpy.mean((old_vm - cur_vm)**2))
    return cur_vm

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