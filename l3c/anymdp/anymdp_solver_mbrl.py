import gym
import numpy
from numpy import random
from l3c.anymdp.anymdp_solver_opt import update_value_matrix


class AnyMDPSolverMBRL(object):
    """
    Solver for AnyMDPEnv with Model-based Reinforcement Learning
    """
    def __init__(self, env, gamma=0.98, c=0.02):
        """
        The constructor for the class AnyMDPSolverQ
        The exploration strategy is controlled by UCB-H with c as its hyperparameter. Increasing c will encourage exploration
        Simulation of the ideal policy when the ground truth is not known
        """
        if(not env.task_set):
            raise Exception("AnyMDPEnv is not initialized by 'set_task', must call set_task first")
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.transition_matrix = env.transition_matrix
        self.reward_matrix = env.reward_matrix
        self.value_matrix = numpy.zeros((self.n_states, self.n_actions))
        self.est_r = numpy.zeros((self.n_states, self.n_actions))
        self.est_r_cnt = numpy.zeros((self.n_states, self.n_actions))
        self.est_t = numpy.ones((self.n_states, self.n_actions, self.n_states))
        self.gamma = gamma
        self._c = c / (1.0 - self.gamma)
        self.step = 0


    def learner(self, s, a, ns, r, done):
        # Update the environment model estimation
        if(self.est_r_cnt[s,a] < 1):
            self.est_r[s,a] = r
        else:
            self.est_r[s,a] += (r - self.est_r[s,a])/self.est_r_cnt[s,a]
        self.est_r_cnt[s,a]+= 1
        self.est_t[s,a,ns] += 1
        est_t = self.est_t / numpy.sum(self.est_t, axis=-1, keepdims=True)

        # Update the value matrix
        self.value_matrix = update_value_matrix(self.est_r, est_t, self.gamma, self.value_matrix, max_iteration=3)

    def policy(self, state):
        # Apply UCB-H exploration strategy
        self.step += 1
        values = self._c * numpy.sqrt(numpy.log(self.step) / numpy.clip(self.est_r_cnt[state], 1.0, 1.0e+8))  + self.value_matrix[state]
        return numpy.argmax(values)