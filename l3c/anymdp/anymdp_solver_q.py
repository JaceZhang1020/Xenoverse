import gym
import numpy
from numpy import random
from l3c.anymdp.anymdp_solver_opt import update_value_matrix


class AnyMDPSolverQ(object):
    """
    Solver for AnyMDPEnv with Q-Learning
    """
    def __init__(self, env, gamma=0.98, alpha=0.01, c=0.02):
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
        self.sa_vistied = numpy.zeros((self.n_states, self.n_actions))
        self.gamma = gamma
        self.alpha = alpha
        self.step = 0
        self._c = c / (1.0 - self.gamma)


    def learner(self, s, a, ns, r, done):
        if(done):
            target = r
        else:
            target = r + self.gamma * max(self.value_matrix[ns])
        error = target - self.value_matrix[s][a]
        self.value_matrix[s][a] += self.alpha * error
        self.sa_vistied[s][a] += 1
        self.step += 1


    def policy(self, state):
        # Apply UCB-H exploration strategy
        values = self._c * numpy.sqrt(numpy.log(self.step + 1) / numpy.clip(self.sa_vistied[state], 1.0, 1.0e+8))  + self.value_matrix[state]
        return numpy.argmax(values)