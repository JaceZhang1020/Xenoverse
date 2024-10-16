"""
Gym Environment For Maze3D
"""
import numpy
import gym
import pygame
import time
from numpy import random

def irwin_hall(n:int, N:int):
    # Generate a random categorical distribution with N categories, which n non-zero values
    val = random.rand(n + 1)
    val[0] = 0
    val[1] = 1
    val.sort()
    val = val[1:] - val[:-1]
    random.shuffle(val)
    val = numpy.concatenate((val, numpy.zeros(N - n)))
    random.shuffle(val)
    return val

def reward_sampler(state_space:int, 
                   action_space:int, 
                   reward_sparsity:float,
                   positive_ratio:float,
                   reward_noise_max:float, 
                   reward_type='binomial'):
    reward_mask = numpy.zeros((state_space, action_space))
    while numpy.sum(reward_mask) < 1:
        reward_sparsity = min(reward_sparsity, 1.0)
        reward_sparsity = random.random() * reward_sparsity
        reward_sparsity = max(1 / (state_space*action_space), reward_sparsity) # prevent all-zeros
        reward_mask = random.binomial(1, reward_sparsity, size=(state_space, action_space))
    reward_noise = reward_noise_max * random.rand(state_space, action_space)

    sign_mask = 2.0 * random.binomial(1, positive_ratio, size=(state_space, action_space)) - 1
    reward_matrix = numpy.abs(random.normal(loc=0, scale=1.0, size=(state_space, action_space))) * reward_mask * sign_mask
    if(reward_type == 'binomial'):
        reward_matrix = numpy.clip(reward_matrix, -1.0, 1.0)
    return reward_matrix, reward_noise

def AnyMDPTaskSampler(state_space:int=8,
                 action_space:int=5, 
                 max_reward_sparsity=0.03,
                 max_positive_reward_ratio=0.50,
                 transition_sparsity = 0.10,
                 reward_noise_max=0.30,
                 reward_noise_type=None,
                 seed=None):
    # Sampling Transition Matrix and Reward Matrix based on Irwin-Hall Distribution and Gaussian Distribution
    if(seed is not None):
        random.seed(seed)
    else:
        random.seed(int(time.time() * 32 + 12345) % 65535)

    if(reward_noise_type not in ['binomial', 'normal', None]):
        raise ValueError('Reward type must be either binomial or normal')
    if(reward_noise_type is None):
        reward_noise_type = random.choice(['binomial', 'normal'])

    transition_matrix = numpy.zeros((state_space, action_space, state_space))
    reward_matrix = numpy.zeros((state_space, action_space))
    
    n = min(max(1, int(transition_sparsity * state_space)), state_space)

    for i in range(state_space):
        for j in range(action_space):
            fn = max(1, int(random.random() * n))  # make sparsity more random
            transition = irwin_hall(fn, state_space)
            transition /= sum(transition)
            transition_matrix[i][j] = transition
    
    # Reward Can not be too sparse
    reward_matrix, reward_noise = reward_sampler(state_space, action_space, 
                                                 max_reward_sparsity,
                                                 random.random() * max_positive_reward_ratio,
                                                 reward_noise_max, 
                                                 reward_noise_type)
    
    return {'transition': transition_matrix, 
            'reward': reward_matrix, 
            'reward_noise': reward_noise, 
            'reward_noise_type': reward_noise_type}

def Resampler(task, seed=None, max_reward_sparsity=0.03, max_positive_reward_ratio=0.50, reward_noise_max=0.30):
    # Sampling Transition Matrix and Reward Matrix based on Irwin-Hall Distribution and Gaussian Distribution
    if(seed is not None):
        random.seed(seed)
    else:
        random.seed(int(time.time() * 32 + 12345) % 65535)

    state_space, action_space = task['reward'].shape

    transition_matrix = numpy.copy(task['transition'])
    reward_matrix = numpy.zeros((state_space, action_space))
    
    # Reward Can not be too sparse
    reward_matrix, reward_noise = reward_sampler(state_space, 
                                                 action_space, 
                                                 max_reward_sparsity, 
                                                 random.random() * max_positive_reward_ratio,
                                                 reward_noise_max,
                                                 task['reward_noise_type'])

    return {'transition': transition_matrix, 
            'reward': reward_matrix, 
            'reward_noise': task['reward_noise'], 
            'reward_noise_type': task['reward_noise_type']}