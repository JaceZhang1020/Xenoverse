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

def AnyMDPTaskSampler(state_space:int=8, 
                 action_space:int=5, 
                 reward_sparsity=0.50,
                 transition_sparsity = 0.50,
                 seed=None):
    # Sampling Transition Matrix and Reward Matrix based on Irwin-Hall Distribution and Gaussian Distribution
    if(seed is not None):
        random.seed(seed)
    else:
        random.seed(int(time.time() * 32 + 12345) % 65535)

    transition_matrix = numpy.zeros((state_space, action_space, state_space))
    reward_matrix = numpy.zeros((state_space, action_space))
    
    n = min(max(1, int(transition_sparsity * state_space)), state_space)
    for i in range(state_space):
        for j in range(action_space):
            transition = irwin_hall(n, state_space)
            transition /= sum(transition)
            transition_matrix[i][j] = transition
    
    # Reward Can not be too sparse
    reward_sparsity = max(6 / (state_space*action_space), reward_sparsity)
    reward_sparsity = min(reward_sparsity, 1.0)
    reward_mask = random.binomial(1, reward_sparsity, size=(state_space, action_space))
    reward_matrix = numpy.clip(
        random.normal(loc=0, scale=1.0, size=(state_space, action_space)),
        -1.0, 1.0) * reward_mask
    
    return {'transition': transition_matrix, 'reward': reward_matrix}

def Resampler(task, seed=None, reward_sparsity=0.5):
    # Sampling Transition Matrix and Reward Matrix based on Irwin-Hall Distribution and Gaussian Distribution
    if(seed is not None):
        random.seed(seed)
    else:
        random.seed(int(time.time() * 32 + 12345) % 65535)

    state_space, action_space = task['reward'].shape

    transition_matrix = numpy.copy(task['transition'])
    reward_matrix = numpy.zeros((state_space, action_space))
    
    # Reward Can not be too sparse
    reward_sparsity = max(6 / (state_space*action_space), reward_sparsity)
    reward_sparsity = min(reward_sparsity, 1.0)
    reward_mask = random.binomial(1, reward_sparsity, size=(state_space, action_space))
    reward_matrix = numpy.clip(
        random.normal(loc=0, scale=1.0, size=(state_space, action_space)),
        -1.0, 1.0) * reward_mask
    
    return {'transition': transition_matrix, 'reward': reward_matrix}