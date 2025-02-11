"""
Any MDP Task Sampler
"""
import numpy
import gym
import pygame
import time
from numpy import random
from copy import deepcopy
from l3c.utils import pseudo_random_seed, RandomMLP
from l3c.anymdp.solver import check_task_trans, check_task_rewards


def sample_action_mapping(task):
    ndim = task['ndim']
    action_dim = task['action_dim']
    action_map = RandomMLP(action_dim, ndim)

    return {"action_map": action_map}

def sample_observation_mapping(task):
    ndim = task['ndim']
    observation_dim = task['state_dim']
    observation_map = RandomMLP(ndim, observation_dim, n_hidden_layers=random.randint(1, ndim-1))
    return {
        "observation_map": observation_map
    }

def sample_born_loc(task):
    born_loc_num = random.randint(1, 10)
    born_loc = [(random.uniform(-1, 1, size=(task['ndim'],)), 
                random.exponential(1, size=(task['ndim'],))) for i in range(born_loc_num)]
    return {"born_loc": born_loc}

def sample_sgoal(task):
    sgoal_num = random.randint(0, 10)
    sgoal_loc = []
    existing_loc = [loc for loc, _ in task['born_loc']]
    for i in range(sgoal_num):
        min_dist = 0.0
        while min_dist < 0.5:
            sloc = random.uniform(-1, 1, size=(task['ndim'],))
            # calculate the distance between the goal and the born location
            min_dist = 10000   
            for loc in existing_loc:
                dist = numpy.linalg.norm(sloc-loc[0])
                if(dist < min_dist):
                    min_dist = dist
        sink_range = random.uniform(0.02, 0.2)
        reward = random.exponential(5.0)
        sgoal_loc.append((numpy.copy(sloc), sink_range, reward))
        existing_loc.append(numpy.copy(sloc))
    return {"sgoal_loc": sgoal_loc}

def sample_pitfalls(task):
    pf_num = random.randint(20, 100)
    pitfalls_loc = []
    existing_loc = [loc[0] for loc in task['born_loc']]
    if(task['mode']=='sgoal'):
        existing_loc.extend([loc[0] for loc in task['sgoal_loc']])
    for i in range(pf_num):
        min_dist = 0.0
        while min_dist < 0.5:
            pfloc = random.uniform(-1, 1, size=(task['ndim'],))
            # calculate the distance between the goal and the born location
            min_dist = 10000   
            for loc in existing_loc:
                dist = numpy.linalg.norm(pfloc-loc[0])
                if(dist < min_dist):
                    min_dist = dist
        sink_range = random.uniform(0.05, 0.20)
        reward = - random.exponential(1.0)
        pitfalls_loc.append((numpy.copy(pfloc), sink_range, reward))
        existing_loc.append(numpy.copy(pfloc))
    return {"pitfalls_loc": pitfalls_loc}

def sample_line_potential_energy(task):
    num = random.randint(0, 4)
    line_potential_energy = []
    for i in range(num):
        dir = random.uniform(-1, 1, size=(task['ndim'],))
        dir /= numpy.linalg.norm(dir)
        line_potential_energy.append((dir, random.exponential(0.50)))
    return {"line_potential_energy": line_potential_energy}

def sample_point_potential_energy(task):
    num = random.randint(0, 4)
    point_potential_energy = []
    for i in range(num):
        pts = random.uniform(-1, 1, size=(task['ndim'],))
        point_potential_energy.append((pts, random.exponential(0.2), -random.exponential(0.50)))
    return {"point_potential_energy": point_potential_energy}

def sample_dgoal(task, max_order=16, max_item=3):
    num = random.randint(0, 4)
    item_num = random.randint(0, max_item + 1)
    dgoal_loc = [(0, random.normal(size=(task['ndim'], 2)))]
    for j in range(item_num):
        # Sample a cos nx + b cos ny
        order = random.randint(1, max_order + 1)
        factor = random.normal(size=(task['ndim'], 2))
        dgoal_loc.append((order, factor))
    r_range = random.uniform(0.05, 0.40)
    dr = random.exponential(1.0)
    return {"dgoal_loc": dgoal_loc, "dgoal_potential": (r_range, dr)}

def AnyMDPv2TaskSampler(state_dim:int=256,
                 action_dim:int=256,
                 seed=None,
                 verbose=False):
    # Sampling Transition Matrix and Reward Matrix based on Irwin-Hall Distribution and Gaussian Distribution
    # Task:
    # mode: static goal or moving goal
    # ndim: number of inner dimensions
    # born_loc: born location and noise
    # sgoal_loc: static goal location, range of sink, and reward
    # pf_loc: pitfall location, range of sink, and reward
    # line_potential_energy: line potential energy specified by direction and detal_V
    # point_potential_energy: point potential energy specified by location and detal_V


    if(seed is not None):
        random.seed(seed)
    else:
        random.seed(pseudo_random_seed())
    
    task = dict()
    mode = random.choice(["sgoal", "dgoal"])

    task["mode"] = mode
    task["state_dim"] = state_dim
    task["action_dim"] = action_dim
    task["ndim"] = random.randint(3, 33) # At most 32-dimensional space
    task["max_steps"] = random.randint(60, 500) # At most 10-dimensional space
    task["action_weight"] = random.uniform(5.0e-3, 0.10, size=(task['ndim'],))
    task["average_cost"] = - random.exponential(0.01) * random.choice([0, 1])
    task["transition_noise"] = max(0, random.normal(scale=5.0e-3))
    task["reward_noise"] = max(0, random.normal(scale=5.0e-3))

    task.update(sample_observation_mapping(task)) # Observation Model
    task.update(sample_action_mapping(task)) # Action Mapping
    task.update(sample_born_loc(task)) # Born Location

    if(task['mode'] == 'sgoal') :
        task.update(sample_sgoal(task)) # Static Goal Location
    else:
        task.update(sample_dgoal(task)) # Moving Goal Location
    task.update(sample_pitfalls(task)) # Pitfall Location
    task.update(sample_line_potential_energy(task)) # Potential Energy
    task.update(sample_point_potential_energy(task)) # Potential Energy

    return task