import gym
import numpy
from numpy import random
from numba import njit
import networkx as nx


@njit(cache=True)
def update_value_matrix(r_mat, t_mat, gamma, vm, max_iteration=-1):
    diff = 1.0
    cur_vm = numpy.copy(vm)
    ns, na = r_mat.shape
    iteration = 0

    while diff > 1.0e-4 and (
            (max_iteration < 0) or 
            (max_iteration > iteration and max_iteration > 1) or
            (iteration < 1 and random.random() < max_iteration)):
        iteration += 1
        old_vm = numpy.copy(cur_vm)
        for s in range(ns):
            for a in range(na):
                exp_q = 0.0
                for sn in range(ns):
                    exp_q += t_mat[s,a,sn] * numpy.max(cur_vm[sn])
                cur_vm[s,a] = r_mat[s,a] + gamma * exp_q
        diff = numpy.sqrt(numpy.mean((old_vm - cur_vm)**2))
    return cur_vm

def check_transition_quality(transition):
    if(transition is None):
        return False
    # acquire state - to - state distribution
    in_dist = numpy.sum(transition, axis=1)

    in_graph = nx.DiGraph()
    nodes = range(in_dist.shape[0])
    in_graph.add_nodes_from(nodes)
    for i in range(in_dist.shape[0]):
        for j in range(in_dist.shape[1]):
            if in_dist[i, j] != 0:
                in_graph.add_edge(i, j, weight=in_dist[i, j])

    if(nx.is_strongly_connected(in_graph)):
        return True
    else:
        print("AnyMDP: Sampling transition graph not strongly connected, need resample...")
        return False

def check_reward_quality(transition, reward):
    if(transition is None or reward is None):
        return False
    ns, na = reward.shape
    vm = update_value_matrix(reward, transition, 0.98, numpy.zeros((ns, na), dtype=float), max_iteration=2)
    if(numpy.std(vm) > 1.0e-2 and numpy.min(vm) < 0 and numpy.max(vm) > 0):
        return True
    else:
        print("AnyMDP: Sampling task has trivial value function, need resample...")
        return False

def check_task_quality(task):
    """
    Check the quality of the task
    Requiring: Q value is diverse enough
               State is connected enough (no isolated state)
    """

    if(task is None):
        return False
    if(check_transition_quality(task["transition"]) and 
            check_reward_quality(task["transition"], task["reward"])):
        return True
    return False


