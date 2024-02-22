#!/usr/bin/env python
# coding=utf8
# File: test.py
import gym
import sys
import l3c.mazeworld
import time
from l3c.mazeworld import MazeTaskSampler
from l3c.mazeworld.agents import SmartSLAMAgent
from numpy import random

def test_agent_maze(n=15,
        maze_type="Discrete2D", 
        max_steps=1000, 
        task_type="NAVIGATION",
        density=0.40,
        n_landmarks=10,
        r_landmarks=0.40):
    print("\n\n--------\n\nStart test for maze_type = %s, task_type = %s, n = %d, steps = %s...\n\n"%(maze_type, task_type, n, max_steps))
    if(maze_type == "Discrete2D"):
        maze_env = gym.make("mazeworld-discrete-2D-v1", enable_render=False, max_steps=max_steps, task_type=task_type)
    elif(maze_type == "Discrete3D"):
        maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=max_steps, task_type=task_type)
    elif(maze_type == "Continuous3D"):
        maze_env = gym.make("mazeworld-continuous-3D-v1", enable_render=False, max_steps=max_steps, task_type=task_type)
    else:
        raise Exception("No such maze world type %s"%task_type)

    task = MazeTaskSampler(n=n, allow_loops=True, 
            wall_density=density,
            landmarks_number=n_landmarks,
            landmarks_avg_reward=r_landmarks,
            verbose=False)

    maze_env.set_task(task)

    # Must intialize agent after reset
    agent = SmartSLAMAgent(maze_env=maze_env, render=False)

    done=False
    observation = maze_env.reset()
    sum_reward = 0
    reward = 0
    while not done:
        action = agent.step(observation, reward)
        observation, reward, done, _ = maze_env.step(action)
        #print("action:", action, ", reward:", reward)
        time.sleep(0.01)
        sum_reward += reward
    print("...Test Finishes. Get score %f, for maze_type = %s task_type = %s, n = %d, steps = %s\n\n---------\n\n"%(sum_reward, maze_type, task_type, n, max_steps))

if __name__=="__main__":
    for n in [9, 15, 25]:
        for task_type in ["NAVIGATION", "SURVIVAL"]:
            for maze_type in ["Discrete2D", "Discrete3D", "Continuous3D"]:
                n_landmarks=random.randint(5,10)
                density=(random.random() + 0.50) * 0.30
                test_agent_maze(n=n, task_type=task_type, density=density, maze_type=maze_type, n_landmarks=n_landmarks)
    print("\n\nCongratulations!!!\n\nAll Tests Have Been Passed\n\n")
