import gym
import sys
import argparse
import time
import l3c.mazeworld
from l3c.mazeworld import MazeTaskSampler
from l3c.mazeworld.agents import SmartSLAMAgent

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Playing the maze world demo with your keyboard')
    parser.add_argument('--maze_type', type=str, choices=["Discrete2D", "Discrete3D", "Continuous3D"], default="Continuous3D")
    parser.add_argument('--scale', type=int, default=15)
    parser.add_argument('--task_type', type=str, choices=["SURVIVAL", "NAVIGATION"], default="NAVIGATION")
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--density', type=float, default=0.30, help="Density of the walls satisfying that all spaces are connected")
    parser.add_argument('--visibility_2D', type=int, default=1, help="Grids vision range, only valid in 2D mode")
    parser.add_argument('--visibility_3D', type=float, default=12, help="3D vision range, Only valid in 3D mode")
    parser.add_argument('--wall_height', type=float, default=3.2, help="Only valid in 3D mode")
    parser.add_argument('--cell_size', type=float, default=2.0, help="Only valid in 3D mode")
    parser.add_argument('--step_reward', type=float, default=-0.01, help="Default rewards per-step")
    parser.add_argument('--n_landmarks', type=int, default=5, help="Number of landmarks, valid for both SURVIVAL and NAVIGATION task")
    parser.add_argument('--r_landmarks', type=float, default=0.50, help="Average rewards of the landmarks, only valid in SURVIVAL task")
    parser.add_argument('--cd_landmarks', type=int, default=200, help="Refresh interval of landmarks")
    parser.add_argument('--save_replay', type=str, default=None, help="Save the replay trajectory in file")
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()

    if(args.maze_type == "Discrete2D"):
        maze_env = gym.make("mazeworld-discrete-2D-v1", enable_render=False, max_steps=args.max_steps, visibility_2D=args.visibility_2D, task_type=args.task_type)
    elif(args.maze_type == "Discrete3D"):
        maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=False, max_steps=args.max_steps, visibility_3D=args.visibility_3D, task_type=args.task_type)
    elif(args.maze_type == "Continuous3D"):
        maze_env = gym.make("mazeworld-continuous-3D-v1", enable_render=False, max_steps=args.max_steps, visibility_3D=args.visibility_3D, task_type=args.task_type)
    else:
        raise Exception("No such maze type %s"%args.maze_type)

    task = MazeTaskSampler(n=args.scale, allow_loops=True, 
            wall_density=args.density,
            cell_size=args.cell_size,
            wall_height=args.wall_height,
            step_reward=args.step_reward,
            landmarks_number=args.n_landmarks,
            landmarks_refresh_interval=args.cd_landmarks,
            landmarks_avg_reward=args.r_landmarks,
            verbose=True)
    maze_env.set_task(task)

    agent = SmartSLAMAgent(maze_env=maze_env, render=True)

    observation = maze_env.reset()
    done=False
    sum_reward = 0
    reward = 0

    while not done:
        action = agent.step(observation, reward)
        observation, reward, done, _ = maze_env.step(action)
        sum_reward += reward
        if(args.verbose):
            print("Instant r = %.2f, Accumulate r = %.2f" % (reward, sum_reward))
        if(maze_env.key_done):
            break
    print("Episode is over! You got %.2f score."%sum_reward)

    if(args.save_replay is not None):
        maze_env.save_trajectory(args.save_replay)
