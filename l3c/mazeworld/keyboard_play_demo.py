import gym
import sys
import l3c.mazeworld
from l3c.mazeworld import MazeTaskSampler
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Playing the maze world demo with your keyboard')
    parser.add_argument('--type', type=str, choices=["Discrete2D", "Discrete3D", "Continuous3D"], default="Continuous3D")
    parser.add_argument('--scale', type=int, default=15)
    parser.add_argument('--rule', type=str, choices=["SURVIVAL", "NAVIGATION"], default="NAVIGATION")
    parser.add_argument('--max_steps', type=int, default=1000000)
    parser.add_argument('--density', type=float, default=0.30, help="Density of the walls satisfying that all spaces are connected")
    parser.add_argument('--obs_grids', type=int, default=1, help="Observation of neighborhood grids, only valid in 2D mode")
    parser.add_argument('--vision_range', type=float, default=100, help="Only valid in 3D mode")
    parser.add_argument('--wall_height', type=float, default=3.2, help="Only valid in 3D mode")
    parser.add_argument('--cell_size', type=float, default=2.0, help="Only valid in 3D mode")
    parser.add_argument('--n_landmarks', type=int, default=5, help="Number of landmarks, valid for both SURVIVAL and NAVIGATION task")
    parser.add_argument('--r_landmarks', type=float, default=0.50, help="Average rewards of the landmarks, only valid in SURVIVAL task")
    parser.add_argument('--save_replay', type=str, default=None, help="Save the replay trajectory in file")
    parser.add_argument('--verbose', type=bool, default=False)

    args = parser.parse_args()

    if(args.type == "Discrete2D"):
        maze_env = gym.make("mazeworld-discrete-2D-v1", max_steps=args.max_steps, view_grid=args.obs_grids, task_type=args.rule)
    elif(args.type == "Discrete3D"):
        maze_env = gym.make("mazeworld-discrete-3D-v1", max_steps=args.max_steps, max_vision_range=args.vision_range, task_type=args.rule)
    elif(args.type == "Continuous3D"):
        maze_env = gym.make("mazeworld-continuous-3D-v1", max_steps=args.max_steps, max_vision_range=args.vision_range, task_type=args.rule)
    else:
        raise Exception("No such maze world type %s"%args.type)

    task = MazeTaskSampler(n=args.scale, allow_loops=True, 
            crowd_ratio=args.density,
            cell_size=args.cell_size,
            wall_height=args.wall_height,
            landmarks_number=args.n_landmarks,
            landmarks_avg_reward=args.r_landmarks,
            verbose=True)
    maze_env.set_task(task)

    maze_env.reset()
    done=False
    sum_reward = 0

    while not done:
        maze_env.render()
        state, reward, done, _ = maze_env.step(None)
        sum_reward += reward
        if(args.verbose):
            print("Instant r = %.2f, Accumulate r = %.2f" % (reward, sum_reward))
        if(maze_env.key_done):
            break
    print("Episode is over! You got %.2f score."%sum_reward)

    if(args.save_replay is not None):
        maze_env.save_trajectory(args.save_replay)
