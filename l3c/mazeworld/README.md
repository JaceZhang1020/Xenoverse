# Introduction

MazeWorld is a powerful and efficient simulator for navigating a randomly generated maze. You may use MazeWorld to generate unlimited type of mazes and tasks. We aim to facilitate researches in Meta-Reinforcement-Learning and Artificial General Intelligence.

## Check some of the demonstrations here:

![Demonstration-Keyboard-Control-1](https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/NAVIGATION-1-demo.gif)

![Demonstration-Keyboard-Control-2](https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/NAVIGATION-2-demo.gif)

![Demonstration-Keyboard-Control-3](https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/SURVIVAL-1-demo.gif)

# Quick Start

## Keyboard Demonstrations

You may try MazeWorld with your own keyboard with the following commands:
```bash
python -m l3c.mazeworld.demo.keyboard_play_demo --help
  --maze_type {Discrete2D,Discrete3D,Continuous3D}
  --scale SCALE
  --task_type {SURVIVAL,NAVIGATION}
  --max_steps MAX_STEPS
  --density DENSITY     Density of the walls satisfying that all spaces are connected
  --visibility_2D VISIBILITY_2D     Grids vision range, only valid in 2D mode
  --visibility_3D VISIBILITY_3D     3D vision range, Only valid in 3D mode
  --wall_height WALL_HEIGHT     Only valid in 3D mode
  --cell_size CELL_SIZE     Only valid in 3D mode
  --step_reward STEP_REWARD     Default rewards per-step
  --n_landmarks N_LANDMARKS     Number of landmarks, valid for both SURVIVAL and NAVIGATION task
  --r_landmarks R_LANDMARKS     Average rewards of the landmarks, only valid in SURVIVAL task
  --cd_landmarks CD_LANDMARKS   Refresh interval of landmarks
  --save_replay SAVE_REPLAY     Save the replay trajectory in file
  --verbose VERBOSE
```

## Smart Automatic Agent Demonstration

We implement a smart agent that can do SLAM & Planning in MazeWorlds. You may check the demonstration with the following commands:
```bash
python -m l3c.mazeworld.demo.agent_play_demo --help
  --maze_type {Discrete2D,Discrete3D,Continuous3D}
  --scale SCALE
  --task_type {SURVIVAL,NAVIGATION}
  --max_steps MAX_STEPS
  --density DENSITY     Density of the walls satisfying that all spaces are connected
  --visibility_2D VISIBILITY_2D Grids vision range, only valid in 2D mode
  --visibility_3D VISIBILITY_3D 3D vision range, Only valid in 3D mode
  --wall_height WALL_HEIGHT Only valid in 3D mode
  --cell_size CELL_SIZE Only valid in 3D mode
  --step_reward STEP_REWARD Default rewards per-step
  --n_landmarks N_LANDMARKS Number of landmarks, valid for both SURVIVAL and NAVIGATION task
  --r_landmarks R_LANDMARKS Average rewards of the landmarks, only valid in SURVIVAL task
  --cd_landmarks CD_LANDMARKS Refresh interval of landmarks
  --save_replay SAVE_REPLAY Save the replay trajectory in file
  --memory_keep_ratio MEMORY_KEEP_RATIO Keep ratio of memory when the agent switch from short to long term memory. 1.0 means perfect memory, 0.0 means no memory
  --verbose VERBOSE
```
![Demonstration-Agent-Control](https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/AGENT-1-demo.gif) 

## APIs

Here is an example of creating and running MazeWorld environments

### Creating Maze Environments
```python
import gym
import l3c.mazeworld
from l3c.mazeworld import MazeTaskSampler

maze_env = gym.make("mazeworld-discrete-2D-v0", enable_render=True, task_type="NAVIGATION") # Creating discrete 2D Maze environments with NAVIGATION task
maze_env = gym.make("mazeworld-discrete-3D-v1", enable_render=True, task_type="NAVIGATION") # Creating discrete 3D Maze environments with NAVIGATION task
maze_env = gym.make("mazeworld-continuous-3D-v1", enable_render=True, task_type="SURVIVAL") # Creating continuous 3D Maze environments with SURVIVAL task
```

### Creating Maze Configurations
```python
#Sample a task by specifying the configurations
task = MazeTaskSampler(
    n            = 15,  # Scale of the maze .
    allow_loops  = False,  # If false, there will be no loops in the maze.
    cell_size    = 2.0, # specifying the size of each cell, only valid for 3D mazes
    wall_height  = 3.2, # specifying the height of the wall, only valid for 3D mazes
    agent_height = 1.6, # specifying the height of the agent, only valid for 3D mazes
    step_reward  = -0.01, # specifying punishment in each step in ESCAPE mode, also the reduction of life in each step in SURVIVAL mode
    goal_reward  = 1.0, # specifying reward of reaching the goal, only valid in ESCAPE mode
    initial_life = 1.0, # specifying the initial life of the agent, only valid in SURVIVAL mode
    max_life     = 2.0, # specifying the maximum life of the agent, acquiring food beyond max_life will not lead to growth in life. Only valid in SURVIVAL mode
    landmarks_number = 5, # specifying the number of landmarks in the maze
    landmarks_avg_reward = 0.60, # In SURVIVAL mode, the expected mean reward of each landmarks
    landmarks_refresh_interval = 200, # In SURVIVAL mode, the landmarks refresh in that much steps
    commands_sequence = 200, # In NAVIGATION mode, the tasks include navigating to that much targets
    wall_density = 0.40, # Specifying the wall density in the region, only valid when loops are allowed. E.g. crowd_ratio=0 means no obstacles (except the boundary)
    )
```

### Running Maze Environment Step By Step
```python
#Set the task configuration to the meta environment
maze_env.set_task(task)
maze_env.reset()

#Start the task
done = False
while not done:
    action = maze_env.action_space.sample() 
    observation, reward, done, info = maze_env.step(action)
    maze_env.render()
```


## Writing your own policy

Specifying action with your own policy without relying on keyboards and rendering, check
```bash
l3c/mazeworld/tests/test.py
```

## Using the build-in agents

We implement a smart agent with simulated localization and mapping capbability. The agent does not have all the ground truth information from the beginning, however, it has perfect memory and planning algorithm, and trade-off exploration & exploitation as well, (which can be regarded as the ideal policy). Below is an example to use the smart-agent API
```python
from l3c.mazeworld.agents import SmartSLAMAgent

agent = SmartSLAMAgent(maze_env=maze_env, render=True)
action = agent.step(observation, reward)
```
It's important to be aware that the "render=True" option cannot be utilized concurrently with "enable_render=True" when configuring the maze environment. This is because the visualization may experience interference under such circumstances.Developers can write their own agents following the guidance of agents/agent_base.py

# Installation

#### Remote installation

```bash
pip install l3c[mazeworld]
```

#### Local installation

```bash
git clone https://github.com/FutureAGI/L3C
cd l3c
pip install .[mazeworld]
```

# Explaining the maze type and task type

3 Types of Mazes

- **mazeworld-discrete-2D-v1** <br>
--- Observation space: its surrounding $(2n+1)\times(2n+1)$ (n specified by visibility_2D parameter) grids<br>
--- Action space: 4-D discrete N/S/W/E <br><br>
- **mazeworld-discrete-3D-v0** <br>
--- Observation space: RGB image of 3D first-person view. <br>
--- Action space: 4-D discrete TurnLeft/TurnRight/GoForward/GoBackward. <br><br>
- **mazeworld-continuous-3D-v0** <br>
--- Observation space: RGB image of 3D first-person view.<br>
--- Action space: 2-D continuous [Left/Right, Forward/Backward]<br><br>

2 Types of Tasks:

- **NAVIGATION** mode <br>
--- Reach an target position (goal) as soon as possible, the target is a specific landmark specified by the color bar in observations <br>
--- Each step the agent receives reward of step_reward <br>
--- Acquire goal_reward when reaching the specified target (goal) <br>
--- Episode terminates when finishing all the specified target in commands_sequence <br><br>
- **SURVIVAL** mode <br>
--- The agent begins with initial_life specified by the task <br>
--- Episode terminates when life goes below 0 <br>
--- For each landmark, a unknown random reward is attached <br>
--- When the agent reaches the landmark, it is consumed to recover its life (can be punishments). The landmarks will be refreshed after certain amount of time <br>
--- The life slowly decreases with time, depeding on step_reward <br>
--- The agent's life is shown by a color bar on the top (in 3D mazes) or the color on the center area (in 2D mazes) <br>
