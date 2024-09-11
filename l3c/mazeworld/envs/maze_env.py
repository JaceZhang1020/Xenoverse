"""
Gym Environment For Maze3D
"""
import numpy
import gym
import pygame

from gym import error, spaces, utils
from gym.utils import seeding
from l3c.mazeworld.envs.maze_discrete_2d import MazeCoreDiscrete2D
from l3c.mazeworld.envs.maze_continuous_3d import MazeCoreContinuous3D
from l3c.mazeworld.envs.maze_discrete_3d import MazeCoreDiscrete3D

class MazeWorldEnvBase(gym.Env):
    """
    All Maze World Environments Use This Base Class
    """
    def __init__(self, 
            maze_type,
            enable_render=True,
            render_scale=480,
            max_steps=5000,
            task_type="NAVIGATION",
            ):
        self.maze_type = maze_type
        self.enable_render = enable_render
        self.render_viewsize = render_scale
        self.task_type = task_type

        self.need_reset = True
        self.need_set_task = True

    def set_task(self, task_config):
        self.maze_core.set_task(task_config)
        self.need_set_task = False

    def reset(self):
        if(self.need_set_task):
            raise Exception("Must call \"set_task\" before reset")
        state = self.maze_core.reset()
        if(self.enable_render):
            self.maze_core.render_init(self.render_viewsize)
            self.keyboard_press = pygame.key.get_pressed()
        self.need_reset = False
        self.key_done = False
        return state

    def action_control(self, action):
        raise NotImplementedError("Must implement the action control logic")

    def step(self, action=None):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")

        internal_action = self.action_control(action)
            
        # In keyboard control, process only continues when key is pressed
        info = {"steps": self.maze_core.steps}
        if(internal_action is None):
            return self.maze_core.get_observation(), 0, False, info 
        reward, done = self.maze_core.do_action(internal_action)

        if(done):
            self.need_reset=True

        return self.maze_core.get_observation(), reward, done, info

    def render(self, mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        if(self.enable_render):
            self.key_done, self.keyboard_press = self.maze_core.render_update()

    def get_local_map(self, map_range=8, resolution=(128, 128)):
        """
        Returns the localized god-view map relative to the agent's standpoint
        """
        return self.maze_core.get_local_map(map_rang=map_range, resolution=resolution)

    def get_global_map(self, resolution=(128, 128)):
        """
        Returns the global god-view map
        """
        return self.maze_core.get_global_map(resolution=resolution)

    def get_target_location(self):
        """
        Acquire relative position of the target to the agent
        Return (Distance, Angle)
        """
        if(self.task_type != "NAVIGATION"):
            raise Exception("Only \"NAVIGATION\" task type is supported")
        target_id = self.maze_core._commands_sequence[self.maze_core._commands_sequence_idx]
        target_grid = self.maze_core._landmarks_coordinates[target_id]
        deta_grid = numpy.zeros(shape=(2,), dtype=numpy.float32)
        deta_grid[0] = target_grid[0] - self.maze_core._agent_grid[0]
        deta_grid[1] = target_grid[1] - self.maze_core._agent_grid[1]
        angle = numpy.arctan2(deta_grid[1], deta_grid[0]) - self.maze_core._agent_ori
        if(angle < -numpy.pi):
            angle += 2 * numpy.pi
        elif(angle > numpy.pi):
            angle -= 2 * numpy.pi
        dist = numpy.sqrt(numpy.sum(deta_grid * deta_grid))
        return dist, angle

    def save_trajectory(self, file_name, view_size=480):
        if(not self.enable_render):
            self.maze_core.render_init(view_size)
        self.maze_core.render_trajectory(file_name)


class MazeWorldDiscrete3D(MazeWorldEnvBase):
    def __init__(self, 
            enable_render=True,
            render_scale=480,
            max_steps=5000,
            task_type="NAVIGATION",
            resolution=(320, 320),
            visibility_3D=12.0,
            ):
        super(MazeWorldDiscrete3D, self).__init__(
            "Discrete3D",
            enable_render=enable_render,
            render_scale=render_scale,
            max_steps=max_steps,
            task_type=task_type
        )
        self.maze_core = MazeCoreDiscrete3D(
                resolution_horizon = resolution[0],
                resolution_vertical = resolution[1],
                max_steps = max_steps,
                visibility_3D=visibility_3D,
                task_type = task_type,
                )
        # Turning Left/Right and go backward / forward
        self.action_space = spaces.Discrete(5)
        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Box(low=numpy.zeros(shape=(resolution[0], resolution[1], 3), dtype=numpy.float32), 
                high=numpy.full((resolution[0], resolution[1], 3), 255, dtype=numpy.float32),
                dtype=numpy.float32)
        self.discrete_actions=[(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    def action_control(self, action):
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(100) # 10 FPS
            return self.maze_core.movement_control(self.keyboard_press)
        else:
            return self.discrete_actions[action]

class MazeWorldContinuous3D(MazeWorldEnvBase):
    def __init__(self, 
            enable_render=True,
            render_scale=480,
            max_steps = 5000,
            task_type = "NAVIGATION",
            resolution=(320, 320),
            visibility_3D=12.0,
            ):
        super(MazeWorldContinuous3D, self).__init__(
            "Continuous3D",
            enable_render=enable_render,
            render_scale=render_scale,
            max_steps=max_steps,
            task_type=task_type
        )
        self.maze_core = MazeCoreContinuous3D(
                resolution_horizon = resolution[0],
                resolution_vertical = resolution[1],
                max_steps = max_steps,
                visibility_3D=visibility_3D,
                task_type = task_type
                )

        # Turning Left/Right and go backward / forward
        self.action_space = spaces.Box(low=numpy.array([-1.0, -1.0]), 
                high=numpy.array([1.0, 1.0]), dtype=numpy.float32)
        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Box(low=numpy.zeros(shape=(resolution[0], resolution[1], 3), dtype=numpy.float32), 
                high=numpy.full((resolution[0], resolution[1], 3), 256, dtype=numpy.float32),
                dtype=numpy.float32)

    def action_control(self, action):
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(20) # 50 FPS
            tr, ws = self.maze_core.movement_control(self.keyboard_press)
        else:
            tr = action[0]
            ws = action[1]
        if(tr is None or ws is None):
            return None
        return [tr, ws]

class MazeWorldDiscrete2D(MazeWorldEnvBase):
    def __init__(self,
            enable_render=True,
            render_scale=480,
            max_steps = 5000,
            task_type = "NAVIGATION",
            visibility_2D = 1):
        super(MazeWorldDiscrete2D, self).__init__(
            "Discrete2D",
            enable_render=enable_render,
            render_scale=render_scale,
            max_steps=max_steps,
            task_type=task_type
            )
        self.maze_core = MazeCoreDiscrete2D(visibility_2D=visibility_2D, max_steps=max_steps, task_type=task_type)

        # Go EAST/WEST/SOUTH/NORTH
        self.action_space = spaces.Discrete(4)
        # observation is the x, y coordinate of the grid
        n_w = 2 * visibility_2D + 1
        self.observation_space = spaces.Box(low=numpy.zeros(shape=(n_w, n_w, 3), dtype=numpy.float32), 
                high=numpy.full((n_w, n_w, 3), 255, dtype=numpy.float32),
                dtype=numpy.float32)

        self.discrete_actions=[(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


    def action_control(self, action):
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(100) # 10 FPS
            return self.maze_core.movement_control(self.keyboard_press)
        else:
            return self.discrete_actions[action]
