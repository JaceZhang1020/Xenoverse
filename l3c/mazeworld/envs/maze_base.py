"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
import time
from pygame import font
from numpy import random as npyrnd
from numpy.linalg import norm
from l3c.mazeworld.envs.ray_caster_utils import landmarks_rgb, landmarks_color
from l3c.mazeworld.envs.dynamics import PI
from l3c.mazeworld.envs.maze_task import MAZE_TASK_MANAGER
from l3c.mazeworld.envs.ray_caster_utils import paint_agent_arrow

class MazeBase(object):
    def __init__(self, **kw_args):
        for k in kw_args:
            self.__dict__[k] = kw_args[k]
        pygame.init()

    def set_task(self, task_config):
        # initialize textures
        self._cell_walls = numpy.copy(task_config.cell_walls)
        self._cell_texts = task_config.cell_texts
        self._start = task_config.start
        self._n = numpy.shape(self._cell_walls)[0]
        self._cell_landmarks = task_config.cell_landmarks
        self._cell_size = task_config.cell_size
        self._wall_height = task_config.wall_height
        self._agent_height = task_config.agent_height
        self._step_reward = task_config.step_reward
        self._goal_reward = task_config.goal_reward
        self._landmarks_rewards = task_config.landmarks_rewards
        self._landmarks_coordinates = task_config.landmarks_coordinates
        self._landmarks_refresh_interval = task_config.landmarks_refresh_interval
        self._commands_sequence = task_config.commands_sequence
        self._max_life = task_config.max_life
        self._initial_life = task_config.initial_life
        self._int_max = 100000000

        assert self._agent_height < self._wall_height and self._agent_height > 0, "the agent height must be > 0 and < wall height"
        assert self._cell_walls.shape == self._cell_texts.shape, "the dimension of walls must be equal to textures"
        assert self._cell_walls.shape[0] == self._cell_walls.shape[1], "only support square shape"

    def refresh_command(self):
        """
        Update the command for selecting the target to navigate
        At the same time, update the instant_rewards
        Valid only for ``NAVIGATION`` mode
        """
        if(self.task_type != "NAVIGATION"):
            return
        if(self._command is not None):
            x,y = self._landmarks_coordinates[self._command]
            self._instant_rewards[x, y] = 0.0

        self._commands_sequence_idx += 1
        if(self._commands_sequence_idx > len(self._commands_sequence) - 1):
            return True
        self._command = self._commands_sequence[self._commands_sequence_idx]
        x,y = self._landmarks_coordinates[self._command]
        self._instant_rewards[x,y] = self._goal_reward
        return False

    def reach_goal(self):
        g_x, g_y = self._landmarks_coordinates[self._command]
        goal = ((g_x == self._agent_grid[0]) and (g_y == self._agent_grid[1]))
        return goal

    def refresh_landmark_attr(self):
        """
        Refresh the landmarks
            refresh the instant rewards in SURVIVAL mode
            refresh the view in SURVIVAL mode
            No need to refresh for NAVIGATION mode
        """
        if(self.task_type != "SURVIVAL"):
            return
        self._instant_rewards = numpy.zeros_like(self._cell_landmarks, dtype="float32")
        self._cell_active_landmarks = numpy.copy(self._cell_landmarks)
        idxes = numpy.argwhere(self._landmarks_refresh_countdown <= self._landmarks_refresh_interval)
        for idx, in idxes:
            x,y = self._landmarks_coordinates[idx]
            self._cell_active_landmarks[x,y] = -1
        for i, (x,y) in enumerate(self._landmarks_coordinates):
            if(self._cell_active_landmarks[x,y] > -1):
                self._instant_rewards[(x,y)] = self._landmarks_rewards[i]

    def reset(self):
        self._agent_grid = numpy.copy(self._start)
        self._agent_loc = self.get_cell_center(self._start)
        self._agent_trajectory = [numpy.copy(self._agent_grid)]

        # Record all observed cells
        self._cell_exposed = numpy.zeros_like(self._cell_walls).astype(numpy.bool)

        # Maximum w and h in the space
        self._size = self._n * self._cell_size

        # Valid in 3D
        self._agent_ori = 0.0
        self._instant_rewards = numpy.zeros_like(self._cell_landmarks, dtype="float32")
        self._landmarks_refresh_countdown = numpy.full(self._landmarks_rewards.shape, self._int_max)

        self._text_surf = []
        for text in MAZE_TASK_MANAGER.grounds:
            self._text_surf.append(pygame.surfarray.make_surface(text))

        # Initialization related to tasks
        if(self.task_type == "SURVIVAL"):
            self._life = self._initial_life
            for i, (x,y) in enumerate(self._landmarks_coordinates):
                self._instant_rewards[(x,y)] = self._landmarks_rewards[i]
            self.refresh_landmark_attr()
        elif(self.task_type == "NAVIGATION"):
            self._commands_sequence_idx = -1
            self._command = None
            self.refresh_command()
        else:
            raise Exception("No such task type: %s" % self.task_type)

        self.update_observation()
        self.steps = 0
        return self.get_observation()

    def evaluation_rule(self):
        self.steps += 1
        self._agent_trajectory.append(numpy.copy(self._agent_grid))
        agent_grid_idx = tuple(self._agent_grid)

        # Landmarks refresh countdown update
        self._landmarks_refresh_countdown -= 1
        idxes = numpy.argwhere(self._landmarks_refresh_countdown <= 0)
        for idx, in idxes:
            self._landmarks_refresh_countdown[idx] = self._int_max

        # Refresh landmarks in SURVIVAL mode, including call back those have been resumed
        if(self.task_type == "SURVIVAL"):
            self.refresh_landmark_attr()

        if(self.task_type == "SURVIVAL"):
            reward = self._instant_rewards[agent_grid_idx] # In survival mode, the step reward is not counted
            self._life += self._step_reward
            self._life = min(self._instant_rewards[agent_grid_idx] + self._life, self._max_life)
            landmark_id = self._cell_landmarks[agent_grid_idx]
            if(landmark_id >= 0 and self._landmarks_refresh_countdown[landmark_id] > self._landmarks_refresh_interval):
                 self._landmarks_refresh_countdown[landmark_id] = self._landmarks_refresh_interval
            done = self._life < 0.0 or self.episode_steps_limit()
        elif(self.task_type == "NAVIGATION"):
            reward = self._instant_rewards[agent_grid_idx] + self._step_reward
            done = False
            if(self.reach_goal()):
                done = self.refresh_command()
            done = done or self.episode_steps_limit()

        return reward, done

    def do_action(self, action):
        raise NotImplementedError()

    def render_init(self, view_size):
        """
        Initialize a God View With Landmarks
        """
        font.init()
        self._font = font.SysFont("Arial", 18)

        #Initialize the agent drawing
        self._render_cell_size = view_size / self._n
        self._view_size = view_size

        self._obs_logo = self._font.render("Observation", 0, pygame.Color("red"))

        self._screen = pygame.Surface((3 * view_size, view_size))
        self._screen = pygame.display.set_mode((3 * view_size, view_size))
        pygame.display.set_caption("MazeWorldRender")
        self._surf_god = pygame.Surface((view_size, view_size))
        self._surf_god.fill(pygame.Color("white"))
        self._surf_lm = pygame.Surface((view_size, view_size))
        self._surf_lm.fill(pygame.Color("grey"))
        it = numpy.nditer(self._cell_walls, flags=["multi_index"])
        for _ in it:
            x,y = it.multi_index
            landmarks_id = self._cell_landmarks[x,y]
            if(self._cell_walls[x,y] > 0):
                pygame.draw.rect(self._surf_god, pygame.Color("black"), (x * self._render_cell_size, y * self._render_cell_size,
                        self._render_cell_size, self._render_cell_size), width=0)
        logo_god = self._font.render("God View - Invisible To Agent", 0, pygame.Color("red"))
        self._surf_god.blit(logo_god,(90, 5))
        logo_loc = self._font.render("Local Map - Invisible To Agent", 0, pygame.Color("red"))
        self._surf_lm.blit(logo_loc,(90, 5))

    def render_godview_dyna(self, scr, offset):
        """
        Cover landmarks with white in case it is not refreshed
        """
        for landmarks_id, (x,y) in enumerate(self._landmarks_coordinates):
            if(self._landmarks_refresh_countdown[landmarks_id] > self._landmarks_refresh_interval):
                pygame.draw.rect(scr, landmarks_color(landmarks_id), 
                        (x * self._render_cell_size + offset[0], y * self._render_cell_size + offset[1],
                        self._render_cell_size, self._render_cell_size), width=0)
        if(self.task_type == "SURVIVAL"):
            txt_life = self._font.render("Life: %f"%self._life, 0, pygame.Color("green"))
            scr.blit(txt_life,(offset[0] + 90, offset[1] + 10))

    def render_map(self, scr, offset):
        """
        Cover landmarks with white in case it is not refreshed
        """
        empty_range = 32
        lm_surf, _ = self.get_global_map((512, 512))
        lm_surf = pygame.transform.scale(lm_surf, (self._view_size - 2 * empty_range, self._view_size - 2 * empty_range))
        scr.blit(lm_surf, (offset[0] + empty_range, offset[1] + empty_range))

    def render_observation(self):
        """
        Need to implement the logic for observation painting
        """
        raise NotImplementedError()

    def render_update(self):
        #Paint God View
        self._screen.blit(self._surf_god, (self._view_size, 0))
        self._screen.blit(self._surf_lm, (2 * self._view_size, 0))
        self.render_godview_dyna(self._screen, (self._view_size, 0))
        self.render_map(self._screen, (2 * self._view_size, 0))

        #Paint Agent and Observation
        self.render_observation()

        pygame.display.update()
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        keys = pygame.key.get_pressed()
        return done, keys

    def render_trajectory(self, file_name, additional=None):
        # Render god view with record on the trajectory
        if(additional is not None):
            aw, ah = (additional["surfaces"][0].get_width(),additional["surfaces"][0].get_height())
        else:
            aw, ah = (0, 0)

        traj_screen = pygame.Surface((self._view_size + aw, max(self._view_size, ah)))
        traj_screen.fill(pygame.Color("white"))
        traj_screen.blit(self._surf_god, (0, 0))

        if(self.task_type == "SURVIVAL"):
            self.render_godview_dyna(traj_screen, (0, 0))

        for i in range(len(self._agent_trajectory)-1):
            factor = i / len(self._agent_trajectory)
            noise = (factor - 0.5) * 0.10
            p = self._agent_trajectory[i]
            n = self._agent_trajectory[i+1]
            p = [(p[0] + 0.5 + noise) * self._render_cell_size, (p[1] + 0.5 + noise) *  self._render_cell_size]
            n = [(n[0] + 0.5 + noise) * self._render_cell_size, (n[1] + 0.5 + noise) *  self._render_cell_size]
            pygame.draw.line(traj_screen, pygame.Color(int(255 * factor), int(255 * (1 - factor)), 0, 255), p, n, width=2)

        for landmarks_id, (x,y) in enumerate(self._landmarks_coordinates):
            pygame.draw.rect(traj_screen, landmarks_color(landmarks_id), 
                    (x * self._render_cell_size, y * self._render_cell_size,
                    self._render_cell_size, self._render_cell_size), width=0)

        # paint some additional surfaces where necessary
        if(additional != None):
            for i in range(len(additional["surfaces"])):
                traj_screen.blit(additional["surfaces"][i], (self._view_size, 0))
                pygame.image.save(traj_screen, file_name.split(".")[0] + additional["file_names"][i] + ".png")
        else:
            pygame.image.save(traj_screen, file_name)

    def episode_steps_limit(self):
        return self.steps > self.max_steps-1

    def get_cell_center(self, cell):
        p_x = cell[0] * self._cell_size + 0.5 * self._cell_size
        p_y = cell[1] * self._cell_size + 0.5 * self._cell_size
        return [p_x, p_y]

    def get_loc_grid(self, loc):
        p_x = int(loc[0] / self._cell_size)
        p_y = int(loc[1] / self._cell_size)
        return [p_x, p_y]

    def get_loc_grid_float(self, loc):
        p_x = (loc[0] / self._cell_size)
        p_y = (loc[1] / self._cell_size)
        return [p_x, p_y]

    def movement_control(self, keys):
        """
        Implement the movement control logic, or ''agent dynamics''
        """
        raise NotImplementedError()

    def update_observation(self):
        """
        Update the observation, which is used for returning the state when ''get_observation''
        """
        raise NotImplementedError()

    def get_observation(self):
        return numpy.copy(self._observation)

    def get_loc_map(self, map_range):
        #Add the ground first
        #Find Relative Cells
        x_s = self._agent_grid[0] - map_range
        x_e = self._agent_grid[0] + map_range + 1
        y_s = self._agent_grid[1] - map_range
        y_e = self._agent_grid[1] + map_range + 1
        size = 2 * map_range + 1
        i_s = 0
        i_e = size
        j_s = 0
        j_e = size
        if(x_s < 0):
            i_s = -x_s
            x_s = 0
        if(x_e > self._n):
            i_e -= x_e - self._n
            x_e = self._n
        if(y_s < 0):
            j_s = -y_s
            y_s = 0
        if(y_e > self._n):
            j_e -= y_e - self._n
            y_e = self._n

        # local map: 
        #    ## =-1 for walls
        #    ## >0 for landmarks
        #    ## =0 for empty grounds
        loc_map = - numpy.ones(shape=(2 * map_range + 1, 2 * map_range + 1), dtype="float32")
        loc_map[i_s:i_e, j_s:j_e] = -self._cell_walls[x_s:x_e, y_s:y_e]
        if(self.task_type == "SURVIVAL"):
            loc_map[i_s:i_e, j_s:j_e] += self._cell_active_landmarks[x_s:x_e, y_s:y_e] + 1 # +1 for cell_active_landmarks in [-1, 0~n]
        else:
            loc_map[i_s:i_e, j_s:j_e] += self._cell_landmarks[x_s:x_e, y_s:y_e] + 1 # +1 for cell_active_landmarks in [-1, 0~n]

        loc_map = numpy.expand_dims(loc_map, axis=-1)
        wall_rgb = numpy.array([0, 0, 0], dtype="int32")
        empty_rgb = numpy.array([255, 255, 255], dtype="int32")

        color_map = ((loc_map == -1).astype("int32") * wall_rgb + 
                (loc_map == 0).astype("int32") * empty_rgb)
        for i in landmarks_rgb:
            color_map += (loc_map == (i + 1)).astype("int32") * landmarks_rgb[i].astype("int32")

        if("_agent_ori" in self.__dict__):
            ori = int(2.0 * self._agent_ori / PI + 0.5) % 4 # 0, 1, 2, 3
        else:
            ori = 0

        return numpy.rot90(color_map, k=ori, axes=(1,0)) # Need to rotate color map according to the orientation

    def get_global_map(self, resolution=(128, 128)):
        surf_map = pygame.Surface(resolution)
        surf_map.fill(pygame.Color("white"))
        render_x = resolution[0] / self._n
        render_y = resolution[1] / self._n
        render_avg = 0.5 * (render_x + render_y)
        it = numpy.nditer(self._cell_walls, flags=["multi_index"])
        landmark_size = 0.60
        landmark_emp = 0.5 * (1.0 - landmark_size)

        for _ in it:
            x,y = it.multi_index
            landmarks_id = self._cell_landmarks[x,y]
            if(self._cell_walls[x,y] > 0):
                text_buffer = pygame.transform.scale(self._text_surf[self._cell_texts[x,y]], (render_x, render_y))
                surf_map.blit(text_buffer,(x * render_x, y * render_y))
            if(landmarks_id > -1):
                pygame.draw.rect(surf_map, landmarks_color(landmarks_id, opacity=0.0), 
                                 ((x + landmark_emp) * render_x, (y + landmark_emp) * render_y, landmark_size * render_x, landmark_size * render_y), 
                                 width=0)
        pos_conversion = numpy.array([render_x, render_y]) / self._cell_size
        agent_pos = numpy.array(self._agent_loc) * pos_conversion
        paint_agent_arrow(surf_map, pygame.Color("gray"), (0, 0), (agent_pos[0], agent_pos[1]), self._agent_ori, 
                0.4 * render_avg, 0.5 * render_avg)
        npy_map = pygame.surfarray.array3d(surf_map)
        return surf_map, npy_map
