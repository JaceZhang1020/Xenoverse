import numpy
import math
import pygame
from .agent_base import AgentBase
from queue import Queue
from pygame import font
from l3c.mazeworld.envs.dynamics import PI
from l3c.mazeworld.envs.ray_caster_utils import landmarks_color, landmarks_rgb, landmarks_rgb_arr, paint_agent_arrow

class SmartSLAMAgent(AgentBase):
    def render_init(self, view_size=(640, 640)):
        """
        Initialize a God View With Landmarks
        """
        font.init()
        self._font = font.SysFont("Arial", 18)

        #Initialize the agent drawing
        self._render_cell_size_x = view_size[0] / self._nx
        self._render_cell_size_y = view_size[1] / self._ny
        self._view_size = view_size

        self._pos_conversion_x = self._render_cell_size_x / self.maze_env.maze_core._cell_size
        self._pos_conversion_y = self._render_cell_size_y / self.maze_env.maze_core._cell_size

        self._screen = pygame.Surface(view_size)
        self._screen = pygame.display.set_mode(view_size)
        pygame.display.set_caption("AgentRender")
        self._surf_god = pygame.Surface(view_size)
        self._surf_god.fill(pygame.Color("white"))

        for x in range(self._nx):
            for y in range(self._ny):
                if(self._god_info[x,y] < 0):
                    pygame.draw.rect(self._surf_god, pygame.Color("black"), (x * self._render_cell_size_x, y * self._render_cell_size_y,
                            self._render_cell_size_x, self._render_cell_size_y), width=0)

    def render_update(self):
        # paint landmarks
        self._screen.blit(self._surf_god, (0, 0))
        for landmarks_id, (x,y) in enumerate(self._landmarks_coordinates):
            if(self._landmarks_cd[landmarks_id] < 1):
                pygame.draw.rect(self._screen, landmarks_color(landmarks_id), 
                        (x * self._render_cell_size_x, y * self._render_cell_size_y,
                        self._render_cell_size_x, self._render_cell_size_y), width=0)
        # paint masks (mists)
        for x in range(self._nx):
            for y in range(self._ny):
                if(self._mask_info[x,y] < 1):
                    pygame.draw.rect(self._screen, pygame.Color("grey"), (x * self._render_cell_size_x, y * self._render_cell_size_y,
                            self._render_cell_size_x, self._render_cell_size_y), width=0)

        # add texts
        if(self.task_type is "SURVIVAL"):
            txt_life = self._font.render("Life: %f"%self._life, 0, pygame.Color("green"))
            self._screen.blit(txt_life,(90, 10))

        # paint agents
        agent_pos = [self._agent_loc[0] * self._pos_conversion_x, self._agent_loc[1] * self._pos_conversion_y]
        paint_agent_arrow(self._screen, pygame.Color("gray"), (0, 0), (agent_pos[0], agent_pos[1]), 0.5 * PI * self._agent_ori, 
                0.4 * self._pos_conversion_x, 0.5 * self._pos_conversion_x)

        # paint target trajectory
        for i in range(len(self._path)-1):
            factor = i / len(self._path)
            p = self._path[i]
            n = self._path[i+1]
            p = [(p[0] + 0.5) * self._render_cell_size_x, (p[1] + 0.5) *  self._render_cell_size_y]
            n = [(n[0] + 0.5) * self._render_cell_size_x, (n[1] + 0.5) *  self._render_cell_size_y]
            pygame.draw.line(self._screen, pygame.Color(int(255 * factor), int(255 * (1 - factor)), 128, 255), p, n, width=1)

        # display
        pygame.display.update()
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        keys = pygame.key.get_pressed()

    def update_cost_map(self, r_exp=0.25):
        # Calculate Shortest Distance using A*
        # In survival mode, consider the loss brought by rewards
        self._cost_map = 10000 * numpy.ones_like(self._god_info)
        refresh_list = Queue()
        refresh_list.put((self._cur_grid[0], self._cur_grid[1]))
        self._cost_map[self._cur_grid[0], self._cur_grid[1]] = 0
        while not refresh_list.empty():
            o_x, o_y = refresh_list.get()
            for d_x, d_y in self.neighbors:
                n_x = o_x + d_x
                n_y = o_y + d_y
                if(n_x >= self._nx or n_x < 0 or n_y >= self._ny or n_y < 0):
                    continue
                c_type = self._god_info[n_x, n_y]
                m_type = self._mask_info[n_x, n_y]
                if(c_type < 0 and m_type > 0):
                    continue
                elif(m_type < 1):
                    cost = 10
                elif(c_type > 0 and self._mask_info[n_x, n_y] > 0): # There is landmarks
                    if(self.task_type == "NAVIGATION"):
                        cost = 1
                    elif(self.task_type == "SURVIVAL"):
                        # Consider the extra costs of known traps
                        cost = 1
                        if(self._landmarks_rewards[c_type - 1] < 0.0 and self._landmarks_cd[c_type - 1] < 1):
                            cost += self._landmarks_rewards[c_type - 1] / self._step_reward
                else:
                    cost = 1
                if(self._cost_map[n_x, n_y] > self._cost_map[o_x, o_y] + cost):
                    self._cost_map[n_x, n_y] = self._cost_map[o_x, o_y] + cost
                    refresh_list.put((n_x, n_y))

    def policy(self, observation, r):
        self.update_cost_map()
        if(self.task_type=="SURVIVAL"):
            return self.policy_survival(observation, r)
        elif(self.task_type=="NAVIGATION"):
            return self.policy_navigation(observation, r)

    def policy_survival(self, observation, r):
        path_greedy, cost = self.navigate_landmarks_survival(0.50)
        path = path_greedy
        if(path is None or cost > 0):
            path_exp = self.exploration()
            if(path_exp is not None):
                path = path_exp
            elif(path is None):
                path = self._cur_grid
        self._path = path
        return self.path_to_action(path)

    def policy_navigation(self, observation, r):
        path_greedy = self.navigate_landmarks_navigate(self._command)
        path = path_greedy
        if(path_greedy is None):
            path_exp = self.exploration()
            if(path_exp is not None):
                path = path_exp
            else:
                path = self._cur_grid
        self._path = path
        return self.path_to_action(path)

    def path_to_action(self, path):
        if(self.maze_type=="Continuous3D"):
            return self.path_to_action_cont3d(path)
        elif(self.maze_type=="Discrete3D"):
            return self.path_to_action_disc3d(path)
        elif(self.maze_type=="Discrete2D"):
            return self.path_to_action_disc2d(path)

    def path_to_action_disc2d(self, path):
        if(len(path) < 2):
            return 0#(0, 0)
        d_x = path[1][0] - path[0][0]
        d_y = path[1][1] - path[0][1]
        if(d_x == -1 and d_y == 0):
            return 1#(-1, 0)
        if(d_x == 1 and d_y == 0):
            return 2#(1, 0)
        if(d_x == 0 and d_y == -1):
            return 3#(-1, 0)
        if(d_x == 0 and d_y == 1):
            return 4#(-1, 0)

    def path_to_action_disc3d(self, path):
        if(len(path) < 2):
            return 0 #(0, 0)
        d_x = path[1][0] - path[0][0]
        d_y = path[1][1] - path[0][1]
        req_ori = 2.0 * math.atan2(d_y, d_x) / PI
        deta_ori = req_ori - self._agent_ori
        deta_ori = int(deta_ori) % 4 + deta_ori - int(deta_ori)
        if(deta_ori > 2):
            deta_ori -= 4
        if(numpy.abs(deta_ori) < 0.2):
            return 4 #(0, 1)
        elif(numpy.abs(deta_ori) > 1.8 and self._god_info[path[1]] >=0 and self._mask_info[path[1]]>0):
            return 3 #(0, -1)
        elif(deta_ori < 0):
            return 2 #(-1, 0)
        else:
            return 1 #(1, 0)

    def path_to_action_cont3d(self, path):
        if(len(path) < 2):
            d_x = path[0][0] + 0.5 - self._cur_grid_float[0]
            d_y = path[0][1] + 0.5 - self._cur_grid_float[1]
            deta_s = numpy.sqrt(d_x ** 2 + d_y ** 2)
            if(deta_s < 0.20):
                return (0, 0)
        else:
            d_x = path[1][0] + 0.5 - self._cur_grid_float[0]
            d_y = path[1][1] + 0.5 - self._cur_grid_float[1]
            deta_s = numpy.sqrt(d_x ** 2 + d_y ** 2)
        req_ori = 2.0 * math.atan2(d_y, d_x) / PI
        deta_ori = req_ori - self._agent_ori
        deta_ori = int(deta_ori) % 4 + deta_ori - int(deta_ori)
        if(deta_ori > 2):
            deta_ori -= 4
        if(numpy.abs(deta_ori) < 0.50):
            spd = min(deta_s, 1.0)
        elif(numpy.abs(deta_ori) > 1.50):
            spd = - min(deta_s, 1.0)
        else:
            spd = 0.0
        if(deta_ori < 0):
            turn = - min(numpy.abs(deta_ori), 1.0)
        else:
            turn = min(numpy.abs(deta_ori), 1.0)
        return (turn, spd)

    def retrieve_path(self, cost_map, goal_idx):
        path = [goal_idx]
        cost = cost_map[goal_idx]
        sel_x, sel_y = goal_idx
        iteration = 0
        while sel_x != self._cur_grid[0] or sel_y != self._cur_grid[1] :
            iteration += 1
            min_cost = cost
            for d_x, d_y in self.neighbors:
                n_x = sel_x + d_x
                n_y = sel_y + d_y
                if(n_x < 0 or n_x > self._nx - 1 or n_y < 0 or n_y > self._ny - 1):
                    continue
                if(cost_map[n_x, n_y] < min_cost):
                    min_cost = cost_map[n_x, n_y]
                    sel_x = n_x
                    sel_y = n_y
                    path.insert(0, (n_x, n_y))
            cost=cost_map[sel_x, sel_y]
        return path

    def exploration(self):
        utility = self._cost_map + 10000 * self._mask_info
        if(numpy.argmax(utility) >= 10000):
            return None 
        target_idx = numpy.unravel_index(numpy.argmin(utility), utility.shape)
        return self.retrieve_path(self._cost_map, target_idx)

    def navigate_landmarks_navigate(self, landmarks_id):
        idxes = numpy.argwhere(self._god_info == landmarks_id + 1)
        for idx in idxes:
            if(self._mask_info[idx[0], idx[1]] < 1):
                continue
            else:
                return self.retrieve_path(self._cost_map, tuple(idx))
        return None

    def navigate_landmarks_survival(self, r_exp):
        cost_map = numpy.copy(self._cost_map)
        for i,idx in enumerate(self._landmarks_coordinates):
            if(i not in self._landmarks_visit):
                cost_map[idx] += r_exp / self._step_reward
            elif(self._landmarks_rewards[i] > 0.0):
                cost_map[idx] += self._landmarks_cd[i] + self._landmarks_rewards[i] / self._step_reward
        target_idx = numpy.unravel_index(numpy.argmin(cost_map), cost_map.shape)
        return self.retrieve_path(self._cost_map, target_idx), cost_map[target_idx]

