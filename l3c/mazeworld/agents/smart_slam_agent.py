import numpy
import math
from .agent_base import AgentBase
from queue import Queue
from l3c.mazeworld.envs.dynamics import PI

class SmartSLAMAgent(AgentBase):
    def render_init(self):
        pass

    def render_update(self):
        pass

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
                        if(self._landmarks_rewards[c_type - 1] < 0.0 and self._landmarks_cd[c_type - 1] < 1):
                            cost = 1 - self._landmarks_rewards[c_type - 1] / self._step_reward
                        else:
                            cost = 1
                else:
                    cost = 1
                if(self._cost_map[n_x, n_y] > self._cost_map[o_x, o_y] + cost):
                    self._cost_map[n_x, n_y] = self._cost_map[o_x, o_y] + cost
                    refresh_list.put((n_x, n_y))

    def policy(self, observation, r, env):
        self.update_cost_map()
        if(self.task_type=="SURVIVAL"):
            return self.policy_survival(observation, r, env)
        elif(self.task_type=="NAVIGATION"):
            return self.policy_navigation(observation, r, env)

    def policy_survival(self, observation, r, env):
        path_greedy, cost = self.navigate_landmarks_survival(0.50)
        path = path_greedy
        if(path is None or cost > 0):
            path_exp = self.exploration()
            if(path_exp is not None):
                path = path_exp
            elif(path is None):
                path = self._cur_grid

        return self.path_to_action(path)

    def policy_navigation(self, observation, r, env):
        path_greedy = self.navigate_landmarks_navigate(self._command)
        path = path_greedy
        if(path_greedy is None):
            path_exp = self.exploration()
            if(path_exp is not None):
                path = path_exp
            else:
                path = self._cur_grid
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
        req_ori = 2.0 * math.atan2(d_x, d_y) / PI
        deta_ori = req_ori - self._agent_ori
        if(numpy.abs(deta_ori) < 0.1):
            return 4 #(0, 1)
        elif(numpy.abs(deta_ori - 2) < 0.1 or numpy.abs(deta_ori + 2) < 0.1):
            return 3 #(0, -1)
        elif((deta_ori > 0  and deta_ori < 2) or deta_ori < -2):
            return 1 #(-1, 0)
        else:
            return 0 #(1, 0)

    def path_to_action_cont3d(self, path):
        if(len(path) < 2):
            return (0, 0)
        d_x = self._cur_grid_float[0] - path[0][0]
        d_y = self._cur_grid_float[1] - path[0][1]
        req_ori = 2.0 * math.atan2(d_x, d_y) / PI
        deta_ori = req_ori - self._agent_ori
        deta_s = numpy.sqrt(d_x ** 2 + d_y ** 2)
        if(numpy.abs(deta_ori) < 0.50):
            spd = min(deta_s, 1.0)
        else:
            spd = - min(deta_s, 1.0)
        if(deta_ori < 0):
            deta_ori += 4
        if(deta_ori > 0  and deta_ori < 2):
            turn = min(deta_ori, 1.0)
        else:
            turn = min(4 - deta_ori, 1.0)
        return (turn, spd)

    def retrieve_path(self, cost_map, goal_idx):
        path = [goal_idx]
        cost = cost_map[goal_idx]
        sel_x, sel_y = goal_idx
        while cost > 0:
            min_cost = cost
            for d_x, d_y in self.neighbors:
                n_x = sel_x + d_x
                n_y = sel_y + d_y
                if(n_x < 0 or n_x > self._nx - 1 and n_y < 0 or n_y > self._ny - 1):
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
        for i,idx in enumerate(self._landmarks_pos):
            if(i not in self._landmarks_visit):
                cost_map[idx] += r_exp / self._step_reward
            elif(self._landmarks_rewards[i] > 0.0):
                cost_map[idx] += 1 + self._landmarks_cd[i] - self._landmarks_rewards[i] / self._step_reward
        target_idx = numpy.unravel_index(numpy.argmin(cost_map), cost_map.shape)
        return self.retrieve_path(self._cost_map, target_idx), cost_map[target_idx]

