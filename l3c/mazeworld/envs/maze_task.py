"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
from collections import namedtuple
from numpy import random as npyrnd
from numpy.linalg import norm

class MazeTaskManager(object):

    # Configurations that decides a specific task
    TaskConfig = namedtuple("TaskConfig", ["start", "cell_landmarks", "cell_walls", "cell_texts", 
        "cell_size", "wall_height", "agent_height", "initial_life", "max_life",
        "step_reward", "goal_reward", "landmarks_rewards", "landmarks_coordinates", "landmarks_refresh_interval", "commands_sequence"])

    def __init__(self, texture_dir, verbose=False):
        pathes = os.path.split(os.path.abspath(__file__))
        texture_dir = os.sep.join([pathes[0], texture_dir])
        texture_files = os.listdir(texture_dir)
        texture_files.sort()
        grounds = [None]
        for file_name in texture_files:
            if(file_name.find("wall") >= 0):
                grounds.append(pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name]))))
            if(file_name.find("ground") >= 0):
                grounds[0] = pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name])))
            if(file_name.find("ceil") >= 0):
                self.ceil = pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name])))
            if(file_name.find("arrow") >= 0):
                self.arrow = pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name])))
        self.grounds = numpy.asarray(grounds, dtype="float32")
        self.verbose = verbose

    @property
    def n_texts(self):
        return self.grounds.shape[0]

    def sample_task(self,
            n=15, 
            allow_loops=True, 
            cell_size=2.0, 
            wall_height=3.2, 
            agent_height=1.6,
            step_reward=-0.01,
            goal_reward=None,
            initial_life=1.0,
            max_life=3.0,
            landmarks_refresh_interval=200,
            landmarks_avg_reward=0.60,
            landmarks_number=5,
            commands_sequence=200,
            wall_density=0.40,
            verbose=False):
        # Initialize the maze ...
        assert n > 6, "Minimum required cells are 7"
        assert n % 2 != 0, "Cell Numbers can only be odd"
        assert landmarks_number > 1, "There must be at least 1 goal, thus landmarks_number must > 1"
        if(landmarks_number > 15):
            landmarks_number = 15
            print("landmarks number too much, set to 15")
        if(self.verbose):
            print("Generating an random maze of size %dx%d, with allow loops=%s, crowd ratio=%f"%(n, n, allow_loops, wall_density))
        cell_walls = numpy.ones(shape=(n, n), dtype="int32")
        cell_texts = numpy.random.randint(1, self.n_texts, size=(n, n))

        # Dig the initial holes
        for i in range(1, n, 2):
            for j in range(1, n, 2):
                cell_walls[i,j] = 0

        #Initialize the logics for prim based maze generation
        wall_dict = dict()
        path_dict = dict()
        rev_path_dict = dict()
        path_idx = 0
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if(cell_walls[i,j] > 0): # we will keep the axial point
                    wall_dict[i, j] = 0
                elif(cell_walls[i,j] == 0):
                    path_dict[i, j] = path_idx
                    rev_path_dict[path_idx] = [(i,j)]
                    path_idx += 1

        #Prim the wall until all points are connected
        max_cell_walls = numpy.product(cell_walls[1:-1, 1:-1].shape)
        while len(rev_path_dict) > 1 or (allow_loops and numpy.sum(cell_walls[1:-1, 1:-1]) > max_cell_walls * wall_density):
            wall_list = list(wall_dict.keys())
            random.shuffle(wall_list)
            for i, j in wall_list:
                new_path_id = -1
                connected_path_id = dict()
                abandon_path_id = dict()
                max_duplicate = 1

                for d_i, d_j in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                    if((d_i > 0  and d_i < n and d_j > 0 and d_j < n)
                            and cell_walls[d_i, d_j] < 1):
                        # calculate duplicate path id that might creat a loop
                        if path_dict[d_i, d_j] not in connected_path_id:
                            connected_path_id[path_dict[d_i, d_j]] = 1
                        else:
                            connected_path_id[path_dict[d_i, d_j]] += 1
                        if(connected_path_id[path_dict[d_i, d_j]] > max_duplicate):
                            max_duplicate = connected_path_id[path_dict[d_i, d_j]]

                        # decide the new path_id and find those to be deleted
                        if(path_dict[d_i, d_j] < new_path_id or new_path_id < 0):
                            if(new_path_id >= 0):
                                abandon_path_id[new_path_id] = (new_i, new_j)
                            new_path_id = path_dict[d_i, d_j]
                            new_i = d_i
                            new_j = d_j
                        elif(path_dict[d_i, d_j] != new_path_id): # need to be abandoned
                            abandon_path_id[path_dict[d_i, d_j]] = (d_i, d_j)
                if(len(abandon_path_id) >= 1 and max_duplicate < 2):
                    break
                if(len(abandon_path_id) >= 1 and max_duplicate > 1 and allow_loops):
                    break
                if(allow_loops and len(rev_path_dict) < 2 and random.random() < 0.2):
                    break

            if(new_path_id < 0):
                continue
                        
            # add the released wall
            rev_path_dict[new_path_id].append((i,j))
            path_dict[i,j] = new_path_id
            cell_walls[i,j] = 0
            del wall_dict[i,j]

            # merge the path
            for path_id in abandon_path_id:
                rev_path_dict[new_path_id].extend(rev_path_dict[path_id])
                for t_i_o, t_j_o in rev_path_dict[path_id]:
                    path_dict[t_i_o,t_j_o] = new_path_id
                del rev_path_dict[path_id]

        #Paint the texture of passways to ground textures 
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                if(cell_walls[i,j] < 1):
                    cell_texts[i,j] = 0

        #Randomize a start point and n landmarks
        landmarks_likelihood = numpy.random.rand(n, n) - cell_walls
        idxes = numpy.argsort(landmarks_likelihood, axis=None)
        topk_idxes = idxes[-landmarks_number-1:]

        def idx_trans(idx):
            return (idx // n, idx % n)

        start = idx_trans(topk_idxes[-1])
        landmarks = [idx_trans(i) for i in topk_idxes[:-1]]

        #Calculate goal reward, default is - n sqrt(n) * step_reward
        assert step_reward < 0, "step_reward must be < 0"
        if(goal_reward is None):
            def_goal_reward = - numpy.sqrt(n) * n * step_reward
        else:
            def_goal_reward = goal_reward
        assert def_goal_reward > 0, "goal reward must be > 0"

        commands_sequence = numpy.random.randint(0, len(landmarks), commands_sequence)

        # Generate landmark rewards
        landmarks_rewards = numpy.random.rand(landmarks_number) - 0.50 + landmarks_avg_reward
        cell_landmarks = numpy.zeros_like(cell_walls) - 1
        for i,idx in enumerate(landmarks):
            cell_landmarks[tuple(idx)] = int(i)
        cell_landmarks = cell_landmarks.astype(cell_walls.dtype)
        if(verbose):
            print("\n\n---------Successfully generate maze task with the following attributes-----------\n")
            print("Maze size %s x %s" %(n, n)) 
            print("Initialze born location: %s,%s" % start)
            integrate_maze = cell_landmarks - cell_walls
            print("Maze configuration (-2: walls, -1 empty, >0 landmarks and ID): \n%s" % integrate_maze)
            print("Commands sequence: \n%s" % commands_sequence)
            print("Landmarks Rewards: \n%s" % landmarks_rewards)
            print("Landmarks Refresh Interval: \n%s" % landmarks_refresh_interval)
            print("\n----------------------\n\n")

        return MazeTaskManager.TaskConfig(
                start=start,
                cell_landmarks=cell_landmarks,
                cell_walls=cell_walls,
                cell_texts=cell_texts,
                cell_size=cell_size,
                step_reward=step_reward,
                goal_reward=def_goal_reward,
                wall_height=wall_height,
                agent_height=agent_height,
                initial_life=initial_life,
                max_life=max_life,
                commands_sequence=commands_sequence,
                landmarks_coordinates=landmarks,
                landmarks_rewards=landmarks_rewards,
                landmarks_refresh_interval=landmarks_refresh_interval
                )

MAZE_TASK_MANAGER=MazeTaskManager("img")
MazeTaskSampler = MAZE_TASK_MANAGER.sample_task

if __name__=="__main__":
    task = MazeTaskSampler(verbose=True)
