"""
Core File of Maze Env
"""
import os
import numpy
import pygame
from numpy import random
from collections import namedtuple
from numpy import random as npyrnd
from numpy.linalg import norm
from copy import deepcopy
from l3c.mazeworld.envs.grid_ops import genmaze_by_primwall


def gentext(cell_walls, textlib_walls, textlib_grounds, textlib_ceilings):
    n = cell_walls.shape[0]
    cell_texts = numpy.random.randint(0, len(textlib_walls), size=cell_walls.shape)

    #Paint the texture of passways to ground textures 
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if(cell_walls[i,j] < 1):
                cell_texts[i,j] = 0

    text_ground = random.randint(0, len(textlib_grounds))
    text_ceiling = random.randint(0, len(textlib_ceilings))

    return cell_texts, text_ground, text_ceiling

def idx_trans(idx, n):
    return (idx // n, idx % n)

def gentargets(cell_walls, landmarks_number):
    #Randomize a start point and n landmarks
    n = cell_walls.shape[0]
    landmarks_likelihood = numpy.random.rand(*cell_walls.shape) - cell_walls
    idxes = numpy.argsort(landmarks_likelihood, axis=None)
    topk_idxes = idxes[-landmarks_number:]
    landmarks = [idx_trans(i, n) for i in idxes[-landmarks_number:]]

    cell_landmarks = numpy.zeros_like(cell_walls, dtype=numpy.int8) - 1
    for i,idx in enumerate(landmarks):
        cell_landmarks[tuple(idx)] = int(i)
    cell_landmarks = cell_landmarks.astype(cell_walls.dtype)
    return landmarks, cell_landmarks

def genstart(cell_walls, cell_landmarks):
    n = cell_walls.shape[0]
    landmarks_likelihood = numpy.random.rand(n, n) - cell_walls - cell_landmarks
    idxes = numpy.argsort(landmarks_likelihood, axis=None)
    return idx_trans(idxes[-1], n)

class MazeTaskManager(object):
    def __init__(self, texture_dir, verbose=False):
        pathes = os.path.split(os.path.abspath(__file__))
        texture_dir = os.sep.join([pathes[0], texture_dir])
        texture_files = os.listdir(texture_dir)
        texture_files.sort()
        textlib_grounds = []
        textlib_ceilings = []
        textlib_walls = []
        for file_name in texture_files:
            if(file_name.find("wall") >= 0):
                textlib_walls.append(pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name]))))
            if(file_name.find("ground") >= 0):
                textlib_grounds.append(pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name]))))
            if(file_name.find("ceil") >= 0):
                textlib_ceilings.append(pygame.surfarray.array3d(pygame.image.load(os.sep.join([texture_dir, file_name]))))
        self.textlib_walls = numpy.asarray(textlib_walls, dtype="float32")
        self.textlib_grounds = numpy.asarray(textlib_grounds, dtype="float32")
        self.textlib_ceilings = numpy.asarray(textlib_ceilings, dtype="float32")
        self.verbose = verbose

    @property
    def n_texts(self):
        return self.grounds.shape[0]

    def sample_cmds(self, n, commands_sequence):
        xs = numpy.random.randint(0, n, commands_sequence)
        for i in range(xs.shape[0]):
            if(i > 0):
                if(xs[i] == xs[i-1]):
                    xs[i] = (xs[i] + random.randint(1, n)) % n
        return xs

    def sample_task(self,
            n=15, 
            allow_loops=True, 
            cell_size=2.0, 
            wall_height=3.2, 
            agent_height=1.6,
            step_reward=-0.01,
            collision_reward=-1.0,
            goal_reward=None,
            initial_life=1.0,
            max_life=3.0,
            landmarks_refresh_interval=200,
            landmarks_avg_reward=0.60,
            landmarks_number=5,
            commands_sequence=200,
            wall_density=0.40,
            seed=None,
            verbose=False):
        # Initialize the maze ...
        if(seed is not None):
            seed = time.time() * 1000 % 65536
        numpy.random.seed(seed)
        assert n > 6, "Minimum required cells are 7"
        assert n % 2 != 0, "Cell Numbers can only be odd"
        assert landmarks_number > 1, "There must be at least 1 goal, thus landmarks_number must > 1"
        if(landmarks_number > 15):
            landmarks_number = 15
            print("landmarks number too much, set to 15")
        if(self.verbose):
            print("Generating an random maze of size %dx%d, with allow loops=%s, crowd ratio=%f"%(n, n, allow_loops, wall_density))

        # Generate the wall topology
        cell_walls = genmaze_by_primwall(n, allow_loops=allow_loops, wall_density=wall_density)

        # Selects the texture
        cell_texts, ground_text, ceiling_text = gentext(cell_walls, self.textlib_walls, self.textlib_grounds, self.textlib_ceilings)

        # Generate landmarks (Potential Navigation Targets)
        landmarks, cell_landmarks = gentargets(cell_walls, landmarks_number)

        # Generate start location
        start = genstart(cell_walls, cell_landmarks)

        #Calculate goal reward, default is - n sqrt(n) * step_reward
        assert step_reward < 0, "step_reward must be < 0"
        if(goal_reward is None):
            def_goal_reward = - numpy.sqrt(n) * n * step_reward
        else:
            def_goal_reward = goal_reward
        assert def_goal_reward > 0, "goal reward must be > 0"

        # Sample Commands Sequences
        commands_sequence = self.sample_cmds(len(landmarks), commands_sequence)

        if(verbose):
            print("\n\n---------Successfully generate maze task with the following attributes-----------\n")
            print("Maze size %s x %s" %(n, n)) 
            print("Initialze born location: %s,%s" % start)
            integrate_maze = cell_landmarks + 1 - cell_walls
            print("Maze configuration (-1: walls, 0 empty, >1 landmarks and ID): \n%s" % integrate_maze)
            print("Commands sequence: \n%s" % commands_sequence)
            print("\n----------------------\n\n")

        return {"start":start,
                "cell_walls":cell_walls,
                "cell_texts":cell_texts,
                "cell_size":cell_size,
                "ground_text":ground_text,
                "ceiling_text":ceiling_text,
                "step_reward":step_reward,
                "goal_reward":def_goal_reward,
                "collision_reward":collision_reward,
                "wall_height":wall_height,
                "agent_height":agent_height,
                "commands_sequence":commands_sequence,
                "landmarks_coordinates":landmarks,
                "cell_landmarks":cell_landmarks}

    def resample_task(self, task, 
            resample_cmd=True, 
            resample_start=True, 
            resample_landmarks_color=False, 
            resample_landmarks=False,
            seed=None):
        # Randomize a start point and n landmarks while keeping the scenario still
        if(seed is not None):
            seed = time.time() * 1000 % 65536
        numpy.random.seed(seed)
        n = task["cell_walls"].shape[0]
        def idx_trans(idx):
            return (idx // n, idx % n)

        landmarks_number = len(task["landmarks_coordinates"])
        if(resample_landmarks):
            landmarks, cell_landmarks = gentargets(task["cell_walls"], landmarks_number)
        elif(resample_landmarks_color):
            landmarks = deepcopy(task["landmarks_coordinates"])
            random.shuffle(landmarks)
            cell_landmarks = numpy.zeros_like(task["cell_walls"]) - 1
            for i,idx in enumerate(landmarks):
                cell_landmarks[tuple(idx)] = int(i)
            cell_landmarks = cell_landmarks.astype(task["cell_walls"].dtype)
        else:
            landmarks = deepcopy(task["landmarks_coordinates"])
            cell_landmarks = deepcopy(task["cell_landmarks"])

        # Generate start location
        start = genstart(task["cell_walls"], cell_landmarks)

        # Generate command sequences
        commands_sequence = self.sample_cmds(len(landmarks), len(task["commands_sequence"]))

        new_task = deepcopy(task)
        new_task["start"] = start
        new_task["landmarks_coordinates"] = landmarks
        new_task["cell_landmarks"] = cell_landmarks
        new_task["commands_sequence"] = commands_sequence

        return new_task

MAZE_TASK_MANAGER=MazeTaskManager("img")
MazeTaskSampler = MAZE_TASK_MANAGER.sample_task
Resampler = MAZE_TASK_MANAGER.resample_task


if __name__=="__main__":
    task = MazeTaskSampler(verbose=False)
    print(task)
    print(Resampler(task))
    print(Resampler(task, resample_landmarks_color=True))
    print(Resampler(task, resample_landmarks=True))
