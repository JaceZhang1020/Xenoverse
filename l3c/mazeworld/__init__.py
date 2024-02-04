#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.envs.registration import register
from l3c.mazeworld.envs import MazeWorldContinuous3D
from l3c.mazeworld.envs import MazeWorldDiscrete3D
from l3c.mazeworld.envs import MazeWorldDiscrete2D
from l3c.mazeworld.envs import MazeTaskSampler

register(
    id='mazeworld-continuous-3D-v1',
    entry_point='l3c.mazeworld:MazeWorldContinuous3D',
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "resolution": (256, 256),
        "max_steps": 5000,
        "visibility_3D": 12.0,
        "task_type": "NAVIGATION"
    }
)

register(
    id='mazeworld-discrete-3D-v1',
    entry_point='l3c.mazeworld:MazeWorldDiscrete3D',
    kwargs={
        "enable_render": True,
        "render_scale": 480,
        "resolution": (256, 256),
        "max_steps": 2000,
        "visibility_3D": 12.0,
        "task_type": "NAVIGATION"
    }
)

register(
    id='mazeworld-discrete-2D-v1',
    entry_point='l3c.mazeworld:MazeWorldDiscrete2D',
    kwargs={
        "enable_render": True,
        "max_steps": 2000,
        "visibility_2D": 1,
        "task_type": "NAVIGATION"
    }
)
