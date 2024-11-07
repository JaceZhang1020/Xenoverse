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

from numpy import random
import time

def pseudo_random_seed(hyperseed=0):
    '''
    Generate a pseudo random seed based on current time and system random number
    '''
    timestamp = time.time_ns()
    system_random = int(random.random() * 100000000)
    pseudo_random = timestamp + system_random + hyperseed
    
    return pseudo_random % (4294967296)