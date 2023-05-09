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
from l3c.metalm.metalmv1 import MetaLMv1
from l3c.metalm.metalmv2 import MetaLMv2

register(
    id='meta-lm-v1',
    entry_point='l3c.metalm:MetaLMv1',
    kwargs={"V": 64,
        "n": 10,
        "l": 64,
        "e": 0.10,
        "L": 2048
    }
)

register(
    id='meta-lm-v2',
    entry_point='l3c.metalm:MetaLMv2',
    kwargs={"V": 64,
        "n": 16,
        "N": 16,
        "e": 0.10,
        "L": 2048
    }
)
