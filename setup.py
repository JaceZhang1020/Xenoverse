#   Copyright (c) 2022 DeepEvolution Authors. All Rights Reserved.
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

import io
from setuptools import setup, find_packages

__version__ = '0.1.1.4'

with io.open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='l3c',
    version=__version__,
    author='WorldEditors',
    author_email='',
    description=('L3C provide abundant environments for Lifelong Learning (L3) in Context.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/FutureAGI/L3C',
    license="Apache",
    packages=[package for package in find_packages()
              if package.startswith('l3c')],
    package_data={'l3c': [
        './mazeworld/envs/img/*',
        './lococrowds/envs/assets/ants/*.xml',
        './lococrowds/envs/assets/humanoids/*.xml',
        './lococrowds/envs/assets/scenes/*',
        './lococrowds/envs/assets/*.png',
        ]
    },
    python_requires='>=3.7',
    tests_require=['pytest', 'mock'],
    include_package_data=True,
    install_requires=[
        'gym>=0.18.0',
        'numpy>=1.16.4',
        'Pillow>=6.2.2',
        'six>=1.12.0',
        'pygame>=2.0.2dev2',
        'numba>=0.55.0'
    ],
    extras_require={},
    zip_safe=False,
)
