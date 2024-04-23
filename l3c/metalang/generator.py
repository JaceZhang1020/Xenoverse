#   Copyright (c) 2021 DeepEvolution Authors. All Rights Reserved.
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

# This file is used to generate data for meta language models

import sys
import argparse
from l3c.metalang import MetaLangV1
from l3c.metalang import MetaLangV2

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generating Pseudo-Training Data')
    parser.add_argument('--version', type=str, choices=['v1', 'v2'], default='v2')
    parser.add_argument('--vocab_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--elements_length', type=int, default=64)
    parser.add_argument('--elements_number', type=int, default=10)
    parser.add_argument('--error_rate', type=float, default=0.20)
    parser.add_argument('--n_gram', type=float, default=3)
    parser.add_argument('--sequence_length', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output_type', type=str, choices=['txt', 'npy'], default='txt')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if(args.version == 'v1'):
        dataset = MetaLangV1(
                V=args.vocab_size,
                n=args.elements_number,
                l=args.elements_length,
                e=args.error_rate,
                L=args.sequence_length)
    elif(args.version == 'v2'):
        dataset = MetaLangV2(
                V=args.vocab_size,
                n=args.n_gram,
                d=args.embedding_size,
                N=args.hidden_size,
                e=args.error_rate,
                L=args.sequence_length)

    if(args.output_type == 'npy'):
        if(args.output is None):
            raise Exception("Must specify --output when output_type is npy")
        dataset.generate_npy(args.samples, args.output)
    elif(args.output_type == 'txt'):
        if(args.output is None):
            dataset.generate_text(args.samples, sys.stdout)
        else:
            dataset.generate_text(args.samples, args.output)
