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

import sys
import _io
import numpy
import gym
from numpy import random

class RandomRNN(object):
    def __init__(self, n_emb=16, n_hidden=64, eps=0.10, n_vocab=256):
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.n_vocab = n_vocab
        self.emb = numpy.random.normal(0, 1.0, size=(self.n_vocab, self.n_emb))
        self.W_i = numpy.random.normal(0, 1.0, size=(self.n_emb, self.n_hidden))
        self.W_h = numpy.random.normal(0, 1.0, size=(self.n_hidden, self.n_hidden))
        self.b_h = numpy.random.normal(0, 1.0, size=(self.n_hidden))
        self.W_o = numpy.random.normal(0, 1.0, size=(self.n_hidden, self.n_vocab))
        self.b_o = numpy.random.normal(0, 1.0, size=(self.n_vocab))
        self.eps = eps

    def softmax(self, x):
        e_x = numpy.exp(x - numpy.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, l, batch):
        ind = 0

        # Start token
        s_tok = numpy.random.randint(0, self.n_vocab, size=(batch,))

        cur_tok = numpy.copy(s_tok)

        h = numpy.zeros((batch, self.n_hidden))
        seqs = []
        while ind < l:
            ind += 1
            i = self.emb[cur_tok]
            h = numpy.tanh(numpy.matmul(i, self.W_i) + numpy.matmul(h, self.W_h) + self.b_h)
            o = numpy.matmul(h, self.W_o) + self.b_o

            tok_greedy = numpy.argmax(o, axis=-1)
            tok_random = numpy.random.randint(0, self.n_vocab, size=(batch,))
            tok_select = (numpy.random.rand(batch) < self.eps)
            cur_tok = numpy.where(numpy.random.rand(batch) < self.eps, tok_random, tok_greedy)
            seqs.append(cur_tok)

        return numpy.transpose(numpy.asarray(seqs, dtype="int32"))

class MetaLMv2(gym.Env):
    """
    Pseudo Langauge Generated from RNN models
    V: vocabulary size
    n: embedding size (input size)
    N: hidden size
    e: epsilon in epsilon greedy
    L: maximum length
    """
    def __init__(self, 
            V=64, 
            n=4,
            N=4,
            e=0.10,
            L=4096):
        self.L = int(L)
        self.V = int(V)
        self.n = n
        self.N = N
        self.eps = e
        assert n > 1 and V > 1 and N > 1 and L > 1 

    def data_generator(self):
        nn = RandomRNN(n_emb = self.n, n_hidden = self.N, n_vocab = self.V + 1, eps=self.eps)
        tokens = nn.forward(self.L + 1, 1)[0]
        feas = tokens[:-1]
        labs = tokens[1:]
        return feas, labs

    def batch_generator(self, batch_size):
        nn = RandomRNN(n_emb = self.n, n_hidden = self.N, n_vocab = self.V + 1, eps=self.eps)
        tokens = nn.forward(self.L + 1, batch_size)
        feas = tokens[:, :-1]
        labs = tokens[:, 1:]
        return feas, labs

    def generate_to_file(self, size, output_stream):
        feas,labs = self.batch_generator(size)
        if(isinstance(output_stream, _io.TextIOWrapper)):
            need_close = False
        elif(isinstance(output_stream, str)):
            output_stream = open(output_stream, "w")
            need_close = True
        for i in range(feas.shape[0]):
            output_stream.write("\t".join(map(lambda x:"%s,%s"%(x[0],x[1]), zip(feas[i].tolist(), labs[i].tolist()))))
            output_stream.write("\n")
        if(need_close):
            output_stream.close()

    @property
    def VocabSize(self):
        return self.V + 1

    @property
    def SepID(self):
        raise Exception("Not Defined")
        

    @property
    def MaskID(self):
        return 0

    @property
    def PaddingID(self):
        return 0
