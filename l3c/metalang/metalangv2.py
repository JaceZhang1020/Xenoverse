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

class RandomNGram(object):
    def __init__(self, batch=1, n_emb=16, n_hidden=64, n_vocab=256, n_gram=3, hardness=5, seed=None):
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.n_vocab = n_vocab
        self.batch = batch
        self.emb = numpy.random.normal(0, 1.0, size=(self.batch, self.n_vocab, self.n_emb))
        self.n_gram = n_gram
        weights = []
        if(seed is not None):
            numpy.random.seed(seed)
        self.weight_i = numpy.random.normal(0, 1.0, size=(self.batch, self.n_gram, self.n_emb, self.n_hidden))
        self.bias_i = numpy.random.normal(0, 1.0, size=(self.batch, self.n_gram, 1, self.n_hidden))
        self.weight_o = numpy.random.normal(0, 1.0, size=(self.batch, self.n_hidden, self.n_vocab))
        self.bias_o = numpy.random.normal(0, 1.0, size=(self.batch, 1, self.n_vocab))
        self.w_arr = numpy.expand_dims(numpy.arange(self.n_gram), axis=[0, 2, 3])
        self.hardness = hardness

        self.s_tok = 0

    def softmax(self, x):
        e_x = numpy.exp(x - numpy.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def forward(self, l):
        ind = 0

        def mean_var_norm(i):
            m_i = numpy.mean(i)
            m_ii = numpy.mean(i * i)
            std = numpy.sqrt(m_ii - m_i * m_i)
            return (1.0 / std) * (i - m_i)

        cur_tok = numpy.full((self.batch,), self.s_tok)
        idxes = numpy.arange(self.batch)
        pad_emb = numpy.expand_dims(numpy.array(self.emb[idxes, cur_tok]), axis=1)
 
        h = numpy.zeros((self.batch, self.n_hidden))

        # mark whether there is end token
        seqs = []
        seqs.append(cur_tok)
        ppl = 0
        tok_cnt = 0
        tok_embs = [pad_emb for _ in range(self.n_gram)]
        while ind < l:
            ind += 1
            tok_emb = numpy.expand_dims(numpy.array(self.emb[idxes, cur_tok]), axis=1)
            tok_embs.append(tok_emb)
            del tok_embs[0]
            tok_emb = numpy.expand_dims(numpy.concatenate(tok_embs[-self.n_gram:], axis=1), axis=2)

            h = numpy.tanh(numpy.matmul(tok_emb, self.weight_i) + self.bias_i)
            h = numpy.mean(self.w_arr * h, axis=1)
            o = numpy.matmul(h, self.weight_o) + self.bias_o
            o = numpy.squeeze(o, axis=1)
            o = self.hardness * mean_var_norm(o)
            exp_o = numpy.exp(o)
            prob = exp_o / numpy.sum(exp_o, axis=-1, keepdims=True)
            cur_tok = (prob.cumsum(1) > numpy.random.rand(prob.shape[0])[:,None]).argmax(1)
            cur_prob = prob[idxes, cur_tok]
            ppl -= numpy.sum(numpy.log(cur_prob))
            tok_cnt += cur_prob.shape[0]

            seqs.append(cur_tok)
        print("GT Perplexity: %f" % (ppl / tok_cnt))

        return numpy.transpose(numpy.asarray(seqs, dtype="int32"))

class MetaLangV2(gym.Env):
    """
    Pseudo Langauge Generated from RNN models
    V: vocabulary size
    d: embedding size (input size)
    n: n-gram
    N: hidden size
    e: inverse of softmax - temporature
    L: maximum length
    """
    def __init__(self, 
            V=64, 
            n=3,
            d=16,
            N=64,
            e=0.20,
            L=4096):
        self.L = int(L)
        self.V = int(V)
        self.n = n
        self.d = d
        self.N = N
        self.hardness = 1.0/e
        assert n > 1 and V > 1 and N > 1 and L > 1 

    def data_generator(self, seed=None):
        nn = RandomNGram(n_emb = self.d, n_gram=self.n, n_hidden = self.N, n_vocab = self.V, hardness=self.hardness, seed=seed)
        tokens = nn.forward(self.L)[0]
        return tokens

    def batch_generator(self, batch_size, seed=None):
        nn = RandomNGram(batch = batch_size, n_emb = self.d, n_gram=self.n, n_hidden = self.N, n_vocab = self.V, hardness=self.hardness, seed=seed)
        tokens = nn.forward(self.L)
        return tokens

    def generate_text(self, size, output_stream):
        tokens = self.batch_generator(size)
        if(isinstance(output_stream, _io.TextIOWrapper)):
            need_close = False
        elif(isinstance(output_stream, str)):
            output_stream = open(output_stream, "w")
            need_close = True
        for i in range(tokens.shape[0]):
            output_stream.write("\t".join(map(str, tokens[i].tolist())))
            output_stream.write("\n")
        if(need_close):
            output_stream.close()

    def generate_npy(self, size, file_name):
        tokens = self.batch_generator(size)
        numpy.save(file_name, tokens)

    @property
    def VocabSize(self):
        return self.V

    @property
    def SepID(self):
        raise Exception("Not Defined")

    @property
    def MaskID(self):
        return -1

    @property
    def PaddingID(self):
        return -1
