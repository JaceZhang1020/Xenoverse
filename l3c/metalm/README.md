# Introduction for MetaLM

# Introduction

Meta language model generates sequences of repeating random integers with noises, aims at facilitating researches in Lifelong In-Context Learning.
$MetaLM(V, n, l, e, L)$ data generator generates the sequence by the following steps:

- 1.Generating elements $s_i \in S$ of length $l_i \sim Poisson(l), i\in[1,n]$ by random pick integers from $[1, V]$.
- 2.At iteration t, randomly sampling $s_t \in S$, disturb the seuqence $s_t$ acquring $\bar{s}_t$ by randomly replacing the number in $s$ with the other numbers or specific number 0. 
- 3.Contatenating $\bar{s}_t$ to $x$, iterate step 2 until $x$ reaching length of L, concatenating $s_t$ to $y$

A meta langauge model:  $p(y_{l+1} \| x_{l}, x_{l-1}, ..., x_{1})$;
The meta language model should be doing better and better as the $l$ increases;

### Motivation

Each $x$ can be regarded as an unknown language composed of $V$ tokens. Its complexity is described by $n$ and $l$. Unlike the pre-trained language model that has effectively memorize the pattern in its parameters, in this dataset, as the elements of $x$ is randomly generated, the model can not possibly predict $y_{l+1}$ by using only the short term context, but has to depend on long-term repeating to correctly memorize the usage of the language. <br>

Although we refer to this model as meta language model, we understand it is a relatively simplified version of language, since a real language (e.g., natural langauge, programming language) can not be totally random. Still, this dataset can be used as valuable benchmarks for long term memory and lifelong In-Context learning. <br>

# Introduction for MetaLM-2

The ultimate goal for Meta Language Model should be: \textbf{Learning a brand new language from context.}

In MetaLM-2, we further increase the complexity of the ``brand new language'' by introducing an Arbitrary Recurrent Neural Network (ARNN) $p_{\theta}$. We generate diverse pieces of language by utilizing this ARNN. A meta language model should precisely predict the output of this ARNN by Learning In Context, which is much harder than predict naively repeated sequences.
$MetaLM_v2(V, n, L)$ data generator generates the sequence by the following steps:

- 1.Randomly sample a n-dimensional ARNN $p_{\theta}$; Sample random embedding vectors for each token $e_i \in \mathbb{R}^{n}$, $i\in[1,V]$; Sample a random start token $x_0 \in [1,V]$
- 2.At step $l$, use $p_{\theta}$ to sample $x_l$ by processing $x_0$ to $x_{l-1}$, util $l \leq L$.

The meta langauge model try to predict:  $p(x_{l+1} \| x_{l}, x_{l-1}, ..., x_{1})$, and should do better as $l$ increases. 

### Motivation

A meta language model should be able to predict any sequence without any prior knowledge. In MetaLM-2, we introduce ARNNs, which represent different kinds of language.

# Install

```bash
pip install l3c
```

#### For local installation, execute following commands:

```bash
git clone https://github.com/PaddlePaddle/l3c
cd l3c
pip install .
```

# Quick Start

## Import

Import and create the meta language generator
```python
import gym
import l3c.metalm_v2

env = gym.make("meta-lm-v2", V=64, n=10, L=4096)
```

## Generating unlimited data

```python
obs, demo_label = env.data_generator() # generate observation & label for one sample
batch_obs, batch_label = env.batch_generator(batch_size) # generate observations & labels for batch of sample (shape of [batch_size, L])
```

## To generate data to file
```bash
python -m l3c.metalm.data_generator --version v1 --vocab_size 256 --elements_length 64 --elements_number 10 --error_rate 0.10 --sequence_length 4096 --samples 2 --output demo.v1.txt
python -m l3c.metalm.data_generator --version v2 --vocab_size 256 --hidden_size 16 --embedding_size 16 --sequence_length 4096 --samples 2 --output demo.v2.txt
```

# Demonstration

