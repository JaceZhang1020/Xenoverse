# Introduction

AnyMDP generates random Markov Decision Processes (MDPs) and provides a set of environments for In-Context Reinforcement Learning (ICRL) and Meta-RL.

# Install

```bash
pip install l3c[anymdp]
```

#### For local installation, execute following commands:

```bash
git clone https://github.com/FutureAGI/L3C
cd L3C
pip install .[anymdp]
```

# Quick Start

## Import

Import and create the AnyMDP environment with 
```python
import gym
import l3c.anymdp

env = gym.make("anymdp-v0", max_steps=5000)
```

## Sampling an AnyMDP task
```python
from l3c.anymdp import AnyMDPTaskSampler

task = AnyMDPTaskSampler(
        state_space=8, # number of states
        action_space=5, # number of actions
        state_sparsity=0.5, # transition matrix sparsity
        reward_sparsity=0.5, # reward matrix sparsity
        )
env.set_task(task)
env.reset()
```

You might resample a MDP task by keeping the transitions unchanged but sample a new reward matrix by

```python
from l3c.anymdp import Resampler
new_task = Resampler(task)
```

## Running the built-in MDP solver
```python
from l3c.anymdp import AnyMDPSolverOpt

solver = AnyMDPSolverOpt(env)  # AnyMDPSolverOpt solves the MDP with ground truth rewards and transition matrix
state, info = env.reset()
done = False
while not done:
    action = solver.policy(state)
    state, reward, done, info = env.step(action)
```

In case you do not want the ground truth rewards and transition to be leaked to the agent, use the AnyMDPSolverQ instead. This solver inplement a ideal environment modeling and a planning-based policy.

```
from l3c.anymdp import AnyMDPSolverQ, AnyMDPSolverMBRL

 # AnyMDPSolverQ solves the MDP with Table Q-Learning, without seeing the ground truth rewards and transition
solver = AnyMDPSolverQ(env) 

# AnyMDPSolverMBRL solves the MDP with Ideal Environment Modeling and Planning, without seeing the ground truth rewards and transition
#solver = AnyMDPSolverMBRL(env) 

state, info = env.reset()
done = False
while not done:
    action = solver.policy(state)
    state, reward, done, info = env.step(action)
    solver.learner(state, action, next_state, reward, done) # update the learner
```