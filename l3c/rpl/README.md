# Introduction

Generating Randomized Pseudo-Language for Benchmarking Long-Term Dependency In-Context Learning From Scratch


# Quick Start

## Keyboard Demonstrations

You may try MazeWorld with your own keyboard with the following commands:
```bash
python -m l3c.rpl.data_generator --help 

optional arguments:
  -h, --help            show this help message and exit
  --version VERSION      currently support v1, v2, v1: generate with repeated random sequences; v2: generate with randomized n-gram NNs
  --vocab_size VOCAB_SIZE
  --embedding_size EMBEDDING_SIZE
  --hidden_size HIDDEN_SIZE
  --elements_length ELEMENTS_LENGTH
  --elements_number ELEMENTS_NUMBER
  --error_rate ERROR_RATE
  --n_gram N_GRAM
  --sequence_length SEQUENCE_LENGTH
  --samples SAMPLES
  --output_type {txt,npy}
  --output OUTPUT
```

## APIs

```python
import gym
import l3c.rpl

rpl_generator = gym.make("randomized-pseudo-language-v2")
sample = rpl_generator.data_generator()
batch_sample = rpl_generator.batch_generator(batch_size=8)
```
