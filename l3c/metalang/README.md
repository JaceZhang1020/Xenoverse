# Introduction

Generating Randomized Pseudo Meta-Language for Benchmarking Long-Term Dependency In-Context Learning From Scratch


# Quick Start

## Keyboard Demonstrations

```bash
python -m l3c.metalang.generator --help 

optional arguments:
  -h, --help            show this help message and exit
  --version {v1,v2}     v1: generate with repeated random sequences; v2: generate with randomized n-gram NNs
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
import l3c.metalang

generator = gym.make("meta-language-v2")
sample = generator.data_generator()
batch_sample = generator.batch_generator(batch_size=8)
```
