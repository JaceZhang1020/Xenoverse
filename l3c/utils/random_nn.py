### A random MLP generator

import time
import numpy
import re
from numpy import random


def pseudo_random_seed(hyperseed=0):
    '''
    Generate a pseudo random seed based on current time and system random number
    '''
    timestamp = time.time_ns()
    system_random = int(random.random() * 100000000)
    pseudo_random = timestamp + system_random + hyperseed
    
    return pseudo_random % (4294967296)

def weights_and_biases(n_in, n_out):
    avg_in = random.normal(loc=0.0, scale=1.0, size=[n_in])
    avg_out = random.normal(loc=0.0, scale=1.0, size=[n_out])
    weights = numpy.outer(avg_out, avg_in) + random.normal(size=[n_out, n_in])
    weights = weights * numpy.sqrt(6.0 / (n_in + n_out))
    bias = 0.1 * random.normal(size=[n_out]) * avg_out
    return weights, bias

def actfunc(name):
    name = name.lower()
    if(name=='sigmoid'):
        return lambda x: 1/(1+numpy.exp(-x))
    elif(name.find('leakyrelu') >= 0):
        return lambda x: numpy.maximum(0.01*x, x)
    elif(name.find('bounded') >= 0):
        pattern = r"bounded\(([-+]?\d*\.\d+|[-+]?\d+),\s*([-+]?\d*\.\d+|[-+]?\d+)\)"
        match = re.match(pattern, name)
        if match:
            B = float(match.group(1).strip())
            T = float(match.group(2).strip())
        else:
            raise ValueError("Bounded support only BOUNDED(min,max) type")
        k = (T - B) / 2
        return lambda x: k*numpy.tanh(x/k) + k + B
    elif(name == 'sin'):
        return lambda x: numpy.concat([numpy.sin(x[:len(x)//2]), numpy.cos(x[len(x)//2:])], axis=-1)
    elif(name == 'none'):
        return lambda x: x

class RandomMLP(object):
    '''
    A class for generating random MLPs with given parameters
    '''
    def __init__(self, n_inputs, n_outputs, 
                 n_hidden_layers=None, 
                 hidden_activation=None, 
                 output_activation=None,
                 seed=None):
        # Set the seed for the random number generator
        if seed is None:
            seed = pseudo_random_seed()
        random.seed(seed)

        # Set the number of hidden units and activation function
        self.hidden_units = [n_inputs]
        if n_hidden_layers is not None:
            if(isinstance(n_hidden_layers, list)):
                self.hidden_units += n_hidden_layers
            elif(isinstance(n_hidden_layers, numpy.ndarray)):
                self.hidden_units += n_hidden_layers.tolist()
            elif(isinstance(n_hidden_layers, tuple)):
                self.hidden_units += list(n_hidden_layers)
            elif(isinstance(n_hidden_layers, int)):
                self.hidden_units.append(n_hidden_layers)
            else:
                raise TypeError(f"Invalid input type of n_hidden_layers: {type(n_hidden_layers)}")
        self.hidden_units.append(n_outputs)
        
        self.activation = []

        if hidden_activation is None:
            for _ in range(len(self.hidden_units)-2):
                self.activation.append(actfunc('sigmoid'))
        else:
            for hidden_act in hidden_activation:
                self.activation.append(actfunc(hidden_act))
        if output_activation is None:
            self.activation = actfunc('none')
        else:
            self.activation = actfunc(output_activation)
        
        # Initialize weights and biases to random values
        self.weights = []
        self.biases = []
        for i in range(len(self.hidden_units)-1):
            w, b = weights_and_biases(self.hidden_units[i], self.hidden_units[i+1])
            self.weights.append(w)
            if(i == len(self.hidden_units)-2):
                b *= 0
            self.biases.append(b)
            
    def forward(self, inputs):
        outputs = inputs
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            outputs = self.activation(weight @ outputs + bias)
        return outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)