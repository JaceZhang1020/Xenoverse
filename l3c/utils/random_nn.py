import time
import numpy

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
    avg_in = random.exponential(1.0, size=[n_in])
    avg_out = random.exponential(1.0, size=[n_out])
    weights = numpy.outer(avg_out, avg_in) + random.normal(size=[n_out, n_in])
    weights *= numpy.sqrt(6 / (n_in + n_out))
    bias = random.normal(size=[n_out]) * avg_out
    return weights, bias

class RandomMLP(object):
    '''
    A class for generating random MLPs with given parameters
    '''
    def __init__(self, n_inputs, n_outputs, 
                 n_hidden_layers=None, 
                 activation=None, 
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
        
        if activation is None:
            self.activation = lambda x:numpy.maximum(0.01*x, x)
        else:
            self.activation = activation
        
        # Initialize weights and biases to random values
        self.weights = []
        self.biases = []
        for i in range(len(self.hidden_units)-1):
            w, b = weights_and_biases(self.hidden_units[i], self.hidden_units[i+1])
            self.weights.append(w)
            self.biases.append(b)
            
    def forward(self, inputs):
        outputs = inputs
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            if(i >= len(self.weights)-1):
                outputs = self.activation(weight @ outputs + bias)
            else:
                outputs = weight @ outputs + bias
        return outputs
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)