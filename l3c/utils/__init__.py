from numpy import random
import time

def pseudo_random_seed(hyperseed=0):
    '''
    Generate a pseudo random seed based on current time and system random number
    '''
    timestamp = time.time_ns()
    system_random = int(random.random() * 100000000)
    pseudo_random = timestamp + system_random + hyperseed
    
    return pseudo_random % (4294967296)