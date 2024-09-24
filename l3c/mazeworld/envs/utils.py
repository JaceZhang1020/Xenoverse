import numpy as np
from numba import njit

@njit(cache=True)
def conv2d_numpy(input_data:np.ndarray, 
                 kernel:np.ndarray, 
                 stride=(1,1)):
    input_height, input_width = input_data.shape
    kernel_height, kernel_width = kernel.shape
    output_height = (input_height - kernel_height) // stride[0] + 1
    output_width = (input_width - kernel_width) // stride[1] + 1
    
    output_data = np.zeros((output_height, output_width))
    
    for i in range(0, input_height - kernel_height + 1, stride[0]):
        for j in range(0, input_width - kernel_width + 1, stride[1]):
            output_data[i // stride[0], j // stride[1]] = np.sum(input_data[i:i+kernel_height, j:j+kernel_width] * kernel)
    
    return output_data
