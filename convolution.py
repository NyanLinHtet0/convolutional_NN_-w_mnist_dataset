import numpy as np

class Convolution():
    def __init__(self, kernel_shape = (2,2), num_kernels=1):
        self.kernel_shape = kernel_shape
        self.depth = num_kernels
        self.kernels = np.random.randn(num_kernels,*self.kernel_shape)
        self.biases = np.random.randn(num_kernels,1,1)
        
    def forward(self, input_data):
        #input matrix shape (height, width)
        input_height, input_width = input_data.shape
        #output matrix shape (num_kernels, output_height, output_width)
        output_height = input_height
        output_width = input_width
        #create output matrix
        self.output_shape = (self.depth, output_height, output_width)
        output = np.zeros(self.output_shape)
        #pad input data with zeroes around the borders
        np.pad(input_data, 1, mode='constant', constant_values=0)
        #perform convolution
        for k in range(self.output_shape[0]):
            for i in range(output_height):
                for j in range(output_width):
                    region = input_data[i:i+self.kernel_shape[0], j:j+self.kernel_shape[1]]
                    output[k, i, j] = np.sum(region * self.kernels[k]) + self.biases[k, 0, 0]
        return output

    def max_pool(self, input_data, pool_size=(2,2), stride=(2,2)):
        #input matrix shape (num_kernels, height, width)
        num_kernels, input_height, input_width = input_data.shape
        #output matrix shape (num_kernels, output_height, output_width)
        output_height = input_height // pool_size[0]
        output_width = input_width // pool_size[1]
        #create output matrix
        self.output_shape = (num_kernels, output_height, output_width)
        output = np.zeros(self.output_shape)
        #perform max pooling
        for k in range(num_kernels):
            for i in range(output_height):
                for j in range(output_width):
                    region = input_data[k, i*pool_size[0]:(i+1)*pool_size[0], j*pool_size[1]:(j+1)*pool_size[1]]
                    output[k, i, j] = np.max(region)
        return output
    
    def flatten(self, input_data):
        output = input_data.flatten().reshape(-1, 1)
        return output

