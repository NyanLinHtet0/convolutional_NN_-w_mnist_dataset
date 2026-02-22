import numpy as np

class Convolution():
    def __init__(self, kernel_shape = (2,2), num_kernels=3):
        self.kernel_shape = kernel_shape
        self.depth = num_kernels
        self.kernels = np.random.randn(num_kernels,*self.kernel_shape)
        self.biases = np.random.randn(num_kernels,1,1)
        
    def forward(self, input_data):
        input_height, input_width = input_data.shape

        pad = 1
        kh, kw = self.kernel_shape

        padded = np.pad(input_data, pad, mode='constant', constant_values=0)

        output_height = input_height + 2*pad - kh + 1
        output_width  = input_width  + 2*pad - kw + 1

        self.output_shape = (self.depth, output_height, output_width)
        output = np.zeros(self.output_shape)

        for k in range(self.output_shape[0]):
            for i in range(output_height):
                for j in range(output_width):
                    region = padded[i:i+kh, j:j+kw]
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

