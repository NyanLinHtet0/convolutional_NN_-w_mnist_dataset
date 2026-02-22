import numpy as np

class Convolution():
    def __init__(self, kernel_shape = (3,3), num_kernels=3):
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
        print(padded.shape)
        self.output_shape = (self.depth, output_height, output_width)
        output = np.zeros(self.output_shape)
        print(f'output shape: {output.shape}')
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
        #mask matrix for backward pass
        self.max_mask = np.zeros_like(input_data)
        #perform max pooling
        for k in range(num_kernels):
            for i in range(output_height):
                for j in range(output_width):
                    region = input_data[k, i*pool_size[0]:(i+1)*pool_size[0], j*pool_size[1]:(j+1)*pool_size[1]]
                    output[k, i, j] = np.max(region)
                    # find index inside region
                    max_idx = np.unravel_index(np.argmax(region), region.shape)

                    # convert local index → global index
                    global_i = i*pool_size[0] + max_idx[0]
                    global_j = j*pool_size[1] + max_idx[1]
                    # store mask
                    self.max_mask[k, global_i, global_j] = 1
        return output
    
    def flatten(self, input_data):
        output = input_data.flatten().reshape(-1, 1)
        return output
    
    def backward(self, output_gradient, learning_rate=0.01):    
        #unflatten the output gradient
        output_gradient = output_gradient.reshape(self.output_shape)

        # gradient to pass down: same shape as pre-pool input (same as mask)
        input_gradient = np.zeros_like(self.max_mask)

        for k in range(self.depth):
            for i in range(output_gradient.shape[1]):      # pooled height (5)
                for j in range(output_gradient.shape[2]):  # pooled width  (5)
                    region = self.max_mask[k, i*2:(i+1)*2, j*2:(j+1)*2]
                    input_gradient[k, i*2:(i+1)*2, j*2:(j+1)*2] = region * output_gradient[k, i, j]

        return input_gradient
