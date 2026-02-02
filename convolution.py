import numpy as np

class Convolution():
    def __init__(self, kernel_shape = (2,2), num_kernels=2):
        self.kernel_shape = kernel_shape
        self.depth = num_kernels
        self.kernels = np.random.randn(num_kernels,*self.kernel_shape)
        self.biases = np.random.randn(num_kernels,1,1)
        
    def forward(self, input_data):
        #input matrix shape (height, width)
        input_height, input_width = input_data.shape
        #output matrix shape (num_kernels, output_height, output_width)
        output_height = input_height - self.kernel_shape[0] + 1
        output_width = input_width - self.kernel_shape[1] + 1
        #create output matrix
        self.output_shape = (self.depth, output_height, output_width)
        output = np.zeros(self.output_shape)
        #perform convolution
        for k in range(self.output_shape[0]):
            for i in range(output_height):
                for j in range(output_width):
                    region = input_data[i:i+self.kernel_shape[0], j:j+self.kernel_shape[1]]
                    output[k, i, j] = np.sum(region * self.kernels[k]) + self.biases[k, 0, 0]
        return output
    
Convolution_instance = Convolution()
print(Convolution_instance.forward(np.random.randn(10,10)))