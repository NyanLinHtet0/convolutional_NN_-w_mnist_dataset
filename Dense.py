import numpy as np

class Dense():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
        
    def forward(self, input_data):
        #input_data shape (input_size, number_of_samples)
        self.input_shape = input_data.shape
        #output_data shape (output_size, number_of_samples)
        output = np.dot(self.weights, input_data) + self.biases
        return output
    
    def backward(self, output_gradient, learning_rate):
        pass

