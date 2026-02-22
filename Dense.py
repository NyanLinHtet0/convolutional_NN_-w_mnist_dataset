import numpy as np

class Dense():
    def __init__(self, input_size=(25,1), output_size=(3,1)):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size[0],input_size[0])
        self.biases = np.random.randn(output_size[0], 1)
        
    def forward(self, input_data):
        #input_data shape (input_size, number_of_samples)
        self.input_shape = input_data.shape
        self.input_data = input_data
        #output_data shape (output_size, number_of_samples)
        output = np.asarray(np.dot(self.weights, input_data) + self.biases)
        return output
    
    def backward(self, output_gradient, learning_rate=0.01):
        self.gradient = np.zeros(self.weights.shape)

        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                self.gradient[i,j] = output_gradient[i,0]* self.input_data[j,0]
        input_gradient = self.weights.T @ output_gradient
        return input_gradient
        

