import numpy as np


class Dense:
    def __init__(self, input_size=(25, 1), output_size=(3, 1)):
        self.input_size = input_size
        self.output_size = output_size
        fan_in = input_size[0]
        self.weights = np.random.randn(output_size[0], input_size[0]) * np.sqrt(2.0 / fan_in)
        self.biases = np.zeros((output_size[0], 1))
        self.gradient = np.zeros(self.weights.shape)
        self.bias_gradient = np.zeros(self.biases.shape)

    def forward(self, input_data):
        self.input_shape = input_data.shape
        self.input_data = input_data
        output = np.asarray(np.dot(self.weights, input_data) + self.biases)
        return output

    def backward(self, output_gradient):
        self.gradient += output_gradient @ self.input_data.T
        self.bias_gradient += output_gradient
        input_gradient = self.weights.T @ output_gradient
        return input_gradient

    def update_parameters(self, learning_rate=0.01):
        self.weights -= learning_rate * self.gradient
        self.biases -= learning_rate * self.bias_gradient
        self.gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.biases)

    def update_parameters_with_grad(self, grad, learning_rate=0.01):
        self.weights -= learning_rate * grad[0]
        self.biases -= learning_rate * grad[1]
        self.gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.biases)

    def return_weights(self):
        return self.weights

    def return_biases(self):
        return self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
        self.gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.biases)

    def get_gradient_parameters(self):
        return (self.gradient, self.bias_gradient)