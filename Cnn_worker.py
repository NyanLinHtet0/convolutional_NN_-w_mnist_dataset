
from Convolution import *
from Dense import Dense
from Loss import Loss
from cnn import CNN




class CNN_worker(CNN):
    def __init__(self, image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size):
        super().__init__(image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size)
    
    def set_parameters(self, conv_kernels, conv_biases, dense_weights, dense_biases):   
        self.convolution_layer.set_parameters(conv_kernels, conv_biases)
        self.dense_layer.set_parameters(dense_weights, dense_biases)

    def train_mini_batch(self, x_train, y_train, index_array, learning_rate=0.01):
        loss_sum = 0.0
        for i in index_array:
            # input data and label
            label = y_train[i]
            input_data = x_train[i]
            # Forward pass
            self.forward(input_data)
            # calculate loss
            model_loss = self.calculate_loss(label)
            loss_sum += float(model_loss)
            # backpropagation
            self.backward(label)

        dense_gradient = self.dense_layer.get_gradient_parameters()
        convu_gradient = self.convolution_layer.get_gradient_parameters()

        return loss_sum, convu_gradient, dense_gradient
    