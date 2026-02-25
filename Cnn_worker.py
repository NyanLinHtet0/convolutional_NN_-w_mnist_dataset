
from Convolution import *
from Dense import Dense
from Loss import Loss
from cnn import CNN




class CNN_worker(CNN):
    def __init__(self, image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size):
        super().__init__(image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size)

    def train_mini_batch_worker(self, conv_param, dense_param, x_shard, y_shard, learning_rate=0.01):
        
        return
    
    def set_parameters(self, conv_kernels, conv_biases, dense_weights, dense_biases):   
        self.convolution_layer.set_parameters(conv_kernels, conv_biases)
        self.dense_layer.set_parameters(dense_weights, dense_biases)

        