import numpy as np
from Convolution import *
from Dense import *
from Loss import *
from Data_pipeline import *
import matplotlib.pyplot as plt

class CNN:
    def __init__(self, image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size):
        self.pool_size = pool_size
        self.stride = stride
        self.output_size = output_size
        flatten_shape = compute_flatten_shape(
                    image_inputsize,
                    kernel_shape,
                    num_kernels,
                    pool_size,
                    stride
                )
        self.convolution_layer = Convolution(kernel_shape, num_kernels)
        self.dense_layer = Dense(flatten_shape,output_size)
        self.loss = Loss()

    def forward(self, input_data):
        convu = self.convolution_layer.forward(input_data)
        maxpooled = self.convolution_layer.max_pool(convu, self.pool_size, self.stride)
        flattened = self.convolution_layer.flatten(maxpooled)
        self.logits = self.dense_layer.forward(flattened)
        return self.logits
    
    def calculate_loss(self, label):
        return self.loss.softmax_crossentropy(self.logits, label)
    
    def backward(self, label):
        output_gradient = self.loss.backward()
        dense_gradient = self.dense_layer.backward(output_gradient)
        convolution_gradient = self.convolution_layer.backward(dense_gradient)

    def update_parameters(self, learning_rate=0.01):
        self.dense_layer.update_parameters(learning_rate)
        self.convolution_layer.update_parameters(learning_rate)
        
    def update_parameters_with_grad(self, grad, learning_rate=0.01):
        self.convolution_layer.update_parameters_with_grad(grad[0], learning_rate)
        self.dense_layer.update_parameters_with_grad(grad[1], learning_rate)

    def get_parameters(self):
        conv_kernels = self.convolution_layer.kernels
        conv_biases = self.convolution_layer.biases
        dense_weights = self.dense_layer.weights
        dense_biases = self.dense_layer.biases
        return [conv_kernels, conv_biases, dense_weights, dense_biases]

    def set_parameters(self, conv_kernels, conv_biases, dense_weights, dense_biases):
        self.convolution_layer.set_parameters(conv_kernels, conv_biases)
        self.dense_layer.set_parameters(dense_weights, dense_biases)

    def predict(self, input_data):
        logits = self.forward(input_data)
        predicted_label = np.argmax(logits)
        return predicted_label
        
    # Stochastic Gradient Descent training method
    def train_SGD(self, x_train, y_train, epochs=10, learning_rate=0.01):
        loss_history = []
        indices = np.random.permutation(x_train.shape[0])
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            epoch_loss_sum = 0.0
            for i in indices:
                # input data and label
                label = y_train[i]
                input_data = x_train[i]

                # Forward pass
                self.forward(input_data)


                # calculate loss
                model_loss = self.calculate_loss(label)
                epoch_loss_sum += float(model_loss)

                # backpropagation
                self.backward(label)
                self.update_parameters(learning_rate=learning_rate)

            epoch_loss_avg = epoch_loss_sum / x_train.shape[0]
            loss_history.append(epoch_loss_avg)
            print("avg loss:", epoch_loss_avg)
        return loss_history
    

