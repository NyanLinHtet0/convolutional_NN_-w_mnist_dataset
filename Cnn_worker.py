
from Convolution import *
from Dense import Dense
from Loss import Loss
from cnn import CNN

GLOBAL_WORKER = None
GLOBAL_X = None
GLOBAL_Y = None

def _run_worker_job(args):
    (
        image_inputsize,
        kernel_shape,
        num_kernels,
        pool_size,
        stride,
        output_size,
        idx_arr,
        conv_k,
        conv_b,
        dense_w,
        dense_b,
        lr
    ) = args
    GLOBAL_WORKER.set_parameters(conv_k, conv_b, dense_w, dense_b)

    return GLOBAL_WORKER.train_mini_batch(
        GLOBAL_X,
        GLOBAL_Y,
        idx_arr,
        learning_rate=lr
    )


def worker_init(x_train, y_train,
                image_inputsize,
                kernel_shape,
                num_kernels,
                pool_size,
                stride,
                output_size):

    global GLOBAL_X, GLOBAL_Y, GLOBAL_WORKER

    GLOBAL_X = x_train
    GLOBAL_Y = y_train

    GLOBAL_WORKER = CNN_worker(
        image_inputsize=image_inputsize,
        kernel_shape=kernel_shape,
        num_kernels=num_kernels,
        pool_size=pool_size,
        stride=stride,
        output_size=output_size
    )



class CNN_worker(CNN):
    def __init__(self, image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size):
        super().__init__(image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size)

    def set_parameters(self, conv_kernels, conv_biases, dense_weights, dense_biases):   
        self.convolution_layer.set_parameters(conv_kernels, conv_biases)
        self.dense_layer.set_parameters(dense_weights, dense_biases)
    
    # def set_parameters(self, conv_kernels, conv_biases, dense_weights, dense_biases):   
    #     self.convolution_layer.set_parameters(conv_kernels, conv_biases)
    #     self.dense_layer.set_parameters(dense_weights, dense_biases)

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
            self.backward()

        dense_gradient = self.dense_layer.get_gradient_parameters()
        convu_gradient = self.convolution_layer.get_gradient_parameters()
        num_samples = int(len(index_array))
        return loss_sum, convu_gradient, dense_gradient, num_samples
    