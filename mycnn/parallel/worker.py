from ..core.cnn import CNN

GLOBAL_WORKER = None
GLOBAL_X = None
GLOBAL_Y = None


def run_worker_job(args):
    (
        image_inputsize,
        kernel_shape,
        widths,
        hidden_layerwidths,
        pool_size,
        stride,
        output_size,
        idx_arr,
        conv_k,
        conv_b,
        dense_w,
        dense_b,
        lr,
    ) = args

    GLOBAL_WORKER.set_parameters(conv_k, conv_b, dense_w, dense_b)

    return GLOBAL_WORKER.train_mini_batch(
        GLOBAL_X,
        GLOBAL_Y,
        idx_arr,
        learning_rate=lr,
    )


def worker_init(
    x_train,
    y_train,
    image_inputsize,
    kernel_shape,
    widths,
    hidden_layerwidths,
    pool_size,
    stride,
    output_size,
):
    global GLOBAL_X, GLOBAL_Y, GLOBAL_WORKER

    GLOBAL_X = x_train
    GLOBAL_Y = y_train

    GLOBAL_WORKER = CNNWorker(
        image_inputsize=image_inputsize,
        kernel_shape=kernel_shape,
        widths=widths,
        hidden_layerwidths=hidden_layerwidths,
        pool_size=pool_size,
        stride=stride,
        output_size=output_size,
    )


class CNNWorker(CNN):
    def __init__(
        self,
        image_inputsize,
        kernel_shape,
        widths,
        pool_size,
        stride,
        output_size,
        hidden_layerwidths=None,
    ):
        super().__init__(
            image_inputsize=image_inputsize,
            kernel_shape=kernel_shape,
            widths=widths,
            pool_size=pool_size,
            stride=stride,
            output_size=output_size,
            hidden_layerwidths=hidden_layerwidths,
        )

    def get_gradient_parameters(self):
        conv_gradients = [
            layer.get_gradient_parameters() for layer in self.convolution_layers
        ]
        dense_gradients = [
            layer.get_gradient_parameters() for layer in self.dense_layers
        ]
        return conv_gradients, dense_gradients

    def train_mini_batch(self, x_train, y_train, index_array, learning_rate=0.01):
        del learning_rate
        loss_sum = 0.0

        for i in index_array:
            label = y_train[i]
            input_data = x_train[i]

            self.forward(input_data)
            model_loss = self.calculate_loss(label)
            loss_sum += float(model_loss)
            self.backward()

        conv_gradients, dense_gradients = self.get_gradient_parameters()
        num_samples = int(len(index_array))

        return loss_sum, conv_gradients, dense_gradients, num_samples