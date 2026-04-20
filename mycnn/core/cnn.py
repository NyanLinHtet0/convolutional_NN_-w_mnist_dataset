import numpy as np

from .dense import Dense
from .loss import Loss
from .cnn_helper import (
    normalize_widths,
    infer_input_depth,
    build_convolution_layers,
    compute_flatten_shape,
)


class CNN:
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
        self.pool_size = pool_size
        self.stride = stride
        self.output_size = output_size

        self.widths = normalize_widths(widths)
        self.depth = len(self.widths)
        self.input_depth = infer_input_depth(image_inputsize)

        self.hidden_layerwidths = self._normalize_hidden_layerwidths(hidden_layerwidths)

        flatten_shape = compute_flatten_shape(
            image_inputsize=image_inputsize,
            kernel_shape=kernel_shape,
            widths=self.widths,
            pool_size=pool_size,
            stride=stride,
        )

        self.flatten_shape = flatten_shape

        self.convolution_layers = build_convolution_layers(
            kernel_shape=kernel_shape,
            widths=self.widths,
            input_depth=self.input_depth,
        )

        self.hidden_layers, self.output_layer = self._build_dense_stack(
            input_size=flatten_shape,
            hidden_layerwidths=self.hidden_layerwidths,
            output_size=output_size,
        )

        # convenience / compatibility
        self.dense_layers = self.hidden_layers + [self.output_layer]
        self.dense_layer = self.output_layer

        self.loss = Loss()

    def _normalize_hidden_layerwidths(self, hidden_layerwidths):
        if hidden_layerwidths is None:
            return []

        if isinstance(hidden_layerwidths, int):
            hidden_layerwidths = [hidden_layerwidths]
        elif not isinstance(hidden_layerwidths, (list, tuple)):
            raise TypeError("hidden_layerwidths must be None, an int, list, or tuple")

        result = [int(w) for w in hidden_layerwidths]

        for width in result:
            if width <= 0:
                raise ValueError("hidden_layerwidths values must be positive integers")

        return result

    def _build_dense_stack(self, input_size, hidden_layerwidths, output_size):
        hidden_layers = []
        current_input_size = (int(input_size[0]), 1)

        for hidden_width in hidden_layerwidths:
            hidden_layer = Dense(
                input_size=current_input_size,
                output_size=(int(hidden_width), 1),
            )
            hidden_layers.append(hidden_layer)
            current_input_size = (int(hidden_width), 1)

        output_layer = Dense(
            input_size=current_input_size,
            output_size=output_size,
        )

        return hidden_layers, output_layer

    def forward(self, input_data):
        current = input_data

        # conv stack first
        for layer in self.convolution_layers:
            current = layer.forward(current)

        # only one pooling stage at the end
        current = self.convolution_layers[-1].max_pool(current, self.pool_size, self.stride)

        flattened = self.convolution_layers[-1].flatten(current)
        current = flattened

        # hidden dense layers with ReLU
        self.hidden_relu_masks = []
        for layer in self.hidden_layers:
            current = layer.forward(current)
            relu_mask = (current > 0).astype(current.dtype)
            self.hidden_relu_masks.append(relu_mask)
            current = np.maximum(0, current)

        # final output layer (no ReLU here)
        self.logits = self.output_layer.forward(current)
        return self.logits

    def calculate_loss(self, label):
        return self.loss.softmax_crossentropy(self.logits, label)

    def backward(self):
        grad = self.loss.backward()

        # output layer first
        grad = self.output_layer.backward(grad)

        # hidden layers backward with ReLU masks
        for layer_idx in reversed(range(len(self.hidden_layers))):
            grad = grad * self.hidden_relu_masks[layer_idx]
            grad = self.hidden_layers[layer_idx].backward(grad)

        # conv stack backward
        for layer_idx in reversed(range(len(self.convolution_layers))):
            layer = self.convolution_layers[layer_idx]

            if layer_idx == len(self.convolution_layers) - 1:
                grad = layer.backward(grad)           # last conv used pooling
            else:
                grad = layer.backward_no_pool(grad)   # earlier convs did not

        return grad

    def update_parameters(self, learning_rate=0.01):
        for layer in self.convolution_layers:
            layer.update_parameters(learning_rate)

        for layer in self.hidden_layers:
            layer.update_parameters(learning_rate)

        self.output_layer.update_parameters(learning_rate)

    def update_parameters_with_grad(self, grad, learning_rate=0.01):
        conv_grad = grad[0]
        dense_grad = grad[1]

        if len(self.convolution_layers) == 1:
            if isinstance(conv_grad, tuple):
                conv_grad = [conv_grad]
            elif not isinstance(conv_grad, list):
                conv_grad = [conv_grad]

        if len(self.dense_layers) == 1:
            if isinstance(dense_grad, tuple):
                dense_grad = [dense_grad]
            elif not isinstance(dense_grad, list):
                dense_grad = [dense_grad]

        if len(conv_grad) != len(self.convolution_layers):
            raise ValueError(
                f"Expected {len(self.convolution_layers)} convolution gradients, "
                f"but got {len(conv_grad)}."
            )

        if len(dense_grad) != len(self.dense_layers):
            raise ValueError(
                f"Expected {len(self.dense_layers)} dense gradients, "
                f"but got {len(dense_grad)}."
            )

        for layer, layer_grad in zip(self.convolution_layers, conv_grad):
            layer.update_parameters_with_grad(layer_grad, learning_rate)

        for layer, layer_grad in zip(self.dense_layers, dense_grad):
            layer.update_parameters_with_grad(layer_grad, learning_rate)

    def get_parameters(self):
        conv_kernels = [layer.kernels for layer in self.convolution_layers]
        conv_biases = [layer.biases for layer in self.convolution_layers]

        dense_weights = [layer.weights for layer in self.dense_layers]
        dense_biases = [layer.biases for layer in self.dense_layers]

        if len(self.convolution_layers) == 1:
            conv_kernels = conv_kernels[0]
            conv_biases = conv_biases[0]

        if len(self.dense_layers) == 1:
            dense_weights = dense_weights[0]
            dense_biases = dense_biases[0]

        return [conv_kernels, conv_biases, dense_weights, dense_biases]

    def set_parameters(self, conv_kernels, conv_biases, dense_weights, dense_biases):
        if len(self.convolution_layers) == 1 and not isinstance(conv_kernels, (list, tuple)):
            conv_kernels = [conv_kernels]
            conv_biases = [conv_biases]

        if len(self.dense_layers) == 1 and not isinstance(dense_weights, (list, tuple)):
            dense_weights = [dense_weights]
            dense_biases = [dense_biases]

        if len(conv_kernels) != len(self.convolution_layers):
            raise ValueError(
                f"Expected {len(self.convolution_layers)} convolution kernel sets, "
                f"but got {len(conv_kernels)}."
            )

        if len(conv_biases) != len(self.convolution_layers):
            raise ValueError(
                f"Expected {len(self.convolution_layers)} convolution bias sets, "
                f"but got {len(conv_biases)}."
            )

        if len(dense_weights) != len(self.dense_layers):
            raise ValueError(
                f"Expected {len(self.dense_layers)} dense weight sets, "
                f"but got {len(dense_weights)}."
            )

        if len(dense_biases) != len(self.dense_layers):
            raise ValueError(
                f"Expected {len(self.dense_layers)} dense bias sets, "
                f"but got {len(dense_biases)}."
            )

        for layer, kernels, biases in zip(self.convolution_layers, conv_kernels, conv_biases):
            layer.set_parameters(kernels, biases)

        for layer, weights, biases in zip(self.dense_layers, dense_weights, dense_biases):
            layer.set_parameters(weights, biases)

    def predict(self, input_data):
        logits = self.forward(input_data)
        predicted_label = np.argmax(logits)
        return predicted_label

    def save_parameters(self, filepath):
        conv_kernels = [layer.kernels for layer in self.convolution_layers]
        conv_biases = [layer.biases for layer in self.convolution_layers]
        dense_weights = [layer.weights for layer in self.dense_layers]
        dense_biases = [layer.biases for layer in self.dense_layers]

        conv_kernel_obj = np.empty(len(conv_kernels), dtype=object)
        conv_bias_obj = np.empty(len(conv_biases), dtype=object)
        dense_weight_obj = np.empty(len(dense_weights), dtype=object)
        dense_bias_obj = np.empty(len(dense_biases), dtype=object)

        for i in range(len(conv_kernels)):
            conv_kernel_obj[i] = conv_kernels[i]
            conv_bias_obj[i] = conv_biases[i]

        for i in range(len(dense_weights)):
            dense_weight_obj[i] = dense_weights[i]
            dense_bias_obj[i] = dense_biases[i]

        np.savez_compressed(
            filepath,
            conv_kernels=conv_kernel_obj,
            conv_biases=conv_bias_obj,
            dense_weights=dense_weight_obj,
            dense_biases=dense_bias_obj,
            widths=np.array(self.widths, dtype=int),
            hidden_layerwidths=np.array(self.hidden_layerwidths, dtype=int),
        )

    def load_parameters(self, filepath):
        data = np.load(filepath, allow_pickle=True)

        conv_kernels = list(data["conv_kernels"])
        conv_biases = list(data["conv_biases"])

        dense_weights_raw = data["dense_weights"]
        dense_biases_raw = data["dense_biases"]

        if isinstance(dense_weights_raw, np.ndarray) and dense_weights_raw.dtype == object:
            dense_weights = list(dense_weights_raw)
        else:
            dense_weights = [dense_weights_raw]

        if isinstance(dense_biases_raw, np.ndarray) and dense_biases_raw.dtype == object:
            dense_biases = list(dense_biases_raw)
        else:
            dense_biases = [dense_biases_raw]

        if len(conv_kernels) != len(self.convolution_layers):
            raise ValueError(
                f"Saved file has {len(conv_kernels)} convolution layers, "
                f"but current model has {len(self.convolution_layers)}."
            )

        if len(dense_weights) != len(self.dense_layers):
            raise ValueError(
                f"Saved file has {len(dense_weights)} dense layers, "
                f"but current model has {len(self.dense_layers)}."
            )

        self.set_parameters(conv_kernels, conv_biases, dense_weights, dense_biases)

    def train_SGD(self, x_train, y_train, epochs=10, learning_rate=0.01):
        loss_history = []

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}")
            epoch_loss_sum = 0.0
            indices = np.random.permutation(x_train.shape[0])

            for i in indices:
                label = y_train[i]
                input_data = x_train[i]

                self.forward(input_data)
                model_loss = self.calculate_loss(label)
                epoch_loss_sum += float(model_loss)

                self.backward()
                self.update_parameters(learning_rate=learning_rate)

            epoch_loss_avg = epoch_loss_sum / x_train.shape[0]
            loss_history.append(epoch_loss_avg)
            print("avg loss:", epoch_loss_avg)

        return loss_history