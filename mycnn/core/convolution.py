import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .shape_utils import normalize_pair


class Convolution:
    def __init__(self, kernel_shape=3, width=3, input_depth=1):
        self.kernel_shape = normalize_pair(kernel_shape, "kernel_shape")
        self.kernel_size = self.kernel_shape[0]

        self.input_depth = int(input_depth)
        self.width = int(width)

        kh, kw = self.kernel_shape

        # He initialization for ReLU-based convolution layers
        fan_in = self.input_depth * kh * kw
        he_scale = np.sqrt(2.0 / fan_in)

        self.kernels = np.random.randn(self.width, self.input_depth, kh, kw) * he_scale
        self.biases = np.zeros((self.width, 1, 1))

        self.dl_dk = np.zeros_like(self.kernels)
        self.dl_db = np.zeros_like(self.biases)

    def _ensure_3d(self, input_data):
        if input_data.ndim == 2:
            return input_data[np.newaxis, :, :], True

        if input_data.ndim == 3:
            return input_data, False

        raise ValueError("input_data must be 2D (H, W) or 3D (C, H, W).")

    def _conv_windows(self, padded, kh, kw):
        return sliding_window_view(padded, (kh, kw), axis=(1, 2))

    def forward(self, input_data, pad=1):
        input_data, input_was_2d = self._ensure_3d(np.asarray(input_data))
        input_channels, input_height, input_width = input_data.shape

        if input_channels != self.input_depth:
            raise ValueError(
                f"Expected {self.input_depth} input channel(s), got {input_channels}."
            )

        self.input_was_2d = input_was_2d
        self.input_shape = input_data.shape
        self.input_data = input_data
        self.pad = int(pad)
        kh, kw = self.kernel_shape

        self.padded = np.pad(
            input_data,
            ((0, 0), (self.pad, self.pad), (self.pad, self.pad)),
            mode="constant",
            constant_values=0,
        )

        output_height = input_height + 2 * self.pad - kh + 1
        output_width = input_width + 2 * self.pad - kw + 1
        self.conv_out_shape = (self.width, output_height, output_width)

        self.conv_windows = self._conv_windows(self.padded, kh, kw)
        output = np.einsum('kcpq,cijpq->kij', self.kernels, self.conv_windows)
        output += self.biases[:, 0, 0][:, None, None]

        self.pre_relu = output.copy()
        self.relu_mask = (self.pre_relu > 0).astype(output.dtype)
        return np.maximum(0, output)

    def max_pool(self, input_data, pool_size=(2, 2), stride=(2, 2)):
        input_data, pooled_input_was_2d = self._ensure_3d(np.asarray(input_data))
        num_feature_maps, input_height, input_width = input_data.shape

        ph, pw = normalize_pair(pool_size, "pool_size")
        sh, sw = normalize_pair(stride, "stride")

        self.pool_size = (ph, pw)
        self.pool_stride = (sh, sw)
        self.pool_input_was_2d = pooled_input_was_2d

        output_height = (input_height - ph) // sh + 1
        output_width = (input_width - pw) // sw + 1

        self.pool_out_shape = (num_feature_maps, output_height, output_width)

        windows = sliding_window_view(input_data, (ph, pw), axis=(1, 2))
        windows = windows[:, ::sh, ::sw, :, :]
        windows = windows[:, :output_height, :output_width, :, :]
        self.pool_windows_shape = windows.shape

        flat_windows = windows.reshape(num_feature_maps, output_height, output_width, ph * pw)
        max_flat_idx = np.argmax(flat_windows, axis=-1)
        output = np.take_along_axis(flat_windows, max_flat_idx[..., None], axis=-1)[..., 0]

        base_rows = np.arange(output_height)[None, :, None] * sh
        base_cols = np.arange(output_width)[None, None, :] * sw
        row_offsets = (max_flat_idx // pw)
        col_offsets = (max_flat_idx % pw)

        self.max_rows = base_rows + row_offsets
        self.max_cols = base_cols + col_offsets
        self.max_mask = np.zeros_like(input_data)

        channel_idx = np.broadcast_to(
            np.arange(num_feature_maps)[:, None, None],
            (num_feature_maps, output_height, output_width),
        )
        np.add.at(self.max_mask, (channel_idx, self.max_rows, self.max_cols), 1)

        if pooled_input_was_2d:
            return output[0]

        return output

    def flatten(self, input_data):
        return np.asarray(input_data).flatten().reshape(-1, 1)

    def _backward_from_relu_grad(self, dl_drelu):
        kh, kw = self.kernel_shape

        dl_dconv = dl_drelu * self.relu_mask
        self.dl_db += np.sum(dl_dconv, axis=(1, 2), keepdims=True)
        self.dl_dk += np.einsum('kij,cijpq->kcpq', dl_dconv, self.conv_windows)

        input_gradient = np.zeros_like(self.padded)
        out_h, out_w = dl_dconv.shape[1], dl_dconv.shape[2]

        for a in range(kh):
            for b in range(kw):
                input_gradient[:, a:a + out_h, b:b + out_w] += np.einsum(
                    'kij,kc->cij', dl_dconv, self.kernels[:, :, a, b]
                )

        pad = self.pad
        if pad > 0:
            input_gradient = input_gradient[:, pad:-pad, pad:-pad]

        if self.input_was_2d:
            return input_gradient[0]

        return input_gradient

    def backward_no_pool(self, output_gradient):
        dl_drelu = np.asarray(output_gradient).reshape(self.conv_out_shape)
        return self._backward_from_relu_grad(dl_drelu)

    def backward(self, output_gradient, learning_rate=0.01):
        del learning_rate

        output_gradient = np.asarray(output_gradient).reshape(self.pool_out_shape)
        dl_drelu = np.zeros_like(self.max_mask)

        channel_idx = np.broadcast_to(
            np.arange(self.max_mask.shape[0])[:, None, None],
            self.pool_out_shape,
        )
        np.add.at(dl_drelu, (channel_idx, self.max_rows, self.max_cols), output_gradient)

        return self._backward_from_relu_grad(dl_drelu)

    def update_parameters(self, learning_rate=0.01):
        self.kernels -= learning_rate * self.dl_dk
        self.biases -= learning_rate * self.dl_db
        self.dl_dk = np.zeros_like(self.kernels)
        self.dl_db = np.zeros_like(self.biases)

    def update_parameters_with_grad(self, grad, learning_rate=0.01):
        self.kernels -= learning_rate * grad[0]
        self.biases -= learning_rate * grad[1]
        self.dl_dk = np.zeros_like(self.kernels)
        self.dl_db = np.zeros_like(self.biases)

    def return_kernels(self):
        return self.kernels

    def return_biases(self):
        return self.biases

    def set_parameters(self, kernels, biases):
        self.kernels = kernels
        self.biases = biases
        self.width = kernels.shape[0]
        self.input_depth = kernels.shape[1]
        self.kernel_shape = (kernels.shape[2], kernels.shape[3])
        self.kernel_size = self.kernel_shape[0]
        self.dl_dk = np.zeros_like(self.kernels)
        self.dl_db = np.zeros_like(self.biases)

    def get_gradient_parameters(self):
        return (self.dl_dk, self.dl_db)
