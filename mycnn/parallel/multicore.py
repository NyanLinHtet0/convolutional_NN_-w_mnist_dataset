import os
import numpy as np
import multiprocessing as mp

from ..core.cnn import CNN
from .worker import worker_init, run_worker_job


class CNNMultiCore(CNN):
    def train_batches(
        self,
        x_train,
        y_train,
        epochs=10,
        mini_batch_size=4,
        learning_rate=0.01,
    ):
        num_workers = 16
        # num_workers = max(1, (os.cpu_count() or 1))
        batch_loss_history = []
        print(f"Using {num_workers} worker processes for training.")

        image_inputsize = tuple(x_train[0].shape)
        ctx = mp.get_context("spawn")

        batch_size = mini_batch_size * num_workers
        epoch_loss_history = []

        with ctx.Pool(
            processes=num_workers,
            initializer=worker_init,
            initargs=(
                x_train,
                y_train,
                image_inputsize,
                self.convolution_layers[0].kernel_shape,
                self.widths,
                self.hidden_layerwidths,
                self.pool_size,
                self.stride,
                self.output_size,
            ),
        ) as pool:
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")
                epoch_loss_sum = 0
                epoch_samples = 0

                indices = np.random.permutation(np.arange(x_train.shape[0]))

                for start in range(0, len(indices), batch_size):
                    batch_indices = indices[start:start + batch_size]
                    if batch_indices.size == 0:
                        continue

                    conv_k, conv_b, dense_w, dense_b = self.get_parameters()

                    if len(self.convolution_layers) == 1:
                        if not isinstance(conv_k, list):
                            conv_k = [conv_k]
                        if not isinstance(conv_b, list):
                            conv_b = [conv_b]

                    if len(self.dense_layers) == 1:
                        if not isinstance(dense_w, list):
                            dense_w = [dense_w]
                        if not isinstance(dense_b, list):
                            dense_b = [dense_b]

                    worker_args = []
                    for worker_id in range(num_workers):
                        worker_start = worker_id * mini_batch_size
                        worker_end = min(worker_start + mini_batch_size, batch_indices.shape[0])
                        index_arr = batch_indices[worker_start:worker_end]

                        if worker_start >= worker_end:
                            break

                        worker_args.append((
                            image_inputsize,
                            self.convolution_layers[0].kernel_shape,
                            self.widths,
                            self.hidden_layerwidths,
                            self.pool_size,
                            self.stride,
                            self.output_size,
                            index_arr,
                            conv_k,
                            conv_b,
                            dense_w,
                            dense_b,
                            learning_rate,
                        ))

                    results = pool.map(run_worker_job, worker_args)

                    total_samples = 0
                    loss_sum = 0

                    conv_k_grad_sum = [np.zeros_like(k) for k in conv_k]
                    conv_b_grad_sum = [np.zeros_like(b) for b in conv_b]
                    dense_w_grad_sum = [np.zeros_like(w) for w in dense_w]
                    dense_b_grad_sum = [np.zeros_like(b) for b in dense_b]

                    for loss, conv_param, dense_param, num_samples in results:
                        loss_sum += loss
                        total_samples += num_samples

                        for layer_idx, (dk, db) in enumerate(conv_param):
                            conv_k_grad_sum[layer_idx] += dk
                            conv_b_grad_sum[layer_idx] += db

                        for layer_idx, (dw, db) in enumerate(dense_param):
                            dense_w_grad_sum[layer_idx] += dw
                            dense_b_grad_sum[layer_idx] += db

                    if total_samples == 0:
                        continue

                    conv_param = []
                    for layer_idx in range(len(conv_k_grad_sum)):
                        conv_param.append((
                            conv_k_grad_sum[layer_idx] / total_samples,
                            conv_b_grad_sum[layer_idx] / total_samples,
                        ))

                    dense_param = []
                    for layer_idx in range(len(dense_w_grad_sum)):
                        dense_param.append((
                            dense_w_grad_sum[layer_idx] / total_samples,
                            dense_b_grad_sum[layer_idx] / total_samples,
                        ))

                    self.update_parameters_with_grad(
                        (conv_param, dense_param),
                        learning_rate=learning_rate,
                    )

                    batch_loss_avg = loss_sum / total_samples
                    batch_loss_history.append(batch_loss_avg)

                    epoch_loss_sum += loss_sum
                    epoch_samples += total_samples

                epoch_loss_history.append(epoch_loss_sum / epoch_samples)
                print(f"Epoch {epoch + 1} Loss: {epoch_loss_history[-1]:.4f}")

        return epoch_loss_history, batch_loss_history