from cnn import CNN
import os
import numpy as np
import multiprocessing as mp
from Cnn_worker import CNN_worker

def _run_worker_job(args):
    (worker, x_train, y_train, idx_arr, lr, conv_k, conv_b, dense_w, dense_b) = args
    worker.set_parameters(conv_k, conv_b, dense_w, dense_b)
    return worker.train_mini_batch(x_train, y_train, idx_arr, learning_rate=lr) + (idx_arr.shape[0],)

class CNNMultiCore(CNN):
    def train_batches(self,
        x_train,
        y_train,
        epochs= 10,
        mini_batch_size= 4,
        learning_rate= 0.01,
        ):
        num_workers = (os.cpu_count() or 1)
        loss_history = []
        print(f"Using {num_workers} worker processes for training.")

        #initialize workers with same parameters as main model
        image_inputsize = (x_train.shape[1], x_train.shape[2])
        workers = []
        for _ in range(num_workers):
            worker = CNN_worker(
                image_inputsize=image_inputsize,
                kernel_shape=self.convolution_layer.kernel_shape,
                num_kernels=self.convolution_layer.depth,
                pool_size=self.pool_size,
                stride=self.stride,
                output_size=self.output_size
            )
            workers.append(worker)

        batch_size = mini_batch_size * num_workers # ensure batch size is a multiple of num_workers for even splitting
        indices = np.random.permutation(np.arange(x_train.shape[0]))  # shuffle indices for each epoch
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            if batch_indices.size == 0:
                continue

            # master parameters to sync to workers
            conv_k = self.convolution_layer.kernels
            conv_b = self.convolution_layer.biases
            dense_w = self.dense_layer.weights
            dense_b = self.dense_layer.biases

            #Generate args for each worker
            worker_args = []
            for worker in range(num_workers):
                start = worker * mini_batch_size
                end = min(start + mini_batch_size, batch_indices.shape[0])
                if start >= end:
                    break

                index_arr = batch_indices[start:end]
                worker_args.append((workers[worker], x_train, y_train, index_arr, learning_rate,
                                    conv_k, conv_b, dense_w, dense_b))

            # spawn processes for each worker
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=len(worker_args)) as pool:
                results = pool.map(_run_worker_job, worker_args)

            # aggregate gradients and update main model parameters
            total_samples = 0
            conv_k_grad_sum = np.zeros_like(conv_k)
            conv_b_grad_sum = np.zeros_like(conv_b)
            dense_w_grad_sum = np.zeros_like(dense_w)
            dense_b_grad_sum = np.zeros_like(dense_b)
            loss_sum = 0
            for loss, conv_param, dense_param, num_samples in results:
                loss_sum += loss
                conv_k_grad_sum += conv_param[0]
                conv_b_grad_sum +=  conv_param[1]
                dense_w_grad_sum += dense_param[0]
                dense_b_grad_sum += dense_param[1]
                total_samples += num_samples
            # average gradients
            conv_k_grad_avg = conv_k_grad_sum / total_samples
            conv_b_grad_avg = conv_b_grad_sum / total_samples
            dense_w_grad_avg = dense_w_grad_sum / total_samples
            dense_b_grad_avg = dense_b_grad_sum / total_samples
            
            conv_param = (conv_k_grad_avg, conv_b_grad_avg)
            dense_param = (dense_w_grad_avg, dense_b_grad_avg)
            
            self.set_parameters(self, conv_kernels, conv_biases, dense_weights, dense_biases)
            # update main model parameters
            
            self.update_parameters(learning_rate=learning_rate)
            
        return