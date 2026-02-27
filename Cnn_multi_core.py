from cnn import CNN
import os
import numpy as np
import multiprocessing as mp
from Cnn_worker import CNN_worker
from Cnn_worker import worker_init
from Cnn_worker import _run_worker_job




class CNNMultiCore(CNN):
    def train_batches(self,
        x_train,
        y_train,
        epochs= 10,
        mini_batch_size= 4,
        learning_rate= 0.01,
        ):
        num_workers = max(1, (os.cpu_count() or 1))
        batch_loss_history = []
        print(f"Using {num_workers} worker processes for training.")

        
        image_inputsize = (x_train.shape[1], x_train.shape[2])
        ctx = mp.get_context("spawn")
        


        batch_size = mini_batch_size * num_workers # ensure batch size is a multiple of num_workers for even splitting
        
        epoch_loss_history = []
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            epoch_loss_sum = 0
            epoch_samples = 0 
            # shuffle indices for each epoch
            indices = np.random.permutation(np.arange(x_train.shape[0]))  

            with ctx.Pool(
                processes=num_workers,
                initializer=worker_init,
                initargs=(x_train, y_train),
            ) as pool:
            
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
                    for worker_id in range(num_workers):
                        worker_start = worker_id * mini_batch_size
                        worker_end = min(worker_start + mini_batch_size, batch_indices.shape[0])
                        index_arr = batch_indices[worker_start:worker_end]
                        if worker_start >= worker_end:
                            break

                        worker_args.append((
                            image_inputsize,
                            self.convolution_layer.kernel_shape,
                            self.convolution_layer.depth,
                            self.pool_size,
                            self.stride,
                            self.output_size,
                            index_arr,
                            conv_k, conv_b, dense_w, dense_b,
                            learning_rate
                        ))
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
                    
                    # update main model parameters
                    self.update_parameters_with_grad((conv_param, dense_param), learning_rate=learning_rate)
                    # average loss for the batch
                    if total_samples == 0:
                        continue
                    batch_loss_avg = loss_sum / total_samples
                    batch_loss_history.append(batch_loss_avg)

                    # accumulate epoch stats
                    epoch_loss_sum += loss_sum
                    epoch_samples += total_samples
            epoch_loss_history.append(epoch_loss_sum / epoch_samples)
            print(f"Epoch {epoch+1} Loss: {epoch_loss_history[-1]:.4f}")

        return epoch_loss_history, batch_loss_history