from cnn import CNN
import os
import numpy as np
import multiprocessing as mp

# --- Worker Function ---
def _worker_compute_grads(args):
    """
    Each worker:
      - builds its own CNN
      - loads master parameters
      - runs train_mini_batch on its shard
      - returns (num_samples, loss_avg, dense_grads_avg, conv_grads_avg)
    """
    (x_shard, y_shard,
     image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size,
     conv_kernels, conv_biases, dense_weights, dense_biases) = args

    local = CNN(
        image_inputsize=image_inputsize,
        kernel_shape=kernel_shape,
        num_kernels=num_kernels,
        pool_size=pool_size,
        stride=stride,
        output_size=output_size
    )
    local.set_parameters(conv_kernels, conv_biases, dense_weights, dense_biases)  # :contentReference[oaicite:2]{index=2}

    n = int(len(x_shard))
    loss_avg, dense_grad_avg, conv_grad_avg = local.train_mini_batch(
        x_train=x_shard,
        y_train=y_shard,
        batch_size=n,
        multi_core=True  # important: do NOT update inside worker :contentReference[oaicite:3]{index=3}
    )
    return (n, loss_avg, dense_grad_avg, conv_grad_avg)

class CNNMultiCore(CNN):
    def train_batches(self,
        x_train,
        y_train,
        epochs=10,
        batch_size=32,
        learning_rate=0.01,
        num_workers=None
        ):
        num_workers = num_workers or (os.cpu_count() or 1)
        loss_history = []
        
        image_inputsize = x_train.shape[1], x_train.shape[2]
        kernel_shape = self.convolution_layer.kernel_shape
        for epoch in range(epochs):
            indices = np.random.permutation(x_train.shape[0])
            epoch_loss_sum = 0.0
            seen = 0


        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            #create partitions for workers
            shards = np.array_split(batch_idx, num_workers)
            # filter out empty shards (can happen if batch_size < num_workers)
            shards = [s for s in shards if len(s) > 0]    
            # save parameters to share with workers
            conv_kernels = self.convolution_layer.return_kernels()
            conv_biases  = self.convolution_layer.return_biases()
            dense_weights = self.dense_layer.return_weights()
            dense_biases  = self.dense_layer.return_biases()   

        # prepare worker args and load data
            worker_args = []
            for s in shards:
                x_shard = x_train[s]
                y_shard = y_train[s]
                worker_args.append((
                    x_shard, y_shard,
                    image_inputsize,
                    self.convolution_layer.kernel_shape,
                    self.convolution_layer.depth,
                    self.pool_size,
                    self.stride,
                    self.output_size,
                    conv_kernels, conv_biases, dense_weights, dense_biases
                ))

        # run workers in parallel (real multi-core)
        with mp.Pool(processes=len(shards)) as pool:
            results = pool.map(_worker_compute_grads, worker_args)

        # --- merge grads (weighted by shard size) ---
        total_n = sum(r[0] for r in results)