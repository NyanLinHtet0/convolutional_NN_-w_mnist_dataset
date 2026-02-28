import numpy as np
from Convolution import *
from Dense import *
from Loss import *
from cnn import CNN
from Cnn_worker import *
from Cnn_multi_core import CNNMultiCore
import matplotlib.pyplot as plt
from Cnn_worker import CNN_worker  # your file
import multiprocessing as mp
# from Mnist_data import *


import numpy as np
import matplotlib.pyplot as plt

from cnn import CNN



def main():
    # ---------- Reproducibility (optional) ----------
    np.random.seed(2)

    # ---------- Model hyperparams ----------
    image_inputsize = (10, 10)
    num_kernels = 5
    kernel_shape = (3, 3)
    pool_size = (2, 2)
    stride = (2, 2)
    #Output classes
    output_size = (4, 1)

    # ---------- Load data ----------
    data = np.load('train_10x10_dataset.npz')
    x_train = data['images']
    y_train = data['labels']
    # print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")

    # ---------- Train ----------
    cnn_multicore = CNNMultiCore(
        image_inputsize=image_inputsize,
        output_size=output_size,
        kernel_shape=kernel_shape,
        num_kernels=num_kernels,
        pool_size=pool_size,
        stride=stride
    )



    #training parameters
    epochs = 10
    learning_rate = 0.01


    loss_history = cnn_multicore.train_batches(
        x_train=x_train,
        y_train=y_train,
        epochs=epochs,
        mini_batch_size=64,
        learning_rate=learning_rate
    )

    conv_k, conv_b, dense_w, dense_b = cnn_multicore.get_parameters()
    np.savez(
    f'trained_parameters_{num_kernels}xkernels_input{image_inputsize}.npz',
    conv_kernels=conv_k,
    conv_biases=conv_b,
    dense_weights=dense_w,
    dense_biases=dense_b
    )

    loaded_param = np.load(f'trained_parameters_{num_kernels}xkernels_input{image_inputsize}.npz')
    conv_k = loaded_param['conv_kernels']
    conv_b = loaded_param['conv_biases']
    dense_w = loaded_param['dense_weights']
    dense_b = loaded_param['dense_biases']

    #load parameters
    cnn_multicore.set_parameters(
        conv_kernels=conv_k,
        conv_biases=conv_b,
        dense_weights=dense_w,
        dense_biases=dense_b
    )

    test_sample = np.load('test_10x10_dataset.npz')
    x_test = test_sample['images']
    y_test = test_sample['labels']
    count = 0
    idx_arr = []
    for i in range(len(x_test)):
        pred_label = cnn_multicore.predict(x_test[i])
        if pred_label == y_test[i]:
            count += 1
        else:
            idx_arr.append(i)

    max_show = 10
    wrong_to_show = idx_arr[:max_show]

    plt.figure(figsize=(10, 5))

    for j, i in enumerate(wrong_to_show):
        plt.subplot(2, 5, j + 1)
        plt.imshow(x_test[i], cmap="gray")
        plt.title(f"T:{y_test[i]} P:{cnn_multicore.predict(x_test[i])}")
        plt.axis("off")

    plt.suptitle("First 10 Wrong Predictions")
    plt.tight_layout()
    plt.show()
    accuracy = count / len(x_test)
    print(f"Test Accuracy: {accuracy:.2%}")

    # # ---------- Plot ----------
    plt.plot(loss_history[0])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()


if __name__ == "__main__":
    main()