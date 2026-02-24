import numpy as np
from convolution import *
from Dense import *
from Loss import *
from Data_pipeline import *
from cnn import CNN
import matplotlib.pyplot as plt
# from Mnist_data import *


import numpy as np
import matplotlib.pyplot as plt

from cnn import CNN



def main():
    # ---------- Reproducibility (optional) ----------
    np.random.seed(2)

    # ---------- Load data ----------
    data = np.load('train_10x10_dataset.npz')
    x_train = data['images']
    y_train = data['labels']

    # ---------- Model hyperparams ----------
    image_inputsize = (10, 10)

    kernel_shape = (3, 3)
    num_kernels = 2

    pool_size = (2, 2)
    stride = (2, 2)

    #Output classes
    output_size = (2, 1)

    # ---------- Build model ----------
    cnn = CNN(
        image_inputsize=image_inputsize,
        output_size=output_size,
        kernel_shape=kernel_shape,
        num_kernels=num_kernels,
        pool_size=pool_size,
        stride=stride
    )

    # ---------- Train ----------
    epochs = 10
    learning_rate = 0.01

    loss_history = cnn.train_SGD(
        x_train=x_train,
        y_train=y_train,
        epochs=epochs,
        learning_rate=learning_rate
    )

    # ---------- Plot ----------
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()


if __name__ == "__main__":
    main()







