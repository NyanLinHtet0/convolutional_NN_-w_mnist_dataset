import numpy as np
from convolution import *
from Dense import *
from Loss import *
from Data_pipeline import *
from cnn import CNN
import matplotlib.pyplot as plt
# from Mnist_data import *


#input parameters
image_inputsize = (10,10)
#output parameters
output_size = (2,1)

#kernel parameters
kernel_shape = (3,3)
num_kernels = 2

#max pooling parameters
pool_size = (2,2)
stride = (2,2)





dp = DataPipeline(root_dir="Data") 

# x_train, y_train = dp.load_and_npz_save("train", target_size=image_inputsize)

data = np.load('train_10x10_dataset.npz')
x_train = data['images']
y_train = data['labels']

# Initialize CNN model
np.random.seed(2)
cnn_model = CNN(image_inputsize, kernel_shape, num_kernels, pool_size, stride, output_size)
loss_history = cnn_model.train_SGD(x_train, y_train, epochs=10)



# plot loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Average loss")
plt.title("Training Loss")
plt.show()

