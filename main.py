import numpy as np
from convolution import *
from Dense import *
from Loss import *
from Data_pipeline import *
import matplotlib.pyplot as plt
# from Mnist_data import *
np.random.seed(2)



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
flatten_shape = compute_flatten_shape(
                    image_inputsize,
                    kernel_shape,
                    num_kernels,
                    pool_size,
                    stride
                )

dp = DataPipeline(root_dir="Data") 
x_train, y_train = dp.load_split("train", target_size=(10,10))
indices = np.random.permutation(x_train.shape[0])
# Initialize layers and loss
convolution_layer = Convolution(kernel_shape, num_kernels)
dense_layer = Dense(flatten_shape,output_size)
loss = Loss()


loss_history = []
training_loop = 10

for epoch in range(training_loop):
    print(f"Epoch {epoch+1}")
    epoch_loss_sum = 0.0
    for i in indices:
        # input data and label
        label = y_train[i]
        input_data = x_train[i]

        # Forward pass
        convu = convolution_layer.forward(input_data)
        maxpooled = convolution_layer.max_pool(convu, pool_size, stride)
        flattened = convolution_layer.flatten(maxpooled)
        output_layer = dense_layer.forward(flattened)


        # calculate loss
        model_loss = loss.softmax_crossentropy(output_layer, label)
        epoch_loss_sum += float(model_loss)

        # backpropagation
        output_gradient = loss.backward()
        dense_gradient = dense_layer.backward(output_gradient)
        convolution_gradient = convolution_layer.backward(dense_gradient)

    epoch_loss_avg = epoch_loss_sum / x_train.shape[0]
    loss_history.append(epoch_loss_avg)
    print("avg loss:", epoch_loss_avg)


# plot loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Average loss")
plt.title("Training Loss")
plt.show()



# for kernel in convolution_gradient:
# for row in convolution_gradient:
#     for cell in row:
#         print(f'{cell:.2f}', end=' ')
#     print()
# print()

