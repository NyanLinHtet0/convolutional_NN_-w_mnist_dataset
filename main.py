import numpy as np
from convolution import *
from Dense import *
from Loss import *

np.random.seed(2)


#input parameters
image_inputsize = (10,10)
#output parameters
output_size = (3,1)

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



# Initialize layers and loss
convolution_layer = Convolution(kernel_shape, num_kernels)
dense_layer = Dense(flatten_shape,output_size)
loss = Loss()

# Forward pass
convu = convolution_layer.forward(np.random.randn(*image_inputsize))
maxpooled = convolution_layer.max_pool(convu, pool_size, stride)
flattened = convolution_layer.flatten(maxpooled)

output_layer = dense_layer.forward(flattened)
# calculate loss
model_loss = loss.softmax_crossentropy(output_layer)

# backpropagation
output_gradient = loss.backward()
dense_gradient = dense_layer.backward(output_gradient)
convolution_gradient = convolution_layer.backward(dense_gradient)



# for kernel in convolution_gradient:
# for row in convolution_gradient:
#     for cell in row:
#         print(f'{cell:.2f}', end=' ')
#     print()
# print()

