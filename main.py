import numpy as np
from Convolution import *
from Dense import *
from Loss import *


convolution_layer = Convolution((2,2),3)
convu = convolution_layer.forward(np.random.randn(10,10))
relu = np.maximum(0, convu)
maxpooled = convolution_layer.max_pool(relu)
flattened = convolution_layer.flatten(maxpooled)


dense_layer = Dense(flattened.shape,(3,1))
loss = Loss()
output_layer = dense_layer.forward(flattened)

model_loss = loss.softmax_crossentropy(output_layer)

output_gradient = loss.backward()
dense_gradient = dense_layer.backward(output_gradient)


print(dense_gradient.shape)

# for row in dense_gradient:
#     for cell in row:
#         print(f'{cell:.2f}', end=' ')
#     print()

# for convo in dense_gradient:
#     for row in convo:
#         for cell in row:
#             print(f'{cell:.2f}', end=' ')
#         print()

# for flattened_value in flattened:
#     print(f'{flattened_value[0]:.2f}', end=' ')