import numpy as np
from Convolution import *
from Dense import *


Convolution_instance = Convolution()
convu = Convolution_instance.forward(np.random.randn(10,10))
relu = np.maximum(0, convu)
maxpooled = Convolution_instance.max_pool(relu)
flattened = Convolution_instance.flatten(maxpooled)


dense_instance = Dense()

print(dense_instance.weights.shape)
print(flattened.shape)
print(dense_instance.forward(flattened).shape)


# for convo in maxpooled:
#     for row in convo:
#         for cell in row:
#             print(f'{cell:.2f}', end=' ')
#         print()

# for flattened_value in flattened:
#     print(f'{flattened_value[0]:.2f}', end=' ')