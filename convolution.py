import numpy as np
def compute_flatten_shape(image_inputsize,
                          kernel_shape,
                          num_kernels,
                          pool_size,
                          stride):

    # convolution output
    conv_height = (image_inputsize[0] + 2*1 - kernel_shape[0]) // 1 + 1
    conv_width  = (image_inputsize[1] + 2*1 - kernel_shape[1]) // 1 + 1

    # maxpool output
    pool_height = (conv_height - pool_size[0]) // stride[0] + 1
    pool_width  = (conv_width  - pool_size[1]) // stride[1] + 1

    # flattened dimensions
    f_height = num_kernels * pool_height * pool_width
    f_width = 1

    return (f_height, f_width)

class Convolution():
    def __init__(self, kernel_shape = (3,3), num_kernels=3):
        self.kernel_shape = kernel_shape
        self.depth = num_kernels
        self.kernels = np.random.randn(num_kernels,*self.kernel_shape)
        self.biases = np.random.randn(num_kernels,1,1)
        self.dl_dk = np.zeros_like(self.kernels)
        self.dl_db = np.zeros_like(self.biases)
        
    def forward(self, input_data, pad = 1):
        input_height, input_width = input_data.shape

        self.pad = pad
        kh, kw = self.kernel_shape

        self.padded = np.pad(input_data, self.pad, mode='constant', constant_values=0)
        output_height = input_height + 2*self.pad - kh + 1
        output_width  = input_width  + 2*self.pad - kw + 1
        self.conv_out_shape = (self.depth, output_height, output_width)
        output = np.zeros(self.conv_out_shape)

        for k in range(self.conv_out_shape[0]):
            for i in range(output_height):
                for j in range(output_width):
                    region = self.padded[i:i+kh, j:j+kw]
                    output[k, i, j] = np.sum(region * self.kernels[k]) + self.biases[k, 0, 0]
        # save pre-ReLU values
        self.pre_relu = output.copy()

        # mask from pre-ReLU (values > 0)
        self.relu_mask = (self.pre_relu > 0).astype(int)

        # apply ReLU
        output = np.maximum(0, output)

        return output

    def max_pool(self, input_data, pool_size=(2,2), stride=(2,2)):
        # input matrix shape (num_kernels, height, width)
        num_kernels, input_height, input_width = input_data.shape

        ph, pw = pool_size
        sh, sw = stride

        # save for backward
        self.pool_size = pool_size
        self.pool_stride = stride

        # output matrix shape (num_kernels, output_height, output_width)
        output_height = (input_height - ph) // sh + 1
        output_width  = (input_width  - pw) // sw + 1

        # create output matrix (POOL output shape)
        self.pool_out_shape = (num_kernels, output_height, output_width)
        output = np.zeros(self.pool_out_shape)

        # mask matrix for backward pass
        self.max_mask = np.zeros_like(input_data)
        # perform max pooling
        for k in range(num_kernels):
            for i in range(output_height):
                for j in range(output_width):
                    r0 = i*sh
                    c0 = j*sw
                    region = input_data[k, r0:r0+ph, c0:c0+pw]
                    output[k, i, j] = np.max(region)
                    # find index inside region
                    max_idx = np.unravel_index(np.argmax(region), region.shape)

                    # convert local index → global index
                    global_i = r0 + max_idx[0]
                    global_j = c0 + max_idx[1]
                    # store mask
                    self.max_mask[k, global_i, global_j] = 1
        return output
    
    def flatten(self, input_data):
        output = input_data.flatten().reshape(-1, 1)
        return output
    
    def backward(self, output_gradient, learning_rate=0.01):    
        # unflatten the output gradient into POOL output shape
        output_gradient = output_gradient.reshape(self.pool_out_shape)

        ph, pw = self.pool_size
        sh, sw = self.pool_stride

        # gradient to pass down
        dl_drelu = np.zeros_like(self.max_mask)

        # fill dl_drelu using the max_mask and output_gradient
        for k in range(self.max_mask.shape[0]):
            for i in range(output_gradient.shape[1]):
                for j in range(output_gradient.shape[2]):
                    r0 = i*sh
                    c0 = j*sw
                    region = self.max_mask[k, r0:r0+ph, c0:c0+pw]
                    dl_drelu[k, r0:r0+ph, c0:c0+pw] += region * output_gradient[k, i, j]

        # calculate dl_dconv by applying ReLU mask to dl_drelu
        dl_dconv = dl_drelu * self.relu_mask

        # dl_db and dl_dk calculation
        for k in range(self.dl_dk.shape[0]):
            # update bias gradient
            self.dl_db[k] += np.sum(dl_dconv[k])
            #  update kernel gradient
            for a in range(self.dl_dk.shape[1]):
                for b in range(self.dl_dk.shape[2]):
                    region_x = self.padded[a:a+dl_dconv.shape[1], b:b+dl_dconv.shape[2]]
                    region_dl_relu = dl_dconv[k]
                    self.dl_dk[k,a,b] += np.sum(region_x * region_dl_relu)

        # Rotate kernels 180 degrees for convolution backward
        rotated_kernels = np.rot90(self.kernels, 2, axes=(1,2))
        # initialize input gradient
        input_gradient = np.zeros_like(self.padded)
        # calculate input gradient using dl_dconv and rotated kernels
        kh, kw = self.kernels.shape[1], self.kernels.shape[2]
        for k in range(self.depth):
            for i in range(dl_dconv.shape[1]):   
                for j in range(dl_dconv.shape[2]):
                    input_gradient[i:i+kh, j:j+kw] += rotated_kernels[k] * dl_dconv[k, i, j]
                    
        pad = self.pad
        return input_gradient[pad:-pad, pad:-pad]
    
    def update_parameters(self, learning_rate=0.01):
        self.kernels -= learning_rate * self.dl_dk
        self.biases  -= learning_rate * self.dl_db
        self.dl_dk = np.zeros_like(self.kernels)
        self.dl_db = np.zeros_like(self.biases)

    #--------------------------MODEL LOADING AND SAVING METHODS -------------------------------
    def return_kernels(self):
        return self.kernels
    
    def return_biases(self):
        return self.biases
    
    def load_kernels(self, kernels):
        self.kernels = kernels

    def load_biases(self, biases):
        self.biases = biases