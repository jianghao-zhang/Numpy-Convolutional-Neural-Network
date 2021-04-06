from functools import reduce
import numpy as np
import math

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, init_params=False):
    
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_h = None
        self.input_w = None
        self.out_h = None
        self.out_w = None

        self.weight_gradient = 0
        self.bias_gradient = 0

        self.init_params = init_params

        self.weight = np.random.randn(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        self.b_rfa = np.random.randn(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)
        self.bias = np.random.randn(self.out_channels, 1)
        self.batch_size = 1


    def forward(self, x):
    
        self.input_map = x

        if not self.init_params:
            self.init_params = True
            weights_scale = math.sqrt(reduce(lambda x, y: x * y, self.input_map.shape) / self.out_channels)

            self.weight = np.random.standard_normal(
                size=(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)) / weights_scale
            self.b_rfa = np.random.standard_normal(
                size=(self.in_channels, self.out_channels, self.kernel_size, self.kernel_size)) / weights_scale
            self.bias = np.random.standard_normal(size=(self.out_channels, 1)) / weights_scale

        self.batch_size, _, self.input_h, self.input_w = x.shape

        self.out_h = (self.input_h-self.kernel_size)//self.stride + 1
        self.out_w = (self.input_w-self.kernel_size)//self.stride + 1

        self.col_images = []

        weight_col = self.weight.reshape(self.out_channels, -1)
        conv_out = np.zeros((self.batch_size, self.out_channels, self.out_h, self.out_w))
        for batch_i in range(self.batch_size):
            image_batch_i = x[batch_i, :]
            image_batch_i_col = im2col(image_batch_i, self.kernel_size, self.stride)
            self.col_images.append(image_batch_i_col)
            conv_out[batch_i] = np.reshape(np.dot(weight_col, np.transpose(image_batch_i_col))+self.bias, (self.out_channels, self.out_h, self.out_w))

        self.col_images = np.array(self.col_images)

        return conv_out


    def calc_gradient(self, error):
        self.error = error
        error_col = self.error.reshape(self.batch_size, self.out_channels, -1)

        for batch_i in range(self.batch_size):
            self.weight_gradient += np.dot(error_col[batch_i], self.col_images[batch_i]).reshape(self.weight.shape)
            
        self.bias_gradient += np.sum(error_col, axis=(0, 2)).reshape(self.bias.shape)

        error_pad = np.pad(self.error, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=0)
        weight_flip = self.weight[:, :, ::-1, ::-1]
        weight_flip = np.swapaxes(weight_flip, 0, 1)
        weight_flip_col = weight_flip.reshape(self.in_channels, -1)

        next_error = np.zeros((self.batch_size, self.in_channels, self.input_h, self.input_w))
        for batch_i in range(self.batch_size):
            error_pad_image_batch_i = error_pad[batch_i, :]
            error_pad_image_batch_i_col = im2col(error_pad_image_batch_i, self.kernel_size, self.stride)
            next_error[batch_i] = np.reshape(np.dot(weight_flip_col, np.transpose(error_pad_image_batch_i_col)), (self.in_channels, self.input_h, self.input_w))

        return next_error


    def calc_gradient_rfa(self, error):
        self.error = error
        error_col = self.error.reshape(self.batch_size, self.out_channels, -1)

        for batch_i in range(self.batch_size):
            self.weight_gradient += np.dot(error_col[batch_i], self.col_images[batch_i]).reshape(self.weight.shape)

        self.bias_gradient += np.sum(error_col, axis=(0, 2)).reshape(self.bias.shape)

        error_pad = np.pad(self.error, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), 'constant', constant_values=0)

        b_flip = self.b_rfa[:, :, ::-1, ::-1]
        b_flip = np.swapaxes(b_flip, 0, 1)
        b_flip_col = b_flip.reshape(self.in_channels, -1)

        next_error = np.zeros((self.batch_size, self.in_channels, self.input_h, self.input_w))
        for batch_i in range(self.batch_size):
            error_pad_image_batch_i = error_pad[batch_i, :]
            error_pad_image_batch_i_col = im2col(error_pad_image_batch_i, self.kernel_size, self.stride)
            next_error[batch_i] = np.reshape(np.dot(b_flip_col, np.transpose(error_pad_image_batch_i_col)), (self.in_channels, self.input_h, self.input_w))

        return next_error


    def backward(self, lr=0.01):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0

    def backward_rfa(self, lr=0.001):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0

    def backward_dfa(self, lr=0.0001):
        self.weight -= lr*self.weight_gradient
        self.bias -= lr*self.bias_gradient

        self.weight_gradient = 0
        self.bias_gradient = 0

    def save_b_rfa(self, name):
        np.save(name, self.b_rfa)

    def save_weight(self, name):
        np.save(name, self.weight)

    def save_bias(self, name):
        np.save(name, self.bias)

    def load_weight(self, b_rfa, weight, bias):
        self.b_rfa = b_rfa
        self.weight = weight
        self.bias = bias


def im2col(img, kernel_size, stride=1):
    img_channel, img_h, img_w = img.shape
    img_cols = None
    for channel_i in range(img_channel):
        img_channel_i = img[channel_i, :]
        img_channel_i_cols = []
        for h_i in range(0, img_h-kernel_size+1, stride):
            for w_i in range(0, img_w-kernel_size+1, stride):
                img_channel_i_patch = img_channel_i[h_i:h_i+kernel_size, w_i:w_i+kernel_size]
                img_channel_i_patch_row = img_channel_i_patch.reshape([-1])
                img_channel_i_cols.append(img_channel_i_patch_row)
                assert img_channel_i_patch_row.shape ==  (kernel_size*kernel_size, )
                
        img_channel_i_cols = np.array(img_channel_i_cols)

        if img_cols is None:
            img_cols = img_channel_i_cols
        else:
            img_cols = np.hstack((img_cols, img_channel_i_cols))

    return img_cols