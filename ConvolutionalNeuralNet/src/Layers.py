'''
Created on Sep 28, 2015

@author: taivu
'''

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.tensor.nnet import softmax

rng = np.random.RandomState(23455)


def ActiveFunction(z):
    return T.maximum(0.0, z)

class ConvolLayer(object):
    
    def __init__(self, w_shape, stride):
        """
        w_shape: a tuple is (#filters, depth filter, width of filter, height of filter)
        stride:
        """
        self.name = "ConvolLayer"
        self.w_shape = w_shape
        self.stride = stride
        
        # Initialize the parameters of convol layer
        fan_in = np.prod(w_shape[1:])
        fan_out = (w_shape[0] * np.prod(w_shape[2:]) /np.prod((2,2)))
        W_bound = np.sqrt(6. / (fan_in + fan_out))     
        
        self.weights = theano.shared(rng.uniform(low=-W_bound, high=W_bound, size = w_shape), name='weights', allow_downcast=True)
        self.biases = theano.shared(rng.uniform(size=w_shape[0]), name='biases', allow_downcast=True)
        
    def forward_propagation(self, inpt):
        """
        This method is used to define how to do forward propagation in ConvolLayer. Parameter:
            + inpt: a 4D matrix whose size is (#samples, depth(#channels) of a sample, width of a sample,
            height of a sample presents data of input of a convolutional layer.
        """
        self.input = inpt
               
        z = conv.conv2d(inpt, self.weights, subsample=(self.stride, self.stride))
        self.active = ActiveFunction(z + self.biases.dimshuffle('x', 0, 'x', 'x'))
        return self.active

class PoolLayer(object):
    
    def __init__(self, window_shape, stride):
        """
        window_shape: a tuple is (width of a window, height of a window)
        stride:
        """
        self.window_shape = window_shape
        self.stride = stride
        self.name = "PoolLayer"
    
    def forward_propagation(self, inpt):
        """
        This method is used to define how to do forward propagation in PoolLayer. Parameter:
            + inpt: a 4D matrix whose size is #inputs x depth(#channels) of a input x width of a input,
            height of a input presents data of input of a Pool layer.
        """
        self.active = downsample.max_pool_2d(inpt, ds=self.window_shape, ignore_border=True,
                                                st=(self.stride, self.stride))
        self.input = inpt
        return self.active

class FCLayer(object):
    def __init__(self, w_shape, flag_last):
        """
        """
        self.w_shape = w_shape
        self.weights = theano.shared(rng.randn(self.w_shape[0],self.w_shape[1])/np.sqrt(w_shape[1]), name='weights', allow_downcast=True)
        self.biases = theano.shared(rng.randn(self.w_shape[0]), name='biases', allow_downcast=True)
        self.flag_last = flag_last
        self.name = "FCLayer"
        
    def forward_propagation(self, inpt):
        """
        """
        self.size_before_reshape = inpt.shape
        self.input = inpt.reshape((self.size_before_reshape[0], -1))
        z = T.dot(self.input, self.weights.T) + self.biases.dimshuffle('x',0)
        if(self.flag_last == False):            
            self.active = ActiveFunction(z)
        else:
            self.active = softmax(z)
        return self.active