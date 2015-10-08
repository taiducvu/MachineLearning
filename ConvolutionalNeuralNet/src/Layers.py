'''
Created on Sep 28, 2015

@author: taivu
'''

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano import function


rng = np.random.RandomState(345)

def ActiveFunction(z):
    return T.maximum(0.0, z)

class ConvolLayer(object):
    
    def __init__(self, inpt, w_shape, stride):
        """
            inpt: input of layer (mini_batch x depth x nb_row x nb_col)
            w_shape: a tuple consists 4 elements: number of filters x depth of filter x filter_row x filter_col
            stride:
            weight: a matrix whose size is width of filter x height of filter x number of filters
        """
        self.input = inpt
        self.w_shape = w_shape
        self.stride = stride
        self.weights = theano.shared(rng.uniform(size = w_shape), name='weights', allow_downcast=True)
        self.biases = theano.shared(rng.uniform(size=w_shape[0]), name='biases', allow_downcast=True)
        
    def forward_propagation(self):
        """
        """
        inpt = T.tensor4(name='input')
        w = T.tensor4(name='w', dtype= inpt.dtype) # weight
        b = T.vector(name='biases', dtype = 'float32') # bias
                
        z = conv.conv2d(inpt, w, subsample=(self.stride, self.stride))
        a = ActiveFunction(z + b.dimshuffle('x', 0, 'x', 'x')) # Active of the current layer
        f = function([inpt, w, b], a)
        
        self.active =  f(self.input, self.weights.get_value(), self.biases.get_value())

class PoolLayer(object):
    
    def __init__(self, inpt, window_shape, stride):
        """
        """
        self.input = inpt
        self.window_shape = window_shape
        self.stride = stride
    
    def forward_propagation(self):
        """
        """
        inpt = T.tensor4(name='input')
        pooled_active = downsample.max_pool_2d(inpt, ds=self.window_shape, ignore_border=False,
                                                st=(self.stride, self.stride))
        f = function([inpt], pooled_active)
        
        self.active = f(self.input)

class FCLayer(object):
    def __init__(self, inpt, nb_neurons):
        """
        """
        self.size_before_reshape = inpt.shape
        self.inpt = inpt.reshape(self.size_before_reshape[0], -1)
        self.nb_neurons = nb_neurons
        self.weights = theano.shared(rng.uniform(size=(nb_neurons, inpt.shape[1])), name='weights', allow_downcast=True)
        self.biases = theano.shared(rng.uniform(size=(nb_neurons)), name='biases', allow_downcast=True)
        
    def forward_propagation(self):
        """
        """        
        inpt = T.matrix(name='input')
        w = T.matrix(name='weight')
        b = T.vector(name='bias')
        z = T.dot(inpt, w.T) + b('x',0)
        a = ActiveFunction(z)
        f = function([inpt, w, b], a)
        
        self.active = f(self.inpt, self.weights, self.biases)