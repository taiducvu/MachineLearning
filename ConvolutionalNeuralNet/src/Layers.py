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
from theano.tensor.nnet import softmax
from theano.tensor.shared_randomstreams import RandomStreams

rng = np.random.RandomState(23455)


def ActiveFunction(z):
    return T.maximum(0.0, z)

class ConvolLayer(object):
    
    def __init__(self, w_shape, stride):
        """
            inpt: input of layer (mini_batch x depth x nb_row x nb_col)
            w_shape: a tuple consists 4 elements: number of filters x depth of filter x filter_row x filter_col
            stride:
            weight: a matrix whose size is width of filter x height of filter x number of filters
        """
        #self.input = inpt
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
        """
        
        #inpt = T.tensor4(name='input')
        #w = T.tensor4(name='w') # weight
        #b = T.vector(name='biases', dtype = 'float32') # bias
        self.input = inpt
               
        z = conv.conv2d(inpt, self.weights, subsample=(self.stride, self.stride))
        self.active = ActiveFunction(z + self.biases.dimshuffle('x', 0, 'x', 'x')) # Active of the current layer
        #f = function([inpt, w, b], a)        
        #self.input = inpt
        #self.active =  f(self.input, self.weights.get_value(), self.biases.get_value())
        return self.active

class PoolLayer(object):
    
    def __init__(self, window_shape, stride):
        """
        """
        #self.input = inpt
        self.window_shape = window_shape
        self.stride = stride
        self.name = "PoolLayer"
    
    def forward_propagation(self, inpt):
        """
        """
        
        #inpt = T.tensor4(name='input')
        self.active = downsample.max_pool_2d(inpt, ds=self.window_shape, ignore_border=False,
                                                st=(self.stride, self.stride))
        #f = function([inpt], pooled_active)
        
        #self.active = f(self.input)
        self.input = inpt
        return self.active

class FCLayer(object):
    def __init__(self, w_shape, flag_last):
        """
        """
        #self.size_before_reshape = inpt.shape
        #self.inpt = inpt.reshape(self.size_before_reshape[0], -1)
        self.w_shape = w_shape
        self.weights = theano.shared(rng.uniform(size= self.w_shape), name='weights', allow_downcast=True)
        self.biases = theano.shared(rng.uniform(size=self.w_shape[0]), name='biases', allow_downcast=True)
        self.flag_last = flag_last
        self.name = "FCLayer"
        
    def forward_propagation(self, inpt):
        """
        """
        self.size_before_reshape = inpt.shape
        self.input = inpt.reshape((self.size_before_reshape[0], -1))
        ##self.weights = theano.shared(rng.uniform(size=(self.nb_neurons, self.input.shape[1])), name='weights', allow_downcast=True)
        
        #inpt = T.matrix(name='input')
        #w = T.matrix(name='weight')
        #b = T.vector(name='bias')
        z = T.dot(self.input, self.weights.T) + self.biases.dimshuffle('x',0)
        if(self.flag_last == False):            
            self.active = ActiveFunction(z)
        else:
            self.active = softmax(z)
            
        #f = function([inpt, w, b], a)
        
        #self.active = f(self.inpt, self.weights, self.biases)
        return self.active