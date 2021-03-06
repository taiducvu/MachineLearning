'''
Created on Dec 7, 2015

@author: tai
'''
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax

rng = np.random.RandomState()

def ReLU(x):
    return T.maximum(0.0, x)

class Layer(object):
    def __init__(self, w_shape, last_flag = False):
        self.weights = theano.shared(rng.randn(w_shape[0],w_shape[1])/np.sqrt(w_shape[1]), name='weights',
                                     allow_downcast=True)
        
        self.biases = theano.shared(rng.randn(w_shape[0]), name='biases', allow_downcast=True)
        self.last_flag = last_flag
        
    def forward_propagation(self, inpt):
        """
        inpt: nx284
        """
        z = T.dot(self.weights, inpt) + self.biases.dimshuffle(0,'x')
        if self.last_flag == False:
            active = ReLU(z)
        else:
            active = softmax(z.T)
            
        return active
    
    def forward_propagation_NAG(self, inpt, velocity, alph):
        z = T.dot(self.weights + alph*velocity, inpt) + self.biases.dimshuffle(0,'x')
        if self.last_flag == False:
            active = ReLU(z)
        else:
            active = softmax(z.T)            
        return active
        