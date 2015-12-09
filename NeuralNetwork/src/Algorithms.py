'''
Created on Dec 8, 2015

@author: tai
'''

import Layers
import theano
import theano.tensor as T
import numpy as np

rng = np.random.RandomState()

class SGD(object):
    def __init__(self, learning_rate):
        self.eta = learning_rate
        
    def Update(self, ls_layers, grad_Ws, grad_Bs):
        updates = []
        for layer, grad_w, grad_b in zip(ls_layers, grad_Ws, grad_Bs):
            updates.append((layer.weights, layer.weights - (self.eta * grad_w)))
            updates.append((layer.biases, layer.biases - (self.eta * grad_b)))
        
        return updates

class Momentum(object):
    def __init__(self, learning_rate, alph):
        self.eta = learning_rate
        self.velocity = []
        self.alpha = alph
        
    def Update(self, ls_layers, grad_Ws, grad_Bs):
        updates = []        
        for layer in ls_layers:        
            self.velocity.append(theano.shared(rng.randn(layer.weights.get_value().shape[0],layer.weights.get_value().shape[1]),
                                          allow_downcast=True))
            
        for layer, grad_w, grad_b, v in zip(ls_layers, grad_Ws, grad_Bs, self.velocity):
            updates.append((v, self.alpha*v + self.eta*grad_w))
            updates.append((layer.weights, layer.weights + v))
            updates.append((layer.biases, layer.biases - (self.eta * grad_b)))
            
        return updates                          