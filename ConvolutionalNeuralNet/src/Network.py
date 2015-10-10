'''
Created on Sep 30, 2015

@author: taivu
'''
import theano
import Layers
import math
import numpy as np
import theano.tensor as T

class Network(object):
    def __init__(self, inpt, expValue, ls_layers, mnb_size, eta):
        self.input = inpt
        self.expectValue = expValue
        self.n_data = inpt.shape[0]
        self.layers = ls_layers
        self.training_data_id = range(self.n_data)
        self.mini_batch_size = mnb_size
        self.eta = eta
    def cost(self):
        inpt = T.tensor4(name='input')
        expectedValues = T.matrix(name='expectedValues', dtype='int32')
        
        a = inpt
        for layer in self.layers:
            a = layer.forward_propagation(a)
        
        self.cost = -T.mean(T.log(a)[T.arange(expectedValues.shape[0]), expectedValues])
        
        idx = []
        ls_weights = []
        ls_biases = []
        updates = []
        
        for i in xrange(len(self.layers)):
            if self.layers[i].name != "PoolLayer":
                ls_weights.append(self.layers[i].weights)
                ls_biases.append(self.layers[i].biases)
                idx.append(i)     
        
        gw = T.grad(self.cost, ls_weights)
        gb = T.grad(self.cost, ls_biases)
        
        
        for l, i in zip(idx, xrange(len(idx))):
            updates.append((self.layers[l].weights, self.layers[l].weights - (self.eta/self.mini_batch_size) * gw[i]))
            updates.append((self.layers[l].biases, self.layers[l].biases - (self.eta/self.mini_batch_size) * gb[i]))
            
        self.train_model = theano.function([inpt, expectedValues], self.cost, updates=updates, allow_input_downcast=True)
        
    def SGD(self, epochs):
#         idx = []
#         ls_weights = []
#         ls_biases = []
#         updates = []
#         
#         for i in xrange(len(self.layers)):
#             if self.layers[i].name != "PoolLayer":
#                 ls_weights.append(self.layers[i].weights)
#                 ls_biases.append(self.layers[i].biases)
#                 idx.append(i)     
#         
#         gw = T.grad(self.cost, ls_weights)
#         gb = T.grad(self.cost, ls_biases)
#         
#         
#         for l, i in zip(idx, xrange(len(idx))):
#             updates.append(self.layers[l].weights, self.layers[l].weights - self.eta * gw[i])
#             updates.append(self.layers[l].biases, self.layers[l].biases - self.eta * gb[i])
#             
#         train_model = theano.function([inpt, expectedValues], self.cost,
#                                       updates=updates
#                                       )              
        for epoch in xrange(epochs):
            np.random.shuffle(self.training_data_id)
            for i in xrange(int(math.ceil(float(self.n_data)/self.mini_batch_size))):
                mini_batch_id = self.training_data_id[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
                inpt_batch = self.input[mini_batch_id,:,:,:]
                expectValue_batch = self.expectValue[mini_batch_id,:]
                print expectValue_batch.shape
                self.train_model(inpt_batch, expectValue_batch)
                