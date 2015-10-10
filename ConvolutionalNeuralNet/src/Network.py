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
    def __init__(self, training_data, validation_data, ls_layers, mnb_size, eta):
        #self.input = inpt
        self.Xtrain, self.Ytrain = training_data
        self.Xvalidate, self.Yvalidate = validation_data
        
        #self.expectValue = expValue
        #self.n_data = self.Xtrain.shape[0]
        self.n_train_data = self.Xtrain.shape[0]
        self.Xtrain = self.Xtrain.reshape(self.n_train_data, 1, 28, 28)
        
        self.n_validate_data = self.Xvalidate.shape[0]
        self.Xvalidate = self.Xvalidate.reshape(self.n_validate_data, 1, 28, 28)
        
        self.layers = ls_layers
        self.mini_batch_size = mnb_size
        self.eta = eta
        
    def cost(self):
        #inpt = T.tensor4(name='input')
        x = T.tensor4(name='x')
        #expectedValues = T.matrix(name='expectedValues', dtype='int32')
        y = T.matrix(name='y', dtype='int32')
        
        a = x
        for layer in self.layers:
            a = layer.forward_propagation(a)
        
        self.cost = -T.mean(T.log(a)[T.arange(y.shape[0]), y])
        self.predict = T.argmax(a, axis=1, keepdims=True)
        self.error = T.mean(T.neq(self.predict, y))
        
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
            #self.eta/self.mini_batch_size
            updates.append((self.layers[l].weights, self.layers[l].weights - (self.eta * gw[i])))
            updates.append((self.layers[l].biases, self.layers[l].biases - (self.eta * gb[i])))
            
        self.train_model = theano.function([x, y], self.cost, updates=updates, allow_input_downcast=True)
        self.predict_model = theano.function([x], self.predict)
        self.validate_model = theano.function([x, y], self.error, allow_input_downcast=True)
        
        
    def SGD(self, epochs):
        training_data_id = range(self.n_train_data)          
        for epoch in xrange(epochs):
            np.random.shuffle(training_data_id)
            for i in xrange(int(math.ceil(float(self.n_train_data)/self.mini_batch_size))):
                mini_batch_id = training_data_id[i*self.mini_batch_size:(i+1)*self.mini_batch_size]
                Xtrain_batch = self.Xtrain[mini_batch_id,:,:,:]
                Ytrain_batch = self.Ytrain[mini_batch_id,:]
                self.train_model(Xtrain_batch, Ytrain_batch)
            err = self.validate_model(self.Xvalidate, self.Yvalidate)
            print('epoch %i, validation error %f' %(epoch, err))
            
            
            