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
    def __init__(self, training_data, validation_data, ls_layers):
        """
        training_data: a list contains 2 objects that are training samples and their labels
        validation_data: a list contains 2 objects that are validation samples and their labels
        ls_layers: a list contains layers in the network
        """
        
        self.Xtrain, self.Ytrain = training_data
        self.Xvalidate, self.Yvalidate = validation_data
        
        self.n_train_data = self.Xtrain.shape[0] # number of samples in train set
        
        """
         Reshape training data into a matrix whose size is (# samples x 1 x 28 x 28)
        to take advantage of available functions in Theano.
        """
        self.Xtrain = self.Xtrain.reshape(self.n_train_data, 1, 28, 28)
        
        self.n_validate_data = self.Xvalidate.shape[0] # number of samples in train set
        
        """ 
         Reshape validation data into a matrix whose size is (# samples x 1 x 28 x 28)
        to take advantage of available functions in Theano.
        """
        self.Xvalidate = self.Xvalidate.reshape(self.n_validate_data, 1, 28, 28)
        
        self.layers = ls_layers
        
    def prepare(self, eta):
        """
        This method is used to prepare functions in the network. Parameters:
            + eta: learning rate
        """
        x = T.tensor4(name='x')
        y = T.matrix(name='y', dtype='int32')
        
        ## Process of forward propagation
        a = x
        for layer in self.layers:
            a = layer.forward_propagation(a)
        
        ## Define functions such as cost function, predict function, ... 
        cost = -T.mean(T.log(a)[T.arange(y.shape[0]), y.T])
        
        predict = T.argmax(a, axis=1, keepdims=True)
        error = T.mean(T.neq(predict, y))
        
        idx = [] # index of layers which are not PoolLayer in ls_layer
        ls_weights = [] # a list contains weights of layers in the network
        ls_biases = [] # a list contains biases of layers in the network
        
        updates = []        
        for i in xrange(len(self.layers)):
            if self.layers[i].name != "PoolLayer":
                ls_weights.append(self.layers[i].weights)
                ls_biases.append(self.layers[i].biases)
                idx.append(i)
        
        ## Compute gradient descent the cost function
        gw = T.grad(cost, ls_weights)
        gb = T.grad(cost, ls_biases)
        
        
        for l, i in zip(idx, xrange(len(idx))):
            updates.append((self.layers[l].weights, self.layers[l].weights - (eta * gw[i])))
            updates.append((self.layers[l].biases, self.layers[l].biases - (eta * gb[i])))
            
        
        self.train_model = theano.function([x, y], cost, updates=updates, allow_input_downcast=True)
        self.predict_model = theano.function([x], predict)
        self.validate_model = theano.function([x, y], error, allow_input_downcast=True)

    def SGD(self, epochs, mnb_size, eta):
        """
        This method is used to do stochastic gradient descent. Parameters:
            + epochs: number of epochs
            + mnb_size: size of mini-batch
            + eta: learning rate
        """
        self.prepare(eta)
        
        training_data_id = range(self.n_train_data)          
        for epoch in xrange(epochs):
            np.random.shuffle(training_data_id)
            for i in xrange(int(math.ceil(float(self.n_train_data)/mnb_size))):
                mini_batch_id = training_data_id[i * mnb_size : (i + 1) * mnb_size]
                Xtrain_batch = self.Xtrain[mini_batch_id,:,:,:]
                Ytrain_batch = self.Ytrain[mini_batch_id,:]
                self.train_model(Xtrain_batch, Ytrain_batch)
                
            err = self.validate_model(self.Xvalidate, self.Yvalidate)
            print('epoch %i, validation error %f' %(epoch, err))