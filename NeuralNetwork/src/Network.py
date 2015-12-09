'''
Created on Dec 7, 2015

@author: tai
'''
import theano
import Layers
import theano.tensor as T
import numpy as np
import Algorithms

class Network(object):
    def __init__(self, training_data, validation_data, ls_layers, optimize_method=True):
        """
        training_data: a list contains 2 objects that are training samples and their labels
        validation_data: a list contains 2 objects that are validation samples and their labels
        ls_layers: a list contains layers in the network
        """
        self.Xtrain, self.Ytrain = training_data
        self.Xvalidate, self.Yvalidate = validation_data
        self.n_train_data = self.Xtrain.shape[0]
        self.layers = ls_layers
         
    def prepare(self, eta):
        x = T.matrix(name='x')
        y = T.matrix(name='y', dtype='int32')
        
        ## Forward propagation
        a = x.T
        for layer in self.layers:
            a = layer.forward_propagation(a)
            
        ## Define functions such as cost function, predict Function
        predict = T.argmax(a, axis = 1, keepdims = True)
        error = T.mean(T.neq(predict, y))

        #theano.function([x, y], cost, updates=opt_alg.Update(self.layers, grad_Ws, grad_Bs), allow_input_downcast=True)    
        self.train_model = Algorithms.NesterovMomentum(0.1, 0.5).Update(self.layers)
        self.predict = theano.function([x], predict)
        self.error = theano.function([x, y], error, allow_input_downcast=True)
        
    def Train(self, max_epochs, mnb_size, eta):
        self.prepare(eta)
        training_data_id = range(self.n_train_data)
        for epoch in xrange(max_epochs):
            np.random.shuffle(training_data_id)
            for start_idx in xrange(0, self.n_train_data, mnb_size):
                Xtrain_batch = self.Xtrain[training_data_id[start_idx:start_idx+mnb_size],:]
                Ytrain_batch = self.Ytrain[training_data_id[start_idx:start_idx+mnb_size],:]
                self.train_model(Xtrain_batch, Ytrain_batch)
            train_err = self.error(self.Xvalidate, self.Yvalidate)
            print('epoch %i, validation error %f' %(epoch, train_err))
                    