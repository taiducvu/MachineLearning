'''
Created on Oct 9, 2015

@author: taivu
'''

import theano
import Layers
import Network
import numpy as np
import mnist_loader as ml

if __name__ == '__main__':
    
    training_data, validation_data, test_data = ml.load_data_unify()
    Xtrain, Ytrain = training_data
    
    nb_sample = Xtrain.shape[0]
    Xtrain = Xtrain.reshape(nb_sample, 1, 28, 28)
    
    convLayer = Layers.ConvolLayer((3, 1, 5, 5), 1)
    poolLayer = Layers.PoolLayer((2,2), 2)
    fcLayer = Layers.FCLayer((50, 432), False)
    outLayer = Layers.FCLayer((10, 50), True)
    
    lsLayer = []
    lsLayer.append(convLayer)
    lsLayer.append(poolLayer)
    lsLayer.append(fcLayer)
    #lsLayer.append(outLayer)
    
    cnn = Network.Network(Xtrain, Ytrain, lsLayer, 10, 0.1)
    cnn.cost()
    cnn.SGD(1)