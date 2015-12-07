'''
Created on Dec 7, 2015

@author: tai
'''

import Layers
import Network
import mnist_loader as ml

if __name__ == '__main__':
    
    training_data, validation_data, test_data = ml.load_data_unify() # This source code is written by Bao Chiem

    fcLayer = Layers.Layer((50, 784), False)
    outLayer = Layers.Layer((10, 50), True)
    
    lsLayer = []
    lsLayer.append(fcLayer)
    lsLayer.append(outLayer)
    
    cnn = Network.Network(training_data, validation_data, lsLayer)
    cnn.SGD(10, 20, 0.1)