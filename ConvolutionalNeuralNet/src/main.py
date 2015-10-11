'''
Created on Oct 9, 2015

@author: taivu
'''

import Layers
import Network
import mnist_loader as ml

if __name__ == '__main__':
    
    training_data, validation_data, test_data = ml.load_data_unify()

    
    convLayer = Layers.ConvolLayer((3, 1, 5, 5), 1)
    poolLayer = Layers.PoolLayer((2,2), 2)
    fcLayer = Layers.FCLayer((50, 432), False)
    outLayer = Layers.FCLayer((10, 50), True)
    
    lsLayer = []
    lsLayer.append(convLayer)
    lsLayer.append(poolLayer)
    lsLayer.append(fcLayer)
    lsLayer.append(outLayer)
    
    cnn = Network.Network(training_data, validation_data, lsLayer)
    cnn.SGD(10, 10, 0.1)