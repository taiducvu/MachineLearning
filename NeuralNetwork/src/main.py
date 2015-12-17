'''
Created on Dec 7, 2015

@author: tai
'''

import Layers
import Network
import Algorithms
import mnist_loader as ml

if __name__ == '__main__':
    
    training_data, validation_data, test_data = ml.load_data_unify() # This source code is written by Bao Chiem

    num_Alg = 3
    lsAlg = []        
    lsLayer = []
    
    #Hien tai tao dang dung 1 thuat toan SGD muon 3 thuat toan thi bo 2 dau thang o duoi di
    #lsAlg.append(Algorithms.SGD(0.1))
    #lsAlg.append(Algorithms.Momentum(0.01, 0.9))
    #lsAlg.append(Algorithms.NesterovMomentum(0.01, 0.9))
    lsAlg.append(Algorithms.AdaGrad(0.1))
    
    for i in range(num_Alg):
        lsLayer.append([])
    for i in range(num_Alg):
        lsLayer[i].append(Layers.Layer((30, 784), False))
        lsLayer[i].append(Layers.Layer((10, 30), True))
      
    
    #Set True neu muon early stoping + Khi set early stoping chi duoc dung 1 thuat toan
    nn = Network.Network(training_data, validation_data, lsLayer, lsAlg, True)
    nn.Train(50, 20, 0.1)