'''
Created on Oct 5, 2015

@author: taivu
'''
import numpy as np
import theano
import theano.tensor as T
from theano.scalar.basic import int32

## Declare symbolic variables
Xtrain, Ytrain =T.dmatrices('Xtrain', 'Ytrain')

