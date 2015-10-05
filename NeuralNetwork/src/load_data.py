import cPickle
import gzip
import numpy as np

def load_data():
	f = gzip.open('../data/mnist.pkl.gz', 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	return (training_data, validation_data, test_data)

def vectorized_result(j):
	result = np.zeros((10, 1))
	result[j] = 1
	return result