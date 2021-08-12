import numpy as np

#loss function and its derivative
def mse(y_true, y_pred):
	return 2*np.mean(np.power(y_true - y_pred, 2));
	
def mse_prime(y_true, y_pred):
	return (y_pred - y_true)/y_true.size;

