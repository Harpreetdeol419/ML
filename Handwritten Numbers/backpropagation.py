import numpy as np

def dL_dZ3(A3, y):
	N = A3.shape[0]
	return (A3 - y) / N # 100 × 10

def dL_dW3(A3, A2, y):
	N = A3.shape[0]
	return A2.T @ ((A3 - y) / N) # 64×10

def dL_db3(A3, y):
	dZ3 = dL_dZ3(A3, y)
	return dZ3.sum(axis=0) # 10,

def dL_dA2(W3, A3, y):
	dZ3 = dL_dZ3(A3, y)
	return dZ3 @ W3.T # 100 × 64

def dL_dZ2(W3, A3, y, Z2):
	dA2 = dL_dA2(W3, A3, y)
	d_relu = (Z2 > 0)
	return dA2 * d_relu # 100×64

def dL_dW2(W3, A3, y, Z2, A1):
	dZ2 = dL_dZ2(W3, A3, y, Z2)
	return A1.T @ dZ2 # 128×64

def dL_db2(W3, A3, y, Z2):
	return dL_dZ2(W3, A3, y, Z2).sum(axis=0) # 64,

def dL_dA1(W3, A3, y, Z2, W2):
	dZ2 = dL_dZ2(W3, A3, y, Z2)
	return dZ2 @ W2.T # 100×128

def dA1_dZ1(Z1):
	return (Z1 > 0) # 100×128

def dL_dZ1(Z1, W3, A3, y, Z2, W2):
	return dL_dA1(W3, A3, y, Z2, W2) * dA1_dZ1(Z1) # 100×128

def dL_dW1(Z1, W3, A3, y, Z2, W2, X):
	return X.T @ dL_dZ1(Z1, W3, A3, y, Z2, W2) # 784×128

def dL_db1(Z1, W3, A3, y, Z2, W2):
	return dL_dZ1(Z1, W3, A3, y, Z2, W2).sum(axis=0) # 128,