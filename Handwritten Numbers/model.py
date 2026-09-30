import numpy as np

class Model:
	def __init__(self, n_features):
		self.n_features = n_features # 784
		self.n_hidden1 = 128
		self.n_hidden2 = 64
		self.n_output = 10
		self.W1 = np.random.randn(self.n_features, self.n_hidden1) * np.sqrt(2 / self.n_features) # 784×128
		self.b1 = np.zeros((self.n_hidden1,)) # 1×128
		self.W2 = np.random.randn(self.n_hidden1, self.n_hidden2) * np.sqrt(2 / self.n_hidden1) # 128×64
		self.b2 = np.zeros((self.n_hidden2,)) # 1×64
		self.W3 = np.random.randn(self.n_hidden2, self.n_output) * np.sqrt(2 / self.n_hidden2) # 64×10
		self.b3 = np.zeros((self.n_output,))
	
	def linear_layer(self, X):
		Z = X @ self.W1 + self.b1 # 100×128
		return Z
	
	def ReLU(self, x):
		return np.maximum(0, x)
	
	def softmax(self, x):
		exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
		return exp_x / np.sum(exp_x, axis=1, keepdims=True)
	
	def forward_pass(self, X):
		Z1 = self.linear_layer(X)
		A1 = self.ReLU(Z1) # 100×128
		Z2 = A1 @ self.W2 + self.b2 # 100×64
		A2 = self.ReLU(Z2) # 100×64
		Z3 = A2 @ self.W3 + self.b3 # 100×10
		A3 = self.softmax(Z3) # 100×10
		return Z1, A1, Z2, A2, Z3, A3