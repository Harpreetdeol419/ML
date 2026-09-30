import os
import numpy as np
from loss import MCELoss
from backpropagation import dL_db1, dL_dW1, dL_db2, dL_dW2, dL_db3, dL_dW3

class Trainer:
	def __init__(self, learning_rate):
		self.learn = learning_rate
	
	def update(self, param, grad):
		return param - self.learn * grad
	
	def iterate_batches(self, X, y, batch_size=100):
	    idx = np.random.permutation(len(X))
	    for s in range(0, len(X), batch_size):
	        b = idx[s:s + batch_size]
	        Xb = X[b].reshape(-1, 28, 28)
	        dx, dy = np.random.randint(-2, 3, 2)
	        Xb = np.roll(Xb, (dy, dx), axis=(1, 2))
	        yield Xb.reshape(len(b), -1), y[b]
	
	def train(self, model, y, X, batch_size=100):
		total = 0
		for X_batch, y_batch in self.iterate_batches(X, y, batch_size):
			Z1, A1, Z2, A2, Z3, A3 = model.forward_pass(X_batch)
			loss = MCELoss(A3, y_batch)
			total += loss
			# layer 3
			dW3 = dL_dW3(A3, A2, y_batch)
			db3 = dL_db3(A3, y_batch)
			# layer 2
			dW2 = dL_dW2(model.W3, A3, y_batch, Z2, A1)
			db2 = dL_db2(model.W3, A3, y_batch, Z2)
			#layer 1
			dW1 = dL_dW1(Z1, model.W3, A3, y_batch, Z2, model.W2, X_batch)
			db1 = dL_db1(Z1, model.W3, A3, y_batch, Z2, model.W2)
			# updates
			model.W3 = self.update(model.W3, dW3)
			model.b3 = self.update(model.b3, db3)
			model.W2 = self.update(model.W2, dW2)
			model.b2 = self.update(model.b2, db2)
			model.W1 = self.update(model.W1, dW1)
			model.b1 = self.update(model.b1, db1)
		return total
	
	def save(self, model, file="model_params.npz"):
		np.savez(file, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2, W3=model.W3, b3=model.b3)
	
	def load(self, model, file="model_params.npz"):
		params = np.load(file)
		model.W1 = params["W1"]
		model.b1 = params["b1"]
		model.W2 = params["W2"]
		model.b2 = params["b2"]
		model.W3 = params["W3"]
		model.b3 = params["b3"]
	
	def setup(self, model, file="model_params.npz"):
		if os.path.exists(file):
			ask = input("[-] Resume training (y/n): ")
			if ask.lower() == "y":
				self.load(model, file)
				print("[-] Loaded saved parameters")
			else:
				print("[-] Started fresh training")
		else:
			print("[-] Started fresh training")
	
	def fit(self, model, X, y, epochs=20, batch_size=100):
		self.setup(model)
		for epoch in range(epochs):
			total = self.train(model, y, X, batch_size)
			num_batches = (len(X) + batch_size - 1) // batch_size
			avg = total / num_batches
			print(
				f"Epoch {epoch + 1} / {epochs} - "
				f"Loss {avg:.6f}"
			)
			self.save(model)