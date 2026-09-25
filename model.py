import numpy as np
from data import (
	DataProcessor
)

class Model:
	def __init__(self):
		self.processor = DataProcessor()
		self.data_size, self.vocab_size = self.processor.build_vocab()
		self.X = self.processor.prepare_X()
		self.Y = self.processor.encoding_label()
		self.W = np.random.randn(self.vocab_size, 1).astype(np.float32)
		self.b = np.zeros((1,))
		self.lr = 0.01
		self.epsilon = 1e-8
		self.threshold = 0.5
		
	def sigmoid(self, Z):
		return 1 / (1 + np.exp(-Z))
	
	# BCE
	def loss(self, Y, A):
		return np.mean(-(Y * np.log(A + self.epsilon) + (1 - Y) * np.log(1 - A + self.epsilon)))
		
	def forward(self):
		Z = self.X @ self.W + self.b
		A = self.sigmoid(Z)
		return Z, A
	
	def train_model(self, epochs=100):
		print("=======================\nTraining :-")
		for epoch in range(epochs):
			Y = self.Y.reshape(-1, 1)
			Z, A = self.forward()
			loss = self.loss(Y, A)
			dl_dw = (self.X.T @ ((np.exp(-Z) / (1 + np.exp(-Z)) ** 2) * ((A - Y) / (A * (1 -  A))))) / self.data_size # dl / dw
			dz_db = 1
			dl_db = np.mean(((np.exp(-Z) / (1 + np.exp(-Z)) ** 2) * ((A - Y) / (A * (1 -  A)))) * 1)
			self.W = self.W - self.lr * dl_dw
			self.b = self.b - self.lr * dl_db
			if epoch % 10 == 0:
				print(f"Loss => {loss}")
			
	def inference(self):
		self.train_model()
		while True:
			text = input("\n=======================\nEnter text: ")
			X = self.processor.prepare_one(text)
			X = X.reshape(-1, 1)
			X = X.T
			Z = X @ self.W + self.b
			A = self.sigmoid(Z)
			if A.item() > self.threshold:
				print("Model prediction: Spam")
			else:
				print("Model prediction: Not spam")

md = Model()
md.inference()