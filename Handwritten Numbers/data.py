import numpy as np
import pandas as pd

class MNISTData:
	def __init__(self):
		self.train_path = "mnist_train.csv"
		self.test_path = "mnist_test.csv"
		self.X = None
		self.y = None
		self.X_test = None
		self.y_test = None
	
	def load(self):
		self.train_df = pd.read_csv(self.train_path, header=None)
		self.test_df = pd.read_csv(self.test_path, header=None)
	
	def separate(self):
		self.X = self.train_df.iloc[:, 1:]
		self.y = self.train_df.iloc[:, 0]
		self.X_test = self.test_df.iloc[:, 1:]
		self.y_test = self.test_df.iloc[:, 0]
	
	def preprocess(self):
		self.X = np.array(self.X) / 255.0
		self.y = np.array(self.y)
		self.y = np.eye(10)[self.y]
		self.X_test = np.array(self.X_test) / 255.0
		self.y_test = np.array(self.y_test)
		self.y_test = np.eye(10)[self.y_test]