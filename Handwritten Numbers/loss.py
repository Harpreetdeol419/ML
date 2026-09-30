import numpy as np

def MCELoss(A3, y):
	loss = -np.mean(np.sum(y * np.log(A3 + 1e-8), axis=1))
	return loss