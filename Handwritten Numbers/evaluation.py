import numpy as np
from PIL import Image
from scipy import ndimage

class Evaluator:
	def __init__(self, model):
		self.model = model
	
	def predict(self, X):
		Z1, A1, Z2, A2, Z3, A3 = self.model.forward_pass(X)
		pred = np.argmax(A3, axis=1)
		return pred
	
	def accuracy(self, X, y):
		pred = self.predict(X)
		true = np.argmax(y, axis=1)
		correct = np.mean(pred == true)
		return correct
	
	def evaluate(self, X_test, y_test):
		accuracy = self.accuracy(X_test, y_test)
		print(f"Test Accuracy: {accuracy * 100:.2f}")
		return accuracy
	
	def pred_one(self, X):
		pred = self.predict(X)
		return pred[0]
	
	def pred_img(self, path):
		img = Image.open(path).convert("L")
		img.thumbnail((400, 400))
		a = np.array(img, dtype=np.float32)
		paper = ndimage.maximum_filter(a, size=25)
		paper = ndimage.gaussian_filter(paper, sigma=15)
		ink = np.clip(1.0 - a / np.maximum(paper, 1), 0, 1)
		ink = np.clip((ink - 0.15) / 0.5, 0, 1)
		ink = (ink * 255).astype(np.uint8)
		rows = np.any(ink > 0, axis=1)
		cols = np.any(ink > 0, axis=0)
		if not rows.any():
			raise ValueError("Unable to detect digit in image")
		y0, y1 = np.where(rows)[0][[0, -1]]
		x0, x1 = np.where(cols)[0][[0, -1]]
		ink = ink[y0:y1 + 1, x0:x1 + 1]
		h, w = ink.shape
		k = max(3, int(0.06 * max(h, w)))
		ink = ndimage.grey_dilation(ink, size=(k, k))
		img = Image.fromarray(ink)
		s = 20.0 / max(w, h)
		img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
		canvas = Image.new("L", (28, 28), 0)
		canvas.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))
		arr = np.array(canvas, dtype=np.float32)
		cy, cx = ndimage.center_of_mass(arr)
		arr = ndimage.shift(arr, (14 - cy, 14 - cx), order=1, mode="constant")
		if arr.max() > 0:
			arr = arr / arr.max() * 255
#		Image.fromarray(arr.astype(np.uint8)).resize((280, 280), Image.NEAREST).save("debug.png")
		probs = self.model.forward_pass((arr / 255.0).reshape(1, -1))[-1][0]
		return int(np.argmax(probs)), float(np.max(probs))