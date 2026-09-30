import os
from model import Model
from training import Trainer
from data import MNISTData
from evaluation import Evaluator

# load Data
data = MNISTData()
data.load()
data.separate()
data.preprocess()

model = Model(data.X.shape[1])

# train
trainer = Trainer(learning_rate=0.01)
saved = os.path.exists("model_params.npz")
ask = input("[-] Start training (y/n): ").lower() if saved else "y"
if ask == "y":
	trainer.fit(model, data.X, data.y, epochs=30, batch_size=100)
else:
	trainer.load(model)
	print("[-] Loaded saved parameters, skipping training")

# evaluate
eva = Evaluator(model)
eva.evaluate(data.X_test, data.y_test)
while True:
	img = input("\n[-] Enter image path: ")
	num, conf = eva.pred_img(img)
	print(f"[-] Prediction: {num}, Confidence: {conf * 100:.2f}%")