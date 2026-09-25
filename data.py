import numpy as np

class DataManager:
	def __init__(self):
		self.data = [] # since dataset is small
		self.clean_data = []
		self.path = "SMSSpamCollection.txt"
		
	def sanitize_data(self):
		self.get_data()
		for i in range(len(self.data)):
			label, message = self.data[i].split("\t", 1)
			pairs = (label,message)
			self.clean_data.append(pairs)
		return self.clean_data
		
	def get_data(self):
		with open(self.path, "r", encoding="utf-8") as f:
			self.data = f.readlines()
		return self.data
		
class DataProcessor:
	def __init__(self):
		self.Y = []
		self.X = None
		self.vocab = {"<UNK>": 0}
		self.label_map = {"ham": 0, "spam": 1}
		self.data_manager = DataManager()
		self.clean_data = self.data_manager.sanitize_data()
		
	def encoding_label(self):
		for label, message in self.clean_data:
			self.Y.append(self.label_map[label])
		return np.array(self.Y)
		
	def build_vocab(self):
		word_list = set()
		for label, message in self.clean_data:
			word_list.update(message.split())
		for token, word in enumerate(word_list, start=1):
			self.vocab[word] = token
		self.vocab_size = len(self.vocab)
		return len(self.clean_data), self.vocab_size
		
	def encode_message(self, text):
		tokens = []
		text = text.split()
		for word in text:
			if word in self.vocab:
				token = self.vocab[word]
				tokens.append(token)
			else:
				token = self.vocab["<UNK>"]
				tokens.append(token)
		return tokens
	
	def prepare_one(self, message):
		tokens = self.encode_message(message)
		row = np.zeros(self.vocab_size, dtype=np.float32)
		for token_id in tokens:
			row[token_id] = 1
		return row
		
	def prepare_X(self):
		self.build_vocab()
		rows = []
		for label, message in self.clean_data:
			row = self.prepare_one(message)
			rows.append(row)
		self.X = np.array(rows, dtype=np.float32)
		return self.X