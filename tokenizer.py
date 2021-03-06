import numpy as np 
import torch

class Tokenizer():
	def __init__(self, vocab_file):

		self.silence_token = "[NULL]"
		self.break_token = "[BRK]"

		self.token2id = {}
		with open(vocab_file) as f:
			lines = f.readlines()
			for line in lines:
				self.token2id[line.strip("\n")] = len(self.token2id)

		self.id2token = {}
		for k, v in self.token2id.items():
			self.id2token[v] = k

	def convert_token(self, token):
		return self.token2id[token]
	def convert_tokens_to_ids(self, tokens):
		return [self.token2id[t] if t in self.token2id else self.token2id[self.silence_token] for t in tokens]

	def convert_ids_to_tokens(self, ids):
		return [self.id2token[i] if i in self.id2token else self.silence_token for i in ids]

	def __len__(self):
		return len(self.token2id)