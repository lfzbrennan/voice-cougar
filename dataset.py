import torch
from torch.utils.data import Dataset

import numpy as np

import glob

class TIMIT(Dataset):
	def __init__(self, train=True, audio_length=2**14):
		self.audio_length = audio_length
		if train:
			files = glob.glob("/u/lab/lfb6ek/datasets/timit/data/TRAIN/**/.PHN", recursive=True)
		else:
			files = glob.glob("/u/lab/lfb6ek/datasets/timit/data/TEST/**/.PHN", recursive=True)

	def __len__(self):
		pass
	def __getitem__(self, idx):
		pass
	def pad(self, arr, legnth=self.audio_length):
		return np.concat((arr, [0] * (self.audio_length - len(arr)))