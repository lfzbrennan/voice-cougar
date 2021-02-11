import torch
from torch.utils.data import Dataset

import numpy as np
import glob
import random

from utils import load_audio, get_phn
from tokenizer import Tokenizer

class TIMIT(Dataset):
	def __init__(self, train=True, audio_length=2**14):
		self.audio_length = audio_length
		self.train = train


		if self.train:
			self.files = glob.glob("/u/lab/lfb6ek/datasets/timit/data/TRAIN/**/.PHN", recursive=True)
		else:
			self.files = glob.glob("/u/lab/lfb6ek/datasets/timit/data/TEST/**/.PHN", recursive=True)

		self.silence_files = glob.glob("/u/lab/lfb6ek/datasets/timit/data/SILENCE/*.wav")

		self.tokenizer = Tokenizer("vocab/phn.txt")

	def __len__(self):
		return len(self.files)
	def __getitem__(self, idx):
		phn_file = self.files[idx]
		wav_file = self.files[idx][:-3] + "WAV.wav"

		labels = self.pad(get_phn(phn_file, self.tokenizer), pad_value=self.tokenizer.convert_tokens_to_ids("[NULL]"))
		audio = load_audio(wav_file)

		random_index = random.randint(len(audio) - self.audio_length)
		labels, audio = labels[random_index:random_index + self.audio_length], audio[random_index:random_index + self.audio_length]

		if random.choice([0, 1]) == 1:
			labels, audio = self.add_silence(labels, audio)

		audio = self.normalize_audio(audio)
		return torch.FloatTensor(audio), torch.IntTensor(labels)
	def normalize_audio(self, audio):
		return audio / (2 ** 16)
	def add_silence(self, labels, audio):
		silence_audio = load_audio(random.choice(self.silence_files))
		silence_length = random.randint(self.audio_length // 4, self.audio_length * 3 // 4)

		silence_index = random.randint(len(silence_audio) - silence_length)
		silence_audio = silence_audio[silence_index:silence_index + silence_length]
		if random.choice([0, 1]) == 1:
			audio[:silence_index] = silence_audio
			labels[:silence_index] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.silence_token)
		else:
			audio[silence_index:] = silence_audio
			labels[silence_index:] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.silence_token)

		return labels, audio


	def pad(self, arr, legnth=self.audio_length, pad_value=0):
		if len(arr) >= self.audio_length:
			return arr
		return np.concat((arr, [pad_value] * (self.audio_length - len(arr)))