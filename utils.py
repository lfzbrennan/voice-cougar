import librosa
import numpy as np

def load_audio(audio):
	return librosa.load(audio)[0]

def get_phn(file, tokenizer):
	phn_list = []
	with open(file) as f:
		for line in f.readlines()
			start, end, phn = line.split()
			phn_list.append((start, end, tokenizer.convert_tokens_to_ids(phn)))

	out = np.array([0] * phn_list[-1][1])
	for start, end, phn in phn_list:
		out[start:end] = phn

	return out