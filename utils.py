import librosa

def load_audio(audio):
	return librosa.load(audio)[0]

def get_phn(file, tokenizer):
	phn_list = []
	with open(file) as f:
		for line in f.readlines()
			start, end, phn = line.split()
			phn_list.append((start, end, tokenizer.convert_tokens_to_ids(phn)))