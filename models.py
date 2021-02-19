import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import math
import numpy as np

from layers import *


class PHNModel(nn.Module):
	def __init__(self, gate_channels=512, residual_channels=128, skip_channels=1024, end_channels=512, vocab_size=None):
		super().__init__()

		self.gate_channels = gate_channels
		self.residual_channels = residual_channels
		self.skip_channels = skip_channels
		self.end_channels = end_channels
		self.vocab_size = vocab_size

		# (n_layers, output_channel)
		self.blocks = [
			10, 
			10,
			10, 
			10
		]

		self.input_linear = Conv1d1x1(1, self.residual_channels)

		# base wavenet blocks
		self.wavenet_blocks = nn.ModuleList()

		for i, layers in enumerate(self.blocks):

			self.wavenet_blocks.append(
				WavenetBlock(gate_channels=self.gate_channels, residual_channels=self.residual_channels, skip_channels=self.skip_channels, n_layers=layers)
			)

		self.skip_layer = Conv1d1x1(self.skip_channels, self.end_channels)
		self.final_layer = Conv1d1x1(self.end_channels, self.vocab_size)


	def forward(self, audio):

		audio = self.input_linear(audio)
		# push through wavenet blocks
		skip = 0
		for i, block in enumerate(self.wavenet_blocks):
			audio, skip = block(audio, skip)

		#print(out.shape)
		out = F.relu(self.skip_layer(skip))

		out = self.final_layer(out)

		return out

class Mapping(nn.Module):
	def __init__(self, hidden, layers, vocab_size):
		super().__init__()

		self.lstm = nn.LSTM(input_size=hidden * 2, hidden_size=2, num_layers=layers, bidirectional=True)

		self.vocab_head = nn.Linear(hidden*2, vocab_size)
		self.timing_head = nn.Linear(hidden*2, 1)

	def head(self, input):
		vocab = self.vocab_head(input)
		timing = self.timing_head(input)

		return vocab, timing

	def forward(self, input):

		out, hidden = self.lstm(input)
		v_out, t_out = self.head(out[-1])

		while True:
			out, hidden = self.lstm(out, hidden)
			dv_out, dt_out = self.head(out[-1])
			v_out = torch.cat((v_out, dv_out))
			t_out = torch.cat((t_out, dt_out))

		return v_out, t_out



class PHNMappingModel(nn.Module):
	def __init__(self, gate_channels=512, residual_channels=128, skip_channels=1024, end_channels=512, vocab_size=None, lstm_hidden=256, lstm_layers=2):
		super().__init__()

		self.wavenet = PHNModel(gate_channels, residual_channels, skip_channels, end_channels, vocab_size)

		self.mapping = Mapping(lstm_hidden, lstm_layers, vocab_size)

	def forward(self, audio):
		out = self.wavenet(audio)
		v_out, t_out = self.mapping(out)
		return v_out, t_out