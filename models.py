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