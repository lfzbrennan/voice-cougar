import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
import numpy as np


# 1d convolution wrapper

def Conv1d(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
	m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
	nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
	if m.bias is not None:
		nn.init.constant_(m.bias, 0)
	return nn.utils.weight_norm(m)


# 1x1 1d convolution wrapper
def Conv1d1x1(in_channels, out_channels, bias=True):
	return Conv1d(in_channels, out_channels, kernel_size=1, padding=0,
				  dilation=1, bias=bias)

		
class ResConvLayer(nn.Module)
	def __init__(self, resid_channels, gate_channels, kernel_size, dilation):
		super().__init__()

		# 1d styled convolution

		self.conv = EqualConv1d(resid_channels, gate_channels, kernel_size=kernel_size, dilation=dilation, padding=(kernel_size - 1) // 2 * dilation)

		# output convolution
		self.conv1x1_out = Conv1d1x1(gate_channels // 2, resid_channels)

	def forward(self, x):
		residual = x

		x = self.conv(x)

		splitdim = 1
		xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

		x = torch.tanh(xa) * torch.sigmoid(xb)

		# for residual connection
		x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

		return x

## descriminator block
class WavenetBlock(nn.Module):
	def __init__(self, resid_channels = 512, out_channels = 512, gate_channels = 512, kernel_size = 3, n_layers = 10, dilation_factor = 2):
		super().__init__()


		#### input = (B, resid_channels, T)
		#### output = (B, out_channels, T*4)
		
		self.conv_layers = nn.ModuleList()

		for i in range(n_layers):
			# add residual layer
			dilation = dilation_factor ** i
			self.conv_layers.append(ResConvLayer(resid_channels, gate_channels, kernel_size, dilation))

		if resid_channels != out_channels:
			self.conv_layers.append(Conv1d1x1(resid_channels, out_channels))

	def forward(self, x):
		for layer in self.conv_layers:
			x = layer(x)

		return x

