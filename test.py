import torch
import torch.nn as nn
import numpy as np 

class exampleLSTM(nn.Module):
	def __init__(self):
		super().__init__()

		self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, bidirectional=True)
		self.fc = nn.Linear(128, 2)
	
	def forward(self, inputs, hidden=None):
		if hidden == None:
			out, hidden = self.lstm(inputs)
		else:
			out, hidden = self.lstm(inputs, hidden)
		for i in range(5):
			out, hidden = self.lstm(out, hidden)

		out = out[-1]

		out = torch.sigmoid(self.fc(out))

		return out

x = torch.randn(2**14, 1, 128)

test = exampleLSTM()
out = test(x)
print(out)
