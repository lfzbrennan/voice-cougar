import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models import PHNModel
from dataset import TIMIT
from logger import Logger
from utils import save_model

from tqdm import trange, tqdm
import os



def train(output_dir):
	########## set device to gpu
	device = torch.device("cuda")

	########## hyperparams
	learning_rate = 1e-4
	batch_size = 8
	val_batch_size = 48
	log_steps = 1000
	epochs = 100

	vocab_size = 66

	########## set up logging
	print("Setting up logging...")
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	logger = Logger(output_dir + "/out.log")

	########## build model
	print("Building model...")
	model = PHNModel(vocab_size=vocab_size)

	########## create datasets and dataloaders
	print("Building datasets...")
	train_dataset = TIMIT(train=True)
	validation_dataset = TIMIT(train=False)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	validation_dataloader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=True, num_workers=4)

	print("Building optimizer...")
	no_decay = ['bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': .01},
		{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
		]

	########## build optimizer and loss function
	opt = AdamW(optimizer_grouped_parameters, learning_rate)
	criterion = nn.CrossEntropyLoss()
	########## training loop
	model = nn.DataParallel(model)
	model.to(device)
	count = 0
	print("Starting training")
	for epoch in trange(0, epochs, desc="Epochs"):
		epoch_dataloader = tqdm(train_dataloader, desc="Iteration")
		for step, (audio, labels) in enumerate(epoch_dataloader):
			count += 1
			audio = audio.to(device)
			labels = labels.to(device)

			########## model output
			model.train()
			outputs = model(audio)

			########## step optim and loss
			loss = criterion(outputs, labels).mean()
			loss.backward()
			opt.step()
			model.zero_grad()

			########## log if neccessary
			if count % log_steps == 0 or count == 1:
				######### validation set
				average_loss = 0
				model.eval()
				with torch.no_grad():
					validation_iterator = tqdm(validation_dataloader)
					for val_step, (val_audio, val_labels) in enumerate(validation_iterator):
						val_audio = val_audio.to(device)
						val_labels = val_labels.to(device)
						val_outputs = model(val_audio)
						average_loss += criterion(val_outputs, val_labels).mean().item()

				average_loss /= len(validation_dataloader)

				######### logging
				logger.log(f"Epoch: {epoch}\tStep: {step}\tLoss: {average_loss}")
				save_model(f"{output_dir}/checkpoint-{count}", model)

	save_model(f"final", model)


if __name__ == "__main__":
	train("outputs/train1")