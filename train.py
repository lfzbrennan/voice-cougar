import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models import PHNModel
from dataset import TIMIT
from logger import Logger
from utils import save_model

from tqdm import trange, tqdm



def train(output_dir):
	########## set device to gpu
	device = torch.device("cuda")

	########## hyperparams
	learning_rate = 1e-3
	batch_size = 8
	val_batch_size = 48
	log_steps = 1000
	epochs = 50

	vocab_size = 58

	########## set up logging
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	logger = Logger(output_dir + "/out.log")

	########## build model
	model = PHNModel()
	model = nn.DataParallel(model)
	model.to(device)

	########## create datasets and dataloaders
	train_dataset = TIMIT(train=True)
	validation_dataset = TIMIT(train=False)

	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	validation_dataloader = DataLoader(validation_dataset, batch_size=val_batch_size, shuffle=True, num_workers=4)

	########## build optimizer and loss function
	opt = AdamW(model.parameters(), learning_rate)
	criterion = nn.CrossEntropyLoss()

	########## training loop
	count = 0
	for epoch in trange(0, epochs, desc="Epochs"):
		epoch_dataloader = tqdm(train_dataloader)
		for step, (audio, labels) in enumerate(epoch_dataloader):
			count += 1
			audio = audio.to(device)
			labels = label.to(device)

			########## model output
			model.train()
			outputs = model(audio)

			########## step optim and loss
			loss = criterion(outputs.view(-1, vocab_size), labels)
			loss.backward()
			opt.step()
			model.zero_grad()

			########## log if neccessary
			if count % log_steps == 0:
				######### validation set
				average_loss = 0
				model.eval()
				with torch.no_grad():
					validation_iterator = tqdm(validation_dataloader)
					for val_step, (audio, labels) in enumerate(validation_iterator):
						audio = audio.to(device)
						labels = labels.to(device)
						outputs = model(audio)
						average_loss += criterion(outputs.view(-1, vocab_size), labels).mean().item()

				average_loss /= len(validation_dataloader)

				######### logging
				logger.log(f"Epoch: {epoch}\tStep: {step}\tLoss: {average_loss}")
				save_model(f"checkpoint-{count}", model)

	save_model(f"final", model)


if __name__ == "__main__":
	train("outputs/train1")