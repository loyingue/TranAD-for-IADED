import os
import torch
from config import *
import torch.nn as nn
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

def cut_array(percentage, arr):
	print(f'{color.BOLD}Slicing dataset to {int(percentage*100)}%{color.ENDC}')
	mid = round(arr.shape[0] / 2)
	window = round(arr.shape[0] * percentage * 0.5)
	return arr[mid - window : mid + window, :]

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'./checkpoints/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/TranAD_{args.dataset}.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(dims):
	from tran_ad import TranAD
	model = TranAD(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'../checkpoints/TranAD_{args.dataset}.ckpt'
	if os.path.exists(fname) and (args.train or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction='mean' if training else 'none')
	feats = dataO.shape[1]
	l = nn.MSELoss(reduction='none')
	data_x = torch.DoubleTensor(data);
	dataset = TensorDataset(data_x, data_x)
	bs = model.batch if training else len(data)
	dataloader = DataLoader(dataset, batch_size=bs)
	n = epoch + 1;
	w_size = model.n_window
	l1s, l2s = [], []
	if training:
		for d, _ in dataloader:
			local_bs = d.shape[0]
			window = d.permute(1, 0, 2)  # 交换维度
			elem = window[-1, :, :].view(1, local_bs, feats)  # 改变维度
			z = model(window, elem)  # 你他妈在干啥？
			l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
			if isinstance(z, tuple): z = z[1]
			l1s.append(torch.mean(l1).item())
			loss = torch.mean(l1)  # l1是损失函数
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
		scheduler.step()
		tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']
	else:
		for d, _ in dataloader:
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, bs, feats)
			z = model(window, elem)
			if isinstance(z, tuple): z = z[1]
		loss = l(z, elem)[0]
		return loss.detach().numpy(), z.detach().numpy()[0]

