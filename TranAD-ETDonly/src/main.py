import sys
from tqdm import tqdm
from time import time

from config import *
from utils import *
from parser import *
from plot import *
from preprocess import *
from dataset import *
from tran_ad import *
from pprint import pprint

if __name__ == '__main__':
	# 数据预处理
	if args.preprocess:
		commands = sys.argv[2:]
		load = []
		if len(commands) > 0:
			for d in commands:
				load_data(d, output_folder, data_folder)
		else:
			print("Usage: python preprocess.py <datasets>")
			print(f"where <datasets> is space separated list of {datasets}")

	# 加载数据集
	train_loader, test_loader, labels = load_dataset(args.dataset)

	# 加载模型
	model, optimizer, scheduler, epoch, accuracy_list = load_model(labels.shape[1])

	# 将数据集制作成时间序列窗口
	train_origin, test_origin = next(iter(train_loader)), next(iter(test_loader))
	train_seq, test_seq = convert_to_windows(train_origin, model), convert_to_windows(test_origin, model)

	# 如果已经有训练的模型，在此基础上继续训练，否则创建一个新的模型训练
	if args.train:
		print(f'{color.HEADER}Training {model.name} on {args.dataset}{color.ENDC}')

		num_epochs = 5; e = epoch + 1; start = time()

		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = back_propagation(e, model, train_seq, train_origin, optimizer, scheduler)
			accuracy_list.append((lossT, lr))

		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{model.name}_{args.dataset}')

	if args.test:
		### Testing phase
		torch.zero_grad = True
		model.eval()
		print(f'{color.HEADER}Testing TranAD on {args.dataset}{color.ENDC}')
		loss, y_pred = backprop(0, model, test_seq, test_origin, optimizer, scheduler, training=False)

		### Plot curves
		if 'TranAD' in model.name: test_origin = torch.roll(test_origin, 1, 0)
		plotter(f'TranAD_{args.dataset}', test_origin, y_pred, loss, labels)
		error_report(f'TranAD_{args.dataset}', loss)

#python src\main.py --d ETD --train --test