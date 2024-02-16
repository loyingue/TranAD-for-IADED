import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import scienceplots
import statistics
import os, torch
from sklearn.metrics import *

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

import matplotlib as mpl

# 设置可选的serif字体列表，确保至少有一个是存在的
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'STIX', 'Palatino Linotype']

# 如果不需要serif字体，也可以全局设置为sans-serif
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = [...]


warning_threshold = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
error_threshold = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
warning_severity = 0.1


def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
	if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()


def plot_accuracies(accuracy_list, folder):
	os.makedirs(f'./plots/{folder}/', exist_ok=True)
	trainAcc = [i[0] for i in accuracy_list]
	lrs = [i[1] for i in accuracy_list]
	plt.xlabel('Epochs')
	plt.ylabel('Average Training Loss')
	plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', linewidth=1, linestyle='-', marker='.')
	plt.twinx()
	plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', linewidth=1, linestyle='--', marker='.')
	plt.savefig(f'./plots/{folder}/training-graph.pdf')
	plt.clf()


def error_report(name, ascore):
	rate = 0
	rate_series = []
	length = ascore.shape[0]
	for dim in range(ascore.shape[1]):
		warning_rate = sum(ascore[:, dim]>warning_threshold[dim]) / length
		error_rate = sum(ascore[:, dim]>error_threshold[dim]) / length
		rate_series.append([error_rate + warning_rate * warning_severity, warning_rate, error_rate])
		rate += error_rate + warning_rate * warning_severity
	with open(f'./plots/{name}/error_report.txt', 'w') as f:
		f.write(f'---Error Report---\n')
		f.write(f'Error Rate: {rate}\n')
		for i in range(ascore.shape[1]):
			f.write(f'Dimension {i}: {rate_series[i][0]} (Warning Rate: {rate_series[i][1]}, Error Rate: {rate_series[i][2]})\n')
	return rate, rate_series
