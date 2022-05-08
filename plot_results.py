import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import dataset
from datetime import datetime
import random
import numpy as np
import copy
import torchvision.models as models
import detector_net
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import os.path
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--type', default='cifar100', help='dataset for training')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 64)')
parser.add_argument('--la', type=float, default = 0.6)
parser.add_argument('--lc', type=float, default = 0.1)
parser.add_argument('--pgd_params', default = '8_4_10')
parser.add_argument('--baseline_model_pth', default = './models/task_train_with_cifar100_sles_mobilenet_v2_out1_16_stage_2_out_16')
parser.add_argument('--test_pth', default = './cifar100/adv_test_set_cifar100_task_cifar100_sles_mobilenet_v2')
parser.add_argument('--task', default = 'PGD50_training')
parser.add_argument('--p', type=float, default = 100)
parser.add_argument('--out', default = '16,32,64')
parser.add_argument('--stage', type=int, default=2,  help='how many stages')

args = parser.parse_args()
out1, out2, out3 = map(int, args.out.split(','))

print('==> Building model..')
if args.type == 'cifar10':
    trainloader, test_loader = dataset.get10(batch_size=args.batch_size)
    model = detector_net.Net()

if args.type == 'cifar100':
    trainloader, test_loader = dataset.get100(batch_size=args.batch_size)
    model = detector_net.Net()

    model.conv1 = nn.Conv2d(3, out1, kernel_size=3, padding=1,bias=True)
    model.conv2 = nn.Conv2d(out1, out2, kernel_size=3, padding=1, bias=True)
    model.conv3 = nn.Conv2d(out2, out3, kernel_size=3, padding=1, bias=True)
    model.conv4 = nn.Conv2d(out3, 32, kernel_size=3, padding=1, bias=True)

model = torch.nn.DataParallel(model)
load_file = torch.load(args.baseline_model_pth, map_location='cpu')
try:
    model.load_state_dict(load_file.state_dict())
except:
    model.load_state_dict(load_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda': model = model.cuda()
model.eval()

criterion = nn.CrossEntropyLoss()

if device == 'cuda':
    lambda_adv = torch.tensor([args.la]).cuda()
    lambda_clean = torch.tensor([args.lc]).cuda()
else:
    lambda_adv = torch.tensor([args.la])
    lambda_clean = torch.tensor([args.lc])

mseloss = torch.nn.MSELoss()
celoss = torch.nn.CrossEntropyLoss()
print('==> Load dataset..')
test_loader_adv = torch.load(args.test_pth, map_location='cpu')

test_loss = 0
correct = 0
energy_a_loss = 0
energy_avg = 0
energy_mean_a = 0
print('==> Testing phase adv..')
soi_list_a = []
soi_list_a_energy = []
for i, (data, target) in enumerate(tqdm(test_loader_adv)):
    indx_target = target.clone()
    if device == 'cuda': data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        energy_a = model(data, args.stage)
        
        energy_mean_a += energy_a.mean()
        energy_a_loss += mseloss(energy_a, lambda_adv)

    tuple_da = (energy_a, target)

    soi_list_a.append(tuple_da)
    soi_list_a_energy += energy_a
energy_a_loss = energy_a_loss/len(test_loader_adv)
mean_soi_a = energy_mean_a/len(test_loader_adv)


energy_c = 0
test_loss = 0
correct = 0
energy_c_loss = 0
energy_mean_c = 0
soi_list_c = []
soi_list_c_energy = []
print('==> Testing phase clean..')
for i, (data, target) in enumerate(tqdm(test_loader)):
    indx_target = target.clone()
    data, target = data.cuda(), target.cuda()
    with torch.no_grad():
        data, target = Variable(data), Variable(target)
        energy_c = model(data, args.stage)

        energy_mean_c += energy_c.mean()
        energy_c_loss += mseloss(energy_c, lambda_clean)
    tuple_dc = (energy_c, target)
    soi_list_c.append(tuple_dc)
    soi_list_c_energy += energy_c
energy_c_loss = energy_c_loss/len(test_loader)
mean_soi_c = energy_mean_c / len(test_loader)


soi_list_a_energy = [float(x.detach().cpu().numpy()) for x in soi_list_a_energy]
soi_list_c_energy = [float(x.detach().cpu().numpy()) for x in soi_list_c_energy]

plt.figure(figsize=(12, 10))
plt.hist(soi_list_c_energy, bins=200, label="Natural")
plt.hist(soi_list_a_energy, bins=200, label="Adversarial")
plt.title(args.test_pth.split('/')[-1] + '_stage_' + str(args.stage), fontsize=15)
plt.xlabel('Energy Values', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.legend(fontsize='large')
plt.savefig('result.jpg')