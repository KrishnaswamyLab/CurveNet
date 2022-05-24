#!/usr/bin/env python
# coding: utf-8

# # Script for running Hessian Project

from __future__ import print_function
import os, logging, pickle, random, multiprocessing

import numpy as np
from numpy import load, save

import torch
import torch.optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torchvision.datasets import MNIST
from cifar10_data import MNISTRandomLabels

#from torchvision.datasets import CIFAR10, CIFAR100
#from cifar10_data import CIFAR10RandomLabels
#from cifar10_data import CIFAR100RandomLabels

from torch.utils.data.sampler import SubsetRandomSampler

from joblib import Parallel, delayed
from tqdm.notebook import tqdm_notebook

import cmd_args
import model_mlp, model_wideresnet

import phate
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from joblib.externals.loky.backend.context import get_context

class AverageMeter(object):
    
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def eval_model(newparams):
    
    # load model/parameters
    n_units = [784,40,10]
    device = torch.device('cpu')
    model = model_mlp.MLP(n_units)
    checkpoint = torch.load('runs/testmlp/model_e200.t7',map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    #print(next(model.parameters()).is_cuda) # returns a boolean
    
    # Defining parameters as pytorch parameters
    ftmatsz = int(n_units[0]) * int(n_units[1])
    firstparams = torch.nn.Parameter(torch.tensor(newparams[:ftmatsz].reshape(int(n_units[1]),int(n_units[0]))))
    secondparams = torch.nn.Parameter(torch.tensor(newparams[ftmatsz:].reshape(int(n_units[2]),int(n_units[1]))))
    finalparams = [firstparams,secondparams]
    
    # copy new paramters to model
    for (p, w) in zip(model.parameters(), finalparams):
        p.data.copy_(w.type(type(p.data)))
        
    #kwargs = {'num_workers': 1, 'pin_memory': True}
    
    #normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    #                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    
    # MNIST
    transform_train = transforms.Compose([
    transforms.ToTensor() #,normalize
    ])
    train_loader = torch.utils.data.DataLoader(
                    MNISTRandomLabels(root='./data', train=True, transform=transform_train, num_classes=10, corrupt_prob=0.0),
                    batch_size=128, shuffle=False, multiprocessing_context=get_context('loky'),num_workers=1)
    
    criterion = nn.CrossEntropyLoss()#.cpu()  #.cuda()
    losses = AverageMeter()

    # switch to evaluate mode

    model.eval()

    for i, (train_X, target) in enumerate(train_loader):
        
        target = target#.cpu()#.cuda(non_blocking=True)
        #train_X = train_X.cpu()
        
        with torch.no_grad():
            
            input_var = torch.autograd.Variable(train_X)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # record loss
            losses.update(loss.item(), train_X.size(0))
    
    return(losses.avg)

# Train NN 
os.system("python3 train.py --rand_seed=8 --epochs=200 --learning-rate=0.001 --arch='mlp' --mlp-spec=40 --name='testmlp' --data='mnist'")

# Load parameter/loss values

trainloss = np.load("trainloss.npy")
testloss = np.load("testloss.npy")
paramload = np.load("paramvalues.npy")

# Getting L2 for parameter values values
diffltwo = []
for i in range(1,len(paramload)):
    diffltwo.append(np.sqrt(np.sum(paramload[i]**2)) - np.sqrt(np.sum(paramload[i-1]**2)))

plt.figure()
plt.title("Train/Test Loss vs Epochs")
plt.plot(testloss)
plt.plot(trainloss)
plt.yscale('log')


plt.figure()
plt.title("L2 difference between parameters vs epoch")
plt.plot(diffltwo)

# For loop to evaulate loss at new parameter values 

epoch = 25

# Initializing loss list
losslist = [trainloss[epoch]]

# Initializing parameter list
optparam = np.array(paramload[epoch])
saveparams = [optparam]

num_cpus = multiprocessing.cpu_count()

# Compute gradient

deltaloss = trainloss[epoch+1] - trainloss[epoch]
deltaparam = np.sqrt(np.sum((paramload[epoch+1] - paramload[epoch])**2))
derivative = deltaloss/deltaparam

# Loop over radii
saveparams = [optparam]
losslist = [trainloss[epoch]]

alpha = 10
radii = 5
sampleradii = 200

for i in range(radii):
    
    print(i)
    
    # loop over sampled params
    for j in range(sampleradii):
        
        # Defining new parameter vector
        u = np.random.normal(0,1,len(optparam))
        norm = np.sum(u**2)**0.5
        nx = u/norm
        
        # update parameters via vector addition
        newparams = optparam + (nx*derivative*alpha)
        
        # NOTE: could also sample in the dir of the gradient
        #normderiv = derivative/np.max(derivative)
        #newparams = optparam + (np.multiply(nx,optparam)) + derivative
        #newparams = optparam +  np.multiply(nx,optparam)*(delta_loss*1000)
        #newparams = optparam + (delta_param*alpha)*nx

        saveparams.append(newparams)
        
    alpha = alpha + 2
    
losslist = Parallel(n_jobs=num_cpus)(delayed(eval_model)(saveparams[i]) for i in range(1, len(saveparams)))
print("Done!")

# converting parameters to numpy arrays
paramarr = np.stack(saveparams,axis=0)
losslist = np.array(losslist)
tlosslist = losslist.reshape(losslist.shape[0],1)
paramarrls = np.hstack((paramarr,tlosslist))
np.save("paramarr_1000_r10_18_uniformOPT.npy",paramarrls)

# PHATE Plots

phate_op = phate.PHATE(n_components = 3)
data_phate = phate_op.fit_transform(paramarrls)

plt.figure(figsize = (12,8))
plt.scatter(data_phate[:,0], data_phate[:,1], c=losslist)
plt.colorbar()
plt.scatter(data_phate[0,0], data_phate[0,1], c='r', s=100)
plt.title("2D Phate (MLP_MNIST loss landscape)",fontsize=20)
plt.ylabel("PHATE axis 2",fontsize=14)
plt.xlabel("PHATE axis 1",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

#3D plot with loss as z axis
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
p = ax.scatter(data_phate[:,0], data_phate[:,1], losslist, c=losslist)
ax.scatter(data_phate[0,0],data_phate[0,1],losslist[0],c='r',s=200)
ax.set_title("2D Phate and Loss",fontsize=20)
ax.set_xlabel('PHATE axis 1',fontsize=14,labelpad=10)
ax.set_ylabel('PHATE axis 2',fontsize=14,labelpad=10)
ax.set_zlabel('Loss',fontsize=14,labelpad=20)
ax.ticklabel_format(style = 'plain')
ax.ticklabel_format(useOffset=False)
ax.tick_params(axis="x",labelsize=14)
ax.tick_params(axis="y",labelsize=14)
ax.tick_params(axis="z",labelsize=14)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)
cb = fig.colorbar(p,shrink=.65, pad = 0.1)
cb.ticklabel_format(style = 'plain')
cb.ticklabel_format(useOffset=False)
plt.show()

# 3D PHATE plot
fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
p = ax.scatter(data_phate[:,0], data_phate[:,1], data_phate[:,2], c=losslist)
ax.scatter(data_phate[0,0],data_phate[0,1],c='r',s=200)
ax.set_title("3D PHATE",fontsize=20)
ax.set_xlabel('PHATE axis 1',fontsize=14,labelpad=10)
ax.set_ylabel('PHATE axis 2',fontsize=14,labelpad=10)
ax.set_zlabel('PHATE axis 3',fontsize=14,labelpad=10)
ax.tick_params(axis="x",labelsize=14)
ax.tick_params(axis="y",labelsize=14)
ax.tick_params(axis="z",labelsize=14)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.grid(False)
fig.colorbar(p,shrink=.65,pad = 0.05)
plt.show()

# PCA Plot

pca = PCA(n_components=3)
pca.fit(paramarrls.T)

print(paramarr.shape)
print(np.shape(losslist))
print(pca.components_.shape)

plt.figure()
plt.title("PCA")
plt.scatter(pca.components_[0],pca.components_[1],c=losslist)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# 3D PCA Plot

fig = plt.figure(figsize=(10,10))
ax = Axes3D(fig)
p = ax.scatter(pca.components_[0,:], pca.components_[1,:], pca.components_[2,:], c= losslist)
ax.set_title("3D PCA",fontsize=20)
ax.set_xlabel('PC 1',fontsize=14,labelpad=10)
ax.set_ylabel('PC 2',fontsize=14,labelpad=10)
ax.set_zlabel('PC 3',fontsize=14,labelpad=10)

# Analyzing loss

plt.figure()
plt.title("Loss vs sampled points (index 0 is optimum)")
plt.ylabel("Loss value")
plt.xlabel("Sampled points")
plt.plot(losslist)

print("Loss at optimum",losslist[0])
print("Mean Loss",np.mean(losslist))
print("Min Loss",min(losslist))
print("Max Loss",max(losslist))