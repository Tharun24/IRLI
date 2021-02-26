# author -gaurav
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import logging
import os
from Net import *
from my_parser import myFastTensorDataLoader
from config import train_config as config
from utils import *
import h5py
import pdb
import matplotlib.pyplot as plt

#for tensorboard
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(comment= "B"+str(config.B) + "_th"+ str(config.th) + "_hdsize"+ str(config.hidden_dim)+ "_lb"+ str(config.k))


#args
parser = argparse.ArgumentParser()
parser.add_argument("--repetition", help="which repetition?", default=0)
parser.add_argument("--gpu", default=0)
parser.add_argument("--gpu_usage", default=0.45)
args = parser.parse_args()

SAVE = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print (device)
trstart = time.time()

for r in range(0,config.R):
    if os.path.exists(config.label_buckets_loc+ 'class_order_'+str(r)+'_'+'.npy'):
        config.label_buckets[r] = torch.from_numpy(np.load(config.label_buckets_loc+ 'class_order_'+str(r)+'_'+'.npy'))
        config.label_buckets[r] = config.label_buckets[r].to(device).long()
    else:
        print (' label_buckets not loaded')
    if os.path.exists(config.label_buckets_loc+'counts_'+str(r)+'_'+'.npy'):
        config.counts[r] = torch.from_numpy(np.load(config.label_buckets_loc+'counts_'+str(r)+'_'+'.npy'))
        config.counts[r] = config.counts[r].to(device).long()
    else:
        print (' counts not loaded')
    if os.path.exists(config.label_buckets_loc+'bucket_order_'+str(r)+'_'+'.npy'):
        config.bucket_order[r] = torch.from_numpy(np.load(config.label_buckets_loc+'bucket_order_'+str(r)+'_'+'.npy'))
        config.bucket_order[r] = config.bucket_order[r].to(device).long()
    else:
        print (' bucket_order not loaded')
        

train_loader = myFastTensorDataLoader(config.train_data_loc, "train", batch_size=config.batch_size, shuffle=True )
torch.save(train_loader.mean, config.meanpath) # saving mean of the points

#get R models
models,criterion,optimizers = getmodel()
loss = {}
logits = {}

# logging.basicConfig(filename = config.logfile+'logs', level=logging.INFO)
total_time = 0
time_diff = 0

# uncomment below to load a trained model
# PATH = '/home/ubuntu/data/batchsz1000/saved_models/b_2000/'+'_epoch_'+str(10)+'.pth'
# model.load_state_dict(torch.load(PATH))

#to check if variabel is in GPU do: "x.is_cuda" it returns true/false
print ("starting training")
train_idx=0

for epoch in range(1, config.n_epochs+1):  # loop over the dataset multiple times
    trlearn = time.time()
    if SAVE:
        f1 = [open(config.logfile+'log_loss_'+ str(re), "a") for re in range(config.R)]
        f2 = [open(config.logfile+'log_reassign_'+ str(re), "a") for re in range(config.R)]
    running_loss = 0.0
    # train_dataset.RamdomNumbers = getRandomIndex(config.N, config.batch_size, config.B, config.counts.clone())#todo var in lazy parser shd be in torch
    
    #step1: learn scores 
    train_loader.y_req = True
    train_loader.shuffle = True
    for batch_idx, (x, y, idx) in enumerate(train_loader):
        x = x.to(device)
        
        if batch_idx == int(config.N/config.batch_size) -1 : # last batch or 1182 for glove
            break
        # training all reps
        for r in range(config.R):
            logits[r] = models[r](x.float())
            y[r] = y[r].to(device)
            loss[r] = criterion[r](logits[r], y[r].float())
            loss[r].backward() 
            optimizers[r].step()
            if SAVE:
                f1[r].write(str(epoch)+','+ str(batch_idx)+','+str(loss[r].item())+'\n')  
        print("epoche: ", epoch,"batch_idx: ", batch_idx,"loss: ", loss[r].item())    
        
        train_idx+=1
        # writer.add_figure('matplotlib', fig)
        # writer.add_scalar("Loss/train", loss, train_idx)
        # writer.add_scalar("Loss/scoreVar", np.var(prob[0]), train_idx)
        # writer.add_scalar("Loss/max-min", np.max(prob[0])- np.min(prob[0]), train_idx)

    unchanged = np.zeros(config.R)
    
    if epoch%1==0 and SAVE:
        for r in range(config.R):
            PATH = config.model_save_loc+'_'+str(r)+'_epoch_'+str(epoch-0.5)+'.pth'
            torch.save(models[r].state_dict(), PATH)
            # f1[r].write("time taken ="+','+ str(time.time() -trlearn))

    trlearn = time.time()
    #step2: reassign
    train_loader.y_req = False
    train_loader.shuffle = False
    for batch_idx, (x, y, idx) in enumerate(train_loader):
        x = x.to(device)
        # y = y.to(device)
        if batch_idx == int(config.N/config.batch_size) -1: # or 1182 for glove
            break
        begin_time = time.time()
        # Topk = []
        for r in range(config.R):
            logits[r] = models[r](x.float())
            prob = F.softmax(logits[r])
            #get bins with max scores
            #values, indices = torch.max(prob, 0)
            topk = torch.topk(prob, config.k)
            unchanged[r] = unchanged[r] + reassignR(topk[1], idx, r) # returns number of unchanged and make reassignment in fwd indx
            if SAVE:
                f2[r].write(str(epoch)+','+ str(batch_idx)+','+str(unchanged[r])+','+str((config.counts[r]>0).sum())+'\n') # str(var[r].item())
        print (' non zero buckets counts: ', (config.counts[r]>0).sum())
        time_diff = time.time()-begin_time
        
    # config.label_buckets reassign
    # te = time.time()
    for r in range(config.R):
        config.label_buckets[r] = reassignInvIdxR(r)
    # print ("time to reassign: ", time.time()-te)

    unchanged = unchanged/(config.N/config.batch_size)
    print ("unchanged:", unchanged)
    
    if epoch%1==0 and SAVE:
        for r in range(config.R):
            np.save(config.label_buckets_learned+'class_order_'+str(r)+'_'+str(epoch-0.5)+'.npy', config.label_buckets[r].cpu().detach().numpy())
            np.save(config.label_buckets_learned+'counts_'+str(r)+'_'+str(epoch-0.5)+'.npy', config.counts[r].cpu().detach().numpy())
            np.save(config.label_buckets_learned+'bucket_order_'+str(r)+'_'+str(epoch-0.5)+'.npy', config.bucket_order[r].cpu().detach().numpy())
            # f2[r].write("time taken ="+','+ str(time.time() -trlearn))
    
    # train_dataset.RamdomNumbers = None
    x = None
    y = None
    logits = {}
    torch.cuda.empty_cache() # is this useful?

    for r in range(0,config.R):
        config.label_buckets[r] = config.label_buckets[r].to(device).long()
        config.counts[r] = config.counts[r].to(device).long()
        config.bucket_order[r] = config.bucket_order[r].to(device).long()

# f1.write("training time for " +str(r) +" reps ="+ str(time.time() -trstart))
# f2.write("training time for " +str(r) +" reps ="+ str(time.time() -trstart))
if SAVE:
    for r in range(config.R):
        f1[r].close()
        f2[r].close()
print ("total training time for " +str(config.R) +" reps ="+ str(time.time() -trstart)) 
print ("=================================================================================================")