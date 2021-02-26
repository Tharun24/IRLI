
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
from Net import * 
from lazy_parser import MultiLabelDataset
from config import train_config as config
from config import eval_config as test_config 
from my_parser import myFastTensorDataLoader
from utils import *
import h5py
import pdb
import os

# for tensorboard
# import torch
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# from livelossplot import PlotLosses

#args
parser = argparse.ArgumentParser()
parser.add_argument("--repetition", help="which repetition?", default=0)
# parser.add_argument("--gpu", default=0)
parser.add_argument("--gpu_usage", default=0.45)
parser.add_argument("--m", default=49)
parser.add_argument("--eph", default=3.5)
parser.add_argument("--eval", default="recallRfast")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print (device)

r = int(args.repetition) # which repetition

# dataloader
# test_dataset = MultiLabelDataset(test_config.test_data_loc, "test")
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_config.batch_size, pin_memory=True, num_workers=6, shuffle=False)

test_loader = myFastTensorDataLoader(test_config.test_data_loc, "test", batch_size=test_config.batch_size, shuffle=False )

# model = Net3()
model = getmodelTest()

# logging.basicConfig(filename = config.logfile+'logs', level=logging.INFO)
total_time = 0
time_diff = 0

# query_lookup = np.load(config.query_lookups_loc+'bucket_order_'+str(r)+'.npy')

# load learned inverted index and counts
Recall = []
SetSize = []
eph = float(args.eph)
print ("eph: ", eph)

test_config.label_buckets = np.empty([test_config.R, test_config.N])
test_config.counts = np.empty([test_config.R, test_config.B+1])
test_config.bucket_order = np.empty([test_config.R, test_config.N])

# for r in range(0,test_config.R):
r=0
for r in range(0,test_config.R):
    # print(test_config.label_buckets_learned+'class_order_'+str(r)+'_'+str(eph)+'.npy')
    if os.path.exists(test_config.label_buckets_learned+'class_order_'+str(r)+'_'+str(eph)+'.npy'):
        test_config.label_buckets[r] = np.load(test_config.label_buckets_learned+'class_order_'+str(r)+'_'+str(eph)+'.npy')
    else:
        print ('learned label_buckets not loaded')
        print (test_config.label_buckets_learned+'class_order_'+str(r)+'_'+str(eph)+'.npy')
    if os.path.exists(test_config.label_buckets_learned+'counts_'+str(r)+'_'+str(eph)+'.npy'):
        test_config.counts[r] = np.load(test_config.label_buckets_learned+'counts_'+str(r)+'_'+str(eph)+'.npy')
    else:
        print ('learned counts not loaded')
    if os.path.exists(test_config.label_buckets_learned+'bucket_order_'+str(r)+'_'+str(eph)+'.npy'):
        test_config.bucket_order[r] = np.load(test_config.label_buckets_learned+'bucket_order_'+str(r)+'_'+str(eph)+'.npy')
    else:
        print ('learned bucket_order not loaded')
    
test_config.label_buckets = test_config.label_buckets.astype(int)
test_config.counts = test_config.counts.astype(int)
test_config.bucket_order = test_config.bucket_order.astype(int)

m = int(args.m)

print ("m: ", m)
bestbin = np.empty([test_config.R, test_config.batch_size, m])
bestbin_score = np.empty([test_config.R, test_config.batch_size, m])
print ("starting testing with "+ args.eval+ " eval")

for test_config.R in range(16,20):
    m = int((test_config.R/3)**2)
    m = 49
    print ("m: ", m)
    test_loader.y_req = False
    test_loader.shuffle = False
    for batch_idx, (x, y, idx) in enumerate(test_loader):
        begin_time = time.time()
        # y is true 100 labels
        x = x.to(device)
        print (x.shape)
        y = y.to(device)
        print (y.shape)
        for r in range(0,test_config.R):
            
            PATH = test_config.model_save_loc+'_'+str(r)+'_epoch_'+str(eph)+'.pth'
            # test_config.datapath = "../../../data3/GloVe100" + str(test_config.B) + "_"+ str(r) + "_"+ str(test_config.hidden_dim)+ "_lb"+ str(test_config.k)
            # PATH = test_config.datapath+ '/saved_models'+'/b_'+str(test_config.B)+'/'+'_epoch_'+str(eph)+'.pth'
            model.load_state_dict(torch.load(PATH))
            # time.sleep(40)
            logits = model(x.float())
            # time.sleep(40)
            probop = F.softmax(logits, dim=1)
            # probop = logits
            # print (min(probop[1,:]), max(probop[1,:]))
            
            bestbin[r] = torch.topk(probop, int(args.m))[1].cpu().detach().numpy()
            bestbin_score[r] = torch.topk(probop, int(args.m))[0].cpu().detach().numpy()
            # print (bestbin_score[r,1,:], bestbin[r,1,:])

        time_diff = time.time()-begin_time
        total_time += time_diff
        print ("Inf time: ", time_diff)
            
        if args.eval =="recallR":
            [score,setsize] = recallR(bestbin.astype(int), bestbin_score.astype(float), x, y, 10,m, metric = "cosine")
        if args.eval =="recallRfast":
            [score,setsize] = recallRfast(bestbin.astype(int), bestbin_score.astype(float), x, y, 10,m, metric = "cosine")
        if args.eval =="recallRcuda":
            [score,setsize] = recallRcuda(bestbin.astype(int), bestbin_score.astype(float), x, y, 10,m, metric = "cosine")
        else:
            print ("mention recallR, recallRfast, recallRcuda")
            break
        break

    SetSize.append(setsize)
    Recall.append(score)
print (Recall, SetSize)
# pdb.set_trace()
print ('Total Recall: ', Recall)

