import h5py
import numpy as np
from multiprocessing import Pool
from sklearn.utils import murmurhash3_32 as mmh3
import random
import sys
from os import path
import pickle
from config import train_config as config
from config import eval_config as test_config 
import pdb
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from scoreAgg import scoreAgg
from Net import *
# from pytorch_memlab import LineProfiler

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print (device)

# some functions taken from ann-benchmarks
#https://github.com/erikbern/ann-benchmarks/blob/55b99509887b606ccb799123b8e0d3a650f61271/ann_benchmarks/datasets.py#L26

class Lsh(object):
    """ Interface for locality-sensitive hashes. """
    def __init__(self):
        # Dimension of our vector space
        dimension = config.feat_dim
        # Create a random binary hash with 10 bits
        bits = int(np.ceil(np.log(config.B)/np.log(2)))
        self.rbp = RandomBinaryProjections('rbp', bits)
        self.engine = Engine(dimension, lshashes=[self.rbp])
        
    def hash_vector(self,vec):
        return int(self.rbp.hash_vector(vec)[0],2)
    

def getRandomIndex(N, batch_size, B, counts):
    cumCount = torch.cumsum(counts, dim=0)
    RamdomNumbers= torch.randint(N, (N,B))
    # RamdomNumbers.is_cuda
    #assert
#     print("RamdomNumbers.is_cuda, counts.is_cuda()", RamdomNumbers.is_cuda, counts.is_cuda)
    # beware! counts can have 0 values. making it temp 1
    counts[1:][counts[1:]==0] =1 
    E = RamdomNumbers%counts[1:] 
    L =(cumCount==N).nonzero()[0]
    E[:,1:L]= cumCount[1:L]+ E[:,1:L] # random point index from each bucket

#     config.label_buckets = config.label_buckets.repeat(N,1) # million times million !!!!!!!!!! change this
#     print (config.label_buckets.size(), E.size())
#     print (config.label_buckets, E)
#     print (sum(sum(E>=10000)))
    for i in range(0,N):
        E[i,:] = torch.gather(config.label_buckets, 0, E[i,:]) # random index from each bucket to actual data point index
    return E.type(torch.short) # 16 bit unsigned int

def get_Ys(N, batch_size, batch, data, B, count, RamdomNumbers):
    # t1 = time.time()
    # RamdomNumbers= torch.randint(N, (config.R, batch_size,B))

    Y = torch.empty(batch_size,B)

    assert torch.sum(count)==config.N
    cumCount = torch.cumsum(count, dim=0)
    # beware! counts can have 0 values. making it temp 1
    cnt_zero_idx = count[1:]==0
    count[1:][cnt_zero_idx] =1 

    E = RamdomNumbers%count[1:] # index will be 0 for bucket with 1 and 0 elements
        
    try:
        L =(cumCount>=N).nonzero()[0] # first element to hit the N mark, we have to stop there
    except IndexError:
        L = cumCount.shape[0]-1
        print (cumCount)
    E[:,1:L]= cumCount[1:L]+ E[:,1:L] # random point index from each bucket, gets pt from next bucket if empty
    #E = E.long()
    for i in range(0,batch_size):
        E[i,:] = torch.gather(config.label_buckets, 0, E[i,:]) # random index from each bucket to actual data point index
    # E = E.type(torch.short) # 16 bit unsigned int
    E = E.long()
    for i,xi in enumerate(data[E]):
        Y[i] = softlabels(batch[i], xi, metric = config.metric)

    # E = torch.reshape(E,(-1,))
    # Y[r] = torch.reshape(softlabelsBatchwise(data[E], batch.repeat(1,B).view(-1,batch.shape[1]),metric = config.metric), (batch_size, B))
        
    count[1:][cnt_zero_idx] =0
    #Y[r,:,cnt_zero_idx] = 0.5/config.B
       
    return(Y)


def get_R_Ys(N, batch_size, batch, B, count, RamdomNumbers, r):
    Y = torch.empty(batch_size,B)
    assert torch.sum(count)==config.N
    cumCount = torch.cumsum(count, dim=0)
    # beware! counts can have 0 values. making it temp 1
    cnt_zero_idx = count[1:]==0
    count[1:][cnt_zero_idx] =1 

    E = RamdomNumbers%count[1:] # index will be 0 for bucket with 1 and 0 elements
        
    try:
        L =(cumCount>=N).nonzero()[0] # first element to hit the N mark, we have to stop there
    except IndexError:
        L = cumCount.shape[0]-1
        print (cumCount)
    E[:,1:L]= cumCount[1:L]+ E[:,1:L] # random point index from each bucket, gets pt from next bucket if empty
    #E = E.long()
    for i in range(0,batch_size):
        E[i,:] = torch.gather(config.label_buckets[r], 0, E[i,:]) # random index from each bucket to actual data point index
    # E = E.type(torch.short) # 16 bit unsigned int
    E = E.long()
    # print (config.datasetTrain.is_cuda)
    # time.sleep(20)
    # for i,xi in enumerate(config.datasetTrain[E]): #config.datasetTrain[E] takes 1000*5000 = 5M i.e. 5 times the original data
    for i in range(0,batch_size):
        xi = config.datasetTrain[E[i]]
        Y[i] = softlabels(batch[i], xi, metric = config.metric)

    count[1:][cnt_zero_idx] =0
    return(Y)

def hardlabels(x, xi, th, metric = "cosine"):
    if metric == "l2":
        y = 1*(torch.sum((x- xi)**2, axis =1)**0.5 <th)
    if metric == "l1":
        y = 1*(torch.sum((abs(x- xi)), axis =1) <th)
    if metric == "cosine":
        y = 1*((1-torch.sum(xi*x, axis=1)/(torch.norm(xi,dim=1)*torch.norm(x))) <th)

    #correctin labels for empty bins
    y[config.counts[1:]==0] = 0
    return y

def softlabels(x, xi, metric = "cosine"):
    if metric == "cosine":
        y = torch.sum(xi*x, axis=1)/(torch.norm(xi,dim=1)*torch.norm(x))
#         y = F.softmax(y)
        y = torch.add(y, 1)/(2*config.B)
#         print (max(y), min(y))
        return y

def softlabelsBatchwise(a1, a2, metric = "cosine"):
    if metric == "cosine":
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        y = cos(a1, a2)
        y = torch.add(y, 1)/(2*config.B)
        return y

def create_universal_lookups(r):
    #read data
    # dataset = h5py.File(config.train_data_loc, 'r') 
    dataset = getData('train')
    # dataset = np.array(dataset['train'])[0:config.N,:]
    mean = np.mean(dataset, axis=0)
    dataset = dataset-mean
    
    counts = np.zeros(config.B+1, dtype=int)
    bucket_order = np.zeros(config.N, dtype=int)
    
#     v = np.random.randn(128)
#     print (type(dataset[0].astype(np.float64)[0]), type(v[0]))
    
    if config.assignment=="random":
        bucket_order = np.random.randint(config.B, size=config.N)
        for i in range(config.N):
            counts[bucket_order[i]+1] += 1
    elif config.assignment=="srp":
        mylsh = Lsh()
        for i in range(config.N):
            bucket = mylsh.hash_vector(dataset[i])%config.B # get feature vector and replace with LSH function
            bucket_order[i] = bucket
            counts[bucket+1] += 1
    elif config.assignment=="l2hash":
        print ("L2 not coded yet, mention either random or srp")
    else:
        print ("Error: provide either random, srp or l2hash assignment!")

        
    cumCounts = np.cumsum(counts)
    rolling_counts = np.zeros(config.B, dtype=int)
    class_order = np.zeros(config.N,dtype=int)
    for i in range(config.N):
        temp = bucket_order[i]
        class_order[cumCounts[temp]+rolling_counts[temp]] = i
        rolling_counts[temp] += 1
    np.save(config.label_buckets_loc+'class_order_'+str(r)+'_'+'.npy', class_order)
    np.save(config.label_buckets_loc+'counts_'+str(r)+'_'+'.npy',counts)
    np.save(config.label_buckets_loc+'bucket_order_'+str(r)+'_'+'.npy', bucket_order)

#optimise it further
def reassign(topk, q_idx):
    cnt =0
    for i in range(len(q_idx)):
        prevbin = config.bucket_order[q_idx[i]]
#         newbin = topk[i][0]
        newbinidx = torch.argmin(config.counts[torch.add(topk[i], 1)]) #coz count is shifted by 1
        newbin = topk[i][newbinidx]
        if prevbin == newbin:
            cnt+=1
        else:
            config.counts[prevbin+1]-=1
            config.counts[newbin+1]+=1
            config.bucket_order[q_idx[i]]=newbin #without load balance, topk[i][0].long(): bucket number of top score, q_idx: index of query data point
#     print ("total unchanged: ", cnt, "out of: ", config.batch_size)
    return cnt
    
#optimise it further
def reassignR(topk, q_idx, r):
    cnt =0
    for i in range(len(q_idx)):
        prevbin = config.bucket_order[r][q_idx[i]]
#         newbin = topk[i][0]
        newbinidx = torch.argmin(config.counts[r][torch.add(topk[i], 1)]) #coz count is shifted by 1
        newbin = topk[i][newbinidx]
        if prevbin == newbin:
            cnt+=1
        else:
            config.counts[r][prevbin+1]-=1
            config.counts[r][newbin+1]+=1
            config.bucket_order[r][q_idx[i]]=newbin #without load balance, topk[i][0].long(): bucket number of top score, q_idx: index of query data point
#     print ("total unchanged: ", cnt, "out of: ", config.batch_size)
    return cnt
    

def reassignInvIdx():
    cumCounts = torch.cumsum(config.counts, dim=0).cpu().detach().numpy()
    rolling_counts = np.zeros(config.B, dtype=int)
    label_buckets = np.zeros(config.N, dtype=int)
    bucket_order = config.bucket_order.cpu().detach().numpy()
    for i in range(config.N):
        temp = bucket_order[i]
        label_buckets[cumCounts[temp]+rolling_counts[temp]] = i
        rolling_counts[temp] += 1
    return torch.from_numpy(label_buckets).to(device).long()

def reassignInvIdxR(r):
    cumCounts = torch.cumsum(config.counts[r], dim=0).cpu().detach().numpy()
    rolling_counts = np.zeros(config.B, dtype=int)
    label_buckets = np.zeros(config.N, dtype=int)
    bucket_order = config.bucket_order[r].cpu().detach().numpy()
    for i in range(config.N):
        temp = bucket_order[i]
        label_buckets[cumCounts[temp]+rolling_counts[temp]] = i
        rolling_counts[temp] += 1
    return torch.from_numpy(label_buckets).to(device).long()

def recall(bestbins, x, y, M, metric = "l2"):
    t1 = time.time()
    # load data first
    # dataset = np.array(h5py.File(test_config.test_data_loc, 'r')['train'])
    dataset = getData('train')
    N = len(dataset) # Total_Points in train set 
#     bestbins = bestbins.long()
    # batch size many bestbins
    cumCounts = torch.cumsum(test_config.counts, dim=0).long()
    
    # y slice
    y = y[:,:M] # confirm they are ranked already
    
    print (len(bestbins), test_config.batch_size)
    cnt = 0
    setsize = 0
    
    topMcandTime = 0
    retrivalTime = 0
    accessTime = 0
    recallComp = 0
    
    x = x.cpu().detach()
    t2 = time.time()
    print ("loading and moving: ", t2-t1)
    
    for i in range(test_config.batch_size):
        t1 = time.time()
        bestbin = bestbins[i]
        a = torch.gather(cumCounts, 0, bestbin)
        b = torch.gather(test_config.counts, 0, torch.add(bestbin, 1))
        
#         a = cumCounts[bestbin[0]]
        t3 = time.time()
    
        candidates  = test_config.label_buckets[a[0] : a[0] +b[0]]
        # iterate for m times
        for k in range(1, len(bestbin)):
            candidates  = torch.cat((candidates, test_config.label_buckets[a[k] : a[k] +b[k]]), 0) #to speedup

        setsize+=len(candidates)
        t2 = time.time()
        retrivalTime = retrivalTime + t2-t1
        accessTime = accessTime + t3-t1
        t1 = time.time()
        pdb.set_trace()
        print (torch.Size(candidates))
        try:
            if metric == "l2":
                dist = torch.sum((x[i]- dataset[candidates])**2, axis =1)**0.5 
#                 dist2 = torch.sum((x[i].cpu().detach()- dataset[y[i].cpu().detach()])**2, axis =1)**0.5 
            if metric == "cosine":
                dist = (1-torch.sum(x[i]*dataset[candidates], axis=1)/(torch.norm(dataset[candidates],dim=1)*torch.norm(x[i])))
#                 dist2 = (1-torch.sum(x[i].cpu().detach()*dataset[y[i].cpu().detach()], axis=1)/(torch.norm(dataset[y[i].cpu().detach()],dim=1)*torch.norm(x[i].cpu().detach())))
#                 print (dist2)
            if len(dist)>M:
                #top M
#                     print (torch.topk(-dist, M))
#                     print (torch.topk(-dist2, M))
                topM = candidates[torch.topk(-dist, M)[1]]
#                 print ("top100:",top100)
            else:
                topM = candidates
        except:
            topM = candidates
        t2 = time.time()
        topMcandTime = topMcandTime + t2-t1
        
        # Create a tensor to compare all values at once, fast in GPU
        t1 = time.time()
        topM = topM.to(device)
        compareview = topM.repeat(y[i].shape[0],1).T
        # Intersection
        cnt = cnt+ (len(y[i][(compareview == y[i]).T.sum(1)==1]))
        t2 = time.time()
        recallComp = recallComp + t2-t1
           
    score = cnt/(M*test_config.batch_size)
    print ("recall: ", score, "avg setsize: ", setsize/test_config.batch_size)
    print ("retrivalTime: ", retrivalTime,"accessTime: ", accessTime, "topMcandTime: ", topMcandTime, "recallComp: ", recallComp)

    return score

#Recall@M
def recallR(bestbins, bestbin_score, x, y, M, m, metric = "l2"):
    t1 = time.time()
    # load data first
    # dataset = np.array(h5py.File(test_config.test_data_loc, 'r')['train'])
    dataset = getData('train')
    N = len(dataset) # Total_Points in train set 
    
    # y slice
    y = y[:,:M] # confirm they are ranked already
    x = x.cpu().detach()
    print (len(bestbins), test_config.batch_size)
    
    # R x batch size many bestbins
    cumCounts = np.cumsum(test_config.counts, axis=1).astype(int) # R x (B+1)

    # check for all these sizes
    # allsizes = np.array([10, 20, 50, 100, 1000, 10000, 30000])
    allsizes = np.array([10000])
    cnt = np.zeros(len(allsizes))
    setsize = np.zeros(len(allsizes))
    
    topMcandTime = 0
    retrivalTime = 0
    accessTime = 0
    recallComp = 0
    
    t2 = time.time()
    print ("loading and moving: ", t2-t1)
    r = 0
    print ("R: ", test_config.R)
    
    # freq ={} # hash map
    # for q in range(test_config.N):
    #     freq[q]=0

    # for each point in test set
    for i in range(test_config.batch_size):
        # membership = np.zeros(test_config.N, dtype=np.uint8)
        candidates = np.array([0])
        t1 = time.time()
        freq = np.zeros(test_config.N)
        for r in range(test_config.R):
            # m = int((r/3)**2)
            
            bestbin = bestbins[r,i,:]
            a = cumCounts[r,bestbin]
            
            b = test_config.counts[r, bestbin +1]
            # a = torch.gather(cumCounts[r], 0, bestbin)
            # b = torch.gather(test_config.counts[r], 0, torch.add(bestbin, 1))

    #         a = cumCounts[bestbin[0]]
            t3 = time.time()

            # candidates1rep  = test_config.label_buckets[r, a[0] : a[0] +b[0]]
            # iterate for m times
            for k in range(0, m+1):
                # candidates1rep  = np.concatenate((candidates1rep, test_config.label_buckets[r, a[k] : a[k] +b[k]]), axis=None) #to speedup
                for key in test_config.label_buckets[r, a[k] : a[k] +b[k]]:
                    freq[key]+= bestbin_score[r,i,k] # scores

            # candidates = np.array([key for key in freq if freq[key]>=test_config.minfreq])

        candidates = np.argwhere(freq>0)
        candidates = np.reshape(candidates,len(candidates)) # all non zero candidates
        freq = np.array([freq[key] for key in candidates])
        desc_idxs = np.argsort(-freq)
        # top_idxs = np.argpartition(freq, -5)[-5:]
        candidates = candidates[desc_idxs]

        # now find recall for each
        for s,size in enumerate(allsizes):
            if len(candidates)<size:
                size = len(candidates)
            candidatesfrac = candidates[:size]
            setsize[s] +=len(candidatesfrac)
            t2 = time.time()
            retrivalTime = retrivalTime + t2-t1
            accessTime = accessTime + t3-t1
            candidatesfrac = torch.from_numpy(candidatesfrac).float()
            
            t1 = time.time()
    #         pdb.set_trace()
    #         print (torch.Size(candidates))
            try:
                if metric == "l2":
                    dist = torch.sum((x[i]- dataset[candidatesfrac])**2, axis =1)**0.5 
                if metric == "cosine":
                    dist = (1-torch.sum(x[i]*dataset[candidatesfrac], axis=1)/(torch.norm(dataset[candidatesfrac],dim=1)*torch.norm(x[i])))
                if len(dist)>M:
                    topM = candidatesfrac[torch.topk(-dist, M)[1]]
                else:
                    topM = candidatesfrac
            except:
                topM = candidatesfrac
            t2 = time.time()
            topMcandTime = topMcandTime + t2-t1
            
            # Create a tensor to compare all values at once, fast in GPU
            t1 = time.time()
            topM =  topM.to(device)
            compareview = topM.repeat(y[i].shape[0],1).T
            # Intersection
            cnt[s] = cnt[s]+ (len(y[i][(compareview == y[i]).T.sum(1)==1]))
            t2 = time.time()
            recallComp = recallComp + t2-t1
           
    score = cnt/(M*test_config.batch_size)
    print ("recall: ", score, "avg setsize: ", setsize/test_config.batch_size)
    print ("retrivalTime: ", retrivalTime, "topMcandTime: ", topMcandTime, "recallComp: ", recallComp)

    return [score, setsize/test_config.batch_size]

def recallRfast(bestbins, bestbin_score, x, y, M, m, metric = "l2"):
    # dataset = np.array(h5py.File(test_config.test_data_loc, 'r')['train'])
    dataset = getData('train')
    N = config.N # Total_Points in train set 
    y = y[:,:M] # confirm they are ranked already
    x = x.cpu().detach()
    
    cumCounts = np.cumsum(test_config.counts, axis=1).astype(int) # R x (B+1)

    allsizes = np.array([10, 20, 50, 100, 200, 300, 500, 700, 1000, 3000, 5000, 10000])
    # allsizes = np.array([10000])
    cnt = np.zeros(len(allsizes))
    setsize = np.zeros(len(allsizes))
    maxsize = allsizes[-1]
    

    topMcandTime = 0
    retrivalTime = 0
    # accessTime = 0
    recallComp = 0
    
    t2 = time.time()
    r = 0

    test_config.label_buckets = np.ascontiguousarray(test_config.label_buckets)
    test_config.counts = np.ascontiguousarray(test_config.counts)
    cumCounts = np.ascontiguousarray(cumCounts)
    #bestbins = np.ascontiguousarray(bestbins)
    #bestbin_score = np.ascontiguousarray(bestbin_score)

    for i in range(test_config.batch_size):
        # print (test_config.label_buckets.dtype)
        t1 = time.time()
        # print (bestbins.dtype)
        candidates = np.zeros(maxsize).astype(int)
        candidates = np.ascontiguousarray(candidates)

        # tales all input return candidates
        #print (test_config.label_buckets.shape)
        #print (test_config.counts.shape)
        # print (bestbin_score[:,i,:])
        #print (bestbin_score[:,i,:].shape)
        # print (i)
        scoreAgg(test_config.label_buckets, test_config.counts, cumCounts, candidates, np.ascontiguousarray(bestbins[:,i,:].astype(int)), np.ascontiguousarray(bestbin_score[:,i,:].astype(float)), test_config.R, test_config.B, config.N, m, maxsize)
        # print (i)
        retrivalTime += time.time() -t1
        t1 = time.time()
        for s,size in enumerate(allsizes):
            if len(candidates)<size:
                size = len(candidates)
            candidatesfrac = candidates[:size]
            setsize[s] +=len(candidatesfrac)
            candidatesfrac = torch.from_numpy(candidatesfrac).float()

            try:
                if metric == "l2":
                    dist = torch.sum((x[i]- dataset[candidatesfrac])**2, axis =1)**0.5 
                if metric == "cosine":
                    dist = (1-torch.sum(x[i]*dataset[candidatesfrac], axis=1)/(torch.norm(dataset[candidatesfrac],dim=1)*torch.norm(x[i])))
                if len(dist)>M:
                    topM = candidatesfrac[torch.topk(-dist, M)[1]]
                else:
                    topM = candidatesfrac
            except:
                topM = candidatesfrac
            
            topMcandTime += time.time()-t1
            
            # Create a tensor to compare all values at once, fast in GPU
            t3 = time.time()
            topM =  topM.to(device)
            # now find recall for each
            compareview = topM.repeat(y[i].shape[0],1).T
            # Intersection
            cnt[s] = cnt[s]+ (len(y[i][(compareview == y[i]).T.sum(1)==1]))
            # t2 = time.time()
            recallComp = recallComp + time.time()-t1
           
    score = cnt/(M*test_config.batch_size)
    print ("recall: ", score, "avg setsize: ", setsize/test_config.batch_size)
    # print ("retrivalTime: ", retrivalTime)
    print ("retrivalTime: ", retrivalTime, "topMcandTime: ", topMcandTime, "recallComp: ", recallComp)

    return [score, setsize/test_config.batch_size]

def recallRcuda(probop, x, y, M, metric = "cosine"):
    
    # load data first
    # dataset = torch.from_numpy(np.array(h5py.File(test_config.test_data_loc, 'r')['train'])).to(device)
    dataset = getData('train')
    N = len(dataset) # Total_Points in train set 

    # y slice
    y = y[:,:M] # confirm they are ranked already
    # cumCounts = torch.cumsum(test_config.counts, dim=0).long()

    # check for all these sizes
    allsizes = np.array([10, 20, 50, 100, 1000, 10000])
    cnt = 0
    setsize = 0
    
    dist_computation = 0
    Aggregation_Time = 0
    recallComp = 0
    TopM = torch.empty(test_config.batch_size, M).to(device)
    # test_config.batch_size = 100
    start1 = torch.cuda.Event(enable_timing=True)
    end1 = torch.cuda.Event(enable_timing=True)  
    start2 = torch.cuda.Event(enable_timing=True)
    end2 = torch.cuda.Event(enable_timing=True) 
    for i in range(test_config.batch_size):
        start1.record()
        #m1
        freq = torch.zeros(test_config.N,).to(device)
        for r in range(0,test_config.R):
            freq += torch.gather(probop[r,i],0,test_config.bucket_order[r])
        candidates = torch.topk(freq, allsizes[-1], sorted=False)[1]

        # #m2
        # freq = torch.gather(probop[:,i],1,test_config.bucket_order)
        # # freq = torch.sum(freq,0)
        # candidates = torch.topk(torch.sum(freq,0), allsizes[-1], sorted=False)[1]
        end1.record()

        start2.record()
        setsize+=len(candidates)
        # t3 = time.time()
        try:
            if metric == "l2":
                dist = torch.sum((x[i]- dataset[candidates])**2, axis =1)**0.5 
            if metric == "cosine":
                dist = (1-torch.sum(x[i]*dataset[candidates], axis=1)/(torch.norm(dataset[candidates],dim=1)*torch.norm(x[i])))
            if len(dist)>M:
                topM = candidates[torch.topk(-dist, M)[1]]
            else:
                topM = candidates
        except:
            topM = candidates # check topm shd be in gpu
        # t4 = time.time()
        del dist
        # dist_computation = dist_computation + t4-t3
        # print (t4-t3)
        # Create a tensor to compare all values at once, fast in GPU
        # t1 = time.time()

        # topM =  topM.to(device)
        # cnt = cnt+ len(np.intersect1d(topM, y[i]))
        # TopM[i] = topM
        compareview = topM.repeat(y[i].shape[0],1).T
        # # Intersection
        cnt = cnt+ (len(y[i][(compareview == y[i]).T.sum(1)==1]))
        # t2 = time.time()
        # recallComp = recallComp + t2-t1
        end2.record()
        torch.cuda.synchronize()
        Aggregation_Time = Aggregation_Time + start1.elapsed_time(end1)
        # print (start1.elapsed_time(end1))
        dist_computation = dist_computation + start2.elapsed_time(end2)
    # torch.cuda.empty_cache()
    # t2 = time.time()
    # Aggregation_Time =  t2-t1
    # t5 = time.time()
    # TopM = TopM.cpu()
    # y = y.cpu()
    # for i in range(test_config.batch_size):
    #     cnt = cnt+ len(np.intersect1d(TopM[i], y[i]))
    # t6 = time.time()
    # recallComp = t6-t5

    score = cnt/(M*test_config.batch_size)
    print ("recall: ", score, "avg setsize: ", setsize/test_config.batch_size)
    print ("Aggregation_Time: ", Aggregation_Time/1000, "dist_computation: ", dist_computation/1000, "recallComp: ", recallComp)
    return [score, setsize/test_config.batch_size]
    

def getmodel():
    models = {}
    optimizers = {}
    criterion = {}
    if config.shared:
        if len(config.hidden_dim)==1:
            models[0] = Net1(config.hidden_dim) 
            for r in range(1,config.R):
                models[r] = Net1shared(config.hidden_dim,shared1 = models[0].fc1)
        elif len(config.hidden_dim)==2:
            models[0] = Net2(config.hidden_dim) 
            for r in range(1,config.R):
                models[r] = Net3shared(config.hidden_dim, shared1 = models[0].fc1, shared2 = models[0].fc2)
        elif len(config.hidden_dim)==3:
            models[0] = Net3(config.hidden_dim) 
            for r in range(1,config.R):
                models[r] = Net3shared(config.hidden_dim, shared1 = models[0].fc1, shared2 = models[0].fc2, shared3 = models[0].fc3)
        else:
            print("upto 3 hidden layer only")
    else:
        if len(config.hidden_dim)==1:
            for r in range(0,config.R):
                models[r] = Net1(config.hidden_dim)
        elif len(config.hidden_dim)==2:
            for r in range(0,config.R):
                models[r] = Net2(config.hidden_dim)
        elif len(config.hidden_dim)==3:
            for r in range(0,config.R):
                models[r] = Net3(config.hidden_dim)
        else:
            print("upto 3 hidden layer only")

    for r in range(config.R):
        criterion[r] = nn.MultiLabelSoftMarginLoss() 
        optimizers[r] = optim.Adam(models[r].parameters(),lr = 0.0001)
    # criterion = nn.CrossEntropyLoss()
    # criterion =  nn.BCEWithLogitsLoss(weight=None, reduction='none')
    # criterion = nn.KLDivLoss()
    return models,criterion,optimizers

def getmodelTest():
    if len(config.hidden_dim)==1:
        model = Net1(config.hidden_dim)
    elif len(config.hidden_dim)==2:
        model = Net2(config.hidden_dim)
    elif len(config.hidden_dim)==3:
        model = Net3(config.hidden_dim)
    else:
        print("upto 3 hidden layer only")
    return model

# not in use
def getData(purpose):
    data = h5py.File(config.train_data_loc, 'r')

    if purpose=='train':
        return np.array(data['train'])
    if purpose=='test':
        return [np.array(data['test']), np.array(data['neighbors'])]


##NOT IN USE#############################################################################################################################################
def featureHash(x_idxs, x_vals, r):
    #use config.feat_dim_orig and config.feat_hash_dim
    #use query_lookup
    query_lookup = np.load(config.query_lookups_loc+'bucket_order_'+str(r)+'.npy')
    
#     query_lookup = torch.from_numpy(query_lookup)
#     x_idxs = torch.from_numpy(x_idxs)
#     x_idxs_hashed = torch.index_select(query_lookup, 0, x_idxs)
    
    if (config.feat_hash_dim ==config.feat_dim_orig):
        x_idxs_hashed = (x_idxs)
    else:
        x_idxs_hashed = (query_lookup[x_idxs])
    x_vals_hashed = (x_vals)
    return x_idxs_hashed, x_vals_hashed

def labelHash(y_idxs, y_vals, r):
    #use config.n_classes and config.B
    if r>-1:
        #use lookup
        lookup = np.load(config.lookups_loc+'bucket_order_'+str(r)+'.npy') 
        y_idxs_hashed = ((lookup[y_idxs]))
    else:
        y_idxs_hashed = ((y_idxs))
    y_vals_hashed = (y_vals)
    return y_idxs_hashed,y_vals_hashed

def create_query_lookups(r):
    bucket_order = np.zeros(config.feat_dim_orig, dtype=int)
    #
    for i in range(config.feat_dim_orig):
        bucket = mmh3(i,seed=r)%config.feat_hash_dim
        bucket_order[i] = bucket
    np.save(config.query_lookups_loc+'bucket_order_'+str(r)+'.npy', bucket_order)
