
import h5py
import numpy as np
import os
import torch

# # not in use
# def getmetaData(filename, purpose):
#     dataset = h5py.File(filename, 'r')
#     N = len(dataset[purpose]) # Total_Points in train set 
#     D = len(dataset[purpose][0]) # Num_Features
#     return [N,D]

class train_config:
    DATASET = {'GloVe100': {'name':"/glove-100-angular.hdf5",'N':1183514, 'd':100 },
                'GloVe200': {'name':"/glove-200-angular.hdf5",'N':1183514, 'd':200 },
                'nytimes': {'name':"/nytimes-256-angular.hdf5",'N':290000, 'd':256 },
                'yfcc1M': {'name':"/YFCC0.hdf5",'N':1000000, 'd':4096 },
                'deep1B': {'name':"/deep-image-96-angular.hdf5",'N':9990000, 'd':96 },
                'sift': {'name':"/sift-128-euclidean.hdf5",'N':1000000, 'd':128 }
                }
    datasetName = 'GloVe100'
    B = 5000 # number of labels, last layer size
    hidden_dim = np.array([1000])
    k = 3 # get top k scores for load balance

    r = 0
    R = 20
    batch_size = 1000
    n_epochs = 4
    load_epoch = 0
    # reg = "_batchNorm"

    print ("Rep: ", r)
    metric = "cosine"
    sep_net = False
    hfP =False
    trainAllR = True
    shared = False
    assignment="srp" # or srp or l2hash or random

    if hfP:
        datapath = datapath+"_halfPrecision"

    datapath = "../../../data3/"+datasetName+"_" + str(B) + "_"+ np.array2string(hidden_dim, separator='-')+ "_lb"+ str(k)  

    train_data_loc = "../../../data3" + DATASET[datasetName]['name']
    meanpath = datapath +"/mean.pt"

    [N, feat_dim] = [DATASET[datasetName]['N'],DATASET[datasetName]['d']]

    model_save_loc = datapath+ '/saved_models'+'/'
    # initial binnings
    label_buckets_loc = datapath+ '/label_buckets'+'/b_'+str(B)+'/'+str(N)
    # to save learned binning
    label_buckets_learned = datapath+ '/label_buckets_learned'+'/b_'+str(B)+'/'+str(N)
    # logs
    logfile = datapath+ '/logs'+'/b_'+str(B)+'/'+str(N)

    label_buckets = {}
    counts = {}
    bucket_order = {}

    datasetTrain = torch.from_numpy(np.array(h5py.File(train_data_loc, 'r')['train']))

class eval_config:
    DATASET = {'GloVe100': {'name':"/glove-100-angular.hdf5",'N':1183514, 'd':100 },
                'GloVe200': {'name':"/glove-200-angular.hdf5",'N':1183514, 'd':200 },
                'nytimes': {'name':"/nytimes-256-angular.hdf5",'N':290000, 'd':256 },
                'yfcc1M': {'name':"/YFCC0.hdf5",'N':1000000, 'd':4096 },
                'deep1B': {'name':"/deep-image-96-angular.hdf5",'N':9990000, 'd':96 },
                'sift': {'name':"/sift-128-euclidean.hdf5",'N':1000000, 'd':128 }
                }
    
    datasetName = 'GloVe100'
    B = 5000 # number of labels, last layer size
    hidden_dim = np.array([1000])
    k = 3 # get top k scores for load balanc

    R = 20
    # print ("R: ", R)
    
    # sep_net = True
    hfP =False
    # reg = "_batchNorm"
    shared = False
    assignment="srp" # or srp or l2hash
    datapath = "../../../data3/"+datasetName+"_" + str(B) + "_"+ "1000"+ "_lb"+ str(k)  
    # datapath = "../../../data3/GloVe100_" + str(B) + "_"+ np.array2string(hidden_dim, separator='-')+ "_lb"+ str(k) + "_jointLayer"

    if hfP:
        datapath = datapath+"_halfPrecision"
    
    test_data_loc = "../../../data3" + DATASET[datasetName]['name']
    meanpath = datapath +"/mean.pt"
    print (meanpath)
    metric = "cosine"
    if os.path.exists(meanpath):
        mean = torch.load(meanpath)
        print ("mean exists")
    
    batch_size = 10000 # number of queries
    eval_epoch = 1
    [N, feat_dim] = [DATASET[datasetName]['N'],DATASET[datasetName]['d']]
    model_save_loc = datapath+ '/saved_models'+'/'
    label_buckets_learned = datapath+ '/label_buckets_learned'+'/b_'+str(B)+'/'+str(N)
    
    label_buckets = {}
    counts = {}
    bucket_order = {}
    minfreq = 1

    #global data load
    # datasetTrain = torch.from_numpy(np.array(h5py.File(test_data_loc, 'r')['train']))


