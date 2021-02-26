class train_config:
    feat_dim =  512
    n_classes = 670091 
    ####
    n_cores = 1 # core count for TF REcord data loader
    B = 3000
    batch_size = 1024
    feat_hash_dim = 512
    hidden_dim = 1024
    ####
    train_data_loc = '../data/amz-670k/'
    tfrecord_loc = '../data/amz-670k/tfrecords/'
    model_save_loc = '../saved_models/amz-670k/b_'+str(B)+'/'
    query_lookups_loc = '../lookups/amz-670k/b_'+str(feat_hash_dim)+'/'
    lookups_loc = '../lookups/amz-670k/b_'+str(B)+'/'
    logfile = '../logs/amz-670k/b_'+str(B)+'/'
    # Only used if training multiple repetitions from the same script
    R_per_gpu = 2

class eval_config:
    feat_dim = 512
    n_classes = 670091
    ###
    B = 3000
    R = 32
    eval_epoch = 30
    R_per_gpu = 16
    num_gpus = 2 # R/R_per_gpu gpus
    n_cores = 32 # core count for parallelizable operations
    batch_size = 320
    feat_hash_dim = 512
    hidden_dim = 1024
    ###
    query_lookups_loc = '../lookups/amz-670k/b_'+str(feat_hash_dim)+'/'
    lookups_loc = '../lookups/amz-670k/b_'+str(B)+'/epoch_'+str(eval_epoch-5)+'/'
    model_loc = '../saved_models/amz-670k/b_'+str(B)+'/'
    eval_data_loc = '../data/amz-670k/'
    tfrecord_loc = '../data/amz-670k/tfrecords/'
    ### only used by approx_eval.py (ignore if you are using evaluate.py)
    topk = 100 # how many top buckets to take
    minfreq = 4 # min number of times a class
    logfile = '../logs/amz-670k/b_'+str(B)+'/R_'+str(R)+'_topk_'+str(topk)+'_mf_'+str(minfreq)+'_epc_'+str(eval_epoch)+'.txt'  

