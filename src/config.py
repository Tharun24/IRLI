class train_config:
    datasets = {'glove':{'N':1183514,'d':100},
                'Sift-128':{'N':1000000, 'd':128}
                }
    dataset_name = 'glove'
    inp_dim = datasets[dataset_name]['d']
    n_classes = datasets[dataset_name]['N']
    ####
    n_cores = 1 # core count for TF REcord data loader
    B = 3000
    R = 16
    gpus = [4,5,6,7]
    num_gpus = len(gpus)
    batch_size = 256
    hidden_dim = 1024
    ####
    train_data_loc = '../../LTH/data/'+dataset_name+'/'
    tfrecord_loc = '../../LTH/data/'+dataset_name+'/tfrecords/'
    model_save_loc = '../saved_models/'+dataset_name+'/b_'+str(B)+'/'
    lookups_loc = '../lookups/'+dataset_name+'/b_'+str(B)+'/'
    logfolder = '../logs/'+dataset_name+'/b_'+str(B)+'/'
    # Only used if training multiple repetitions from the same script
    R_per_gpu = 2

