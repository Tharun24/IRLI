from utils import create_tfrecords, create_universal_lookups
import glob
import time
from multiprocessing import Pool
from config import train_config as config
import os

os.system('cd util; make; cd ..')

if not os.path.isdir(config.lookups_loc):
    os.makedirs(config.lookups_loc+'epoch_0')

if not os.path.isdir(config.model_save_loc):
    os.makedirs(config.model_save_loc)

if not os.path.isdir(config.logfolder):
    os.makedirs(config.logfolder)

######## Create TF Records ##########
begin_time = time.time()
nothing = create_tfrecords(config.train_data_loc+'train.txt')
nothing = create_tfrecords(config.train_data_loc+'test.txt')

print('elapsed_time:', time.time()-begin_time)

########## Prepare Label lookups (for MACH grouping)

begin_time = time.time()
p = Pool(config.R)
p.map(create_universal_lookups, list(range(config.R)))
p.close()
p.join()
print('elapsed_time:', time.time()-begin_time)

