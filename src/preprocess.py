from utils import create_tfrecords_ann, create_universal_lookups
import glob
import time
from multiprocessing import Pool
from config import train_config as config
import os


if not os.path.isdir(config.lookups_loc):
    os.makedirs(config.lookups_loc+'epoch_0')

if not os.path.isdir(config.model_save_loc):
    os.makedirs(config.model_save_loc)

if not os.path.isdir(config.logfolder):
    os.makedirs(config.logfolder)

######## Create TF Records ##########
begin_time = time.time()
files = glob.glob(config.train_data_loc+'*.txt')
for file in files:
    nothing = create_tfrecords_ann(file)

print('elapsed_time:', time.time()-begin_time)

########## Prepare Label lookups (for MACH grouping)

begin_time = time.time()
p = Pool(32)
p.map(create_universal_lookups, list(range(32)))
p.close()
p.join()
print('elapsed_time:', time.time()-begin_time)

