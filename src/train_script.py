import subprocess
from config import train_config as config

for r in range(config.R):
    command = 'export PYTHONPATH=util/; python3 train.py --repetition='+str(r)+' --gpu='+str(config.gpus[r//config.num_gpus])+' --load_epoch=0 --n_epochs=30>../logs/'+config.dataset_name+'/b_'+str(config.B)+'/terminal_log_'+str(r)+'.txt'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    # (out, err) = process.communicate()
    # wgetLink = str(out).split(',')[0][10:]
    # wgetCommand = 'wget ' + wgetLink + ' -O learn_' + str(i).zfill(2)
    # print "Downloading learn chunk " + str(i).zfill(2) + ' ...'
    # process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
    # process.stdin.write('e')
    # process.wait()