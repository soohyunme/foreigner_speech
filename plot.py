import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
 
## relating to loading and saving
parser = argparse.ArgumentParser(description='save plot')
parser.add_argument('--path',       type=str, default='',       help='load log file')
parser.add_argument('--save_path',  type=str, default='graph',  help='location to save png')
args = parser.parse_args()

## log file path check
if args.path == '':
    print('ERROR : File not found')
    quit() 

## load data from log file
with open(args.path, 'r') as f:
    data = f.readlines()[1:]
    
tloss_list = []
vloss_list = []
for row in data:
    _, _, _, _, sloss, _, _, vloss = row.replace(',', '').split()
    tloss_list.append(float(sloss))
    vloss_list.append(float(vloss))

## plot the graph
x_len = np.arange(len(tloss_list))
plt.plot(x_len, tloss_list, c='red', label="Train-set Loss")
plt.plot(x_len, vloss_list, c='blue', label="Validation-set Loss")
plt.legend(loc='upper right')
plt.grid()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

## Save plot to png file
os.makedirs(args.save_path,exist_ok=True)
plt.savefig('{}.png'.format(os.path.join(args.save_path,args.path.split('/')[-2])))