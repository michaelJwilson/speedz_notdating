import os
import glob
import argparse
import numpy as np

from   astropy.io import ascii
from   astropy.table import Table


np.random.seed(42)

root_dir=os.environ['CSCRATCH'] + '/speedz_notdating/'

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', default=root_dir, type=str)
parser.add_argument('--maxround', type=int, default=10)
parser.add_argument('--specperround', type=np.int, default=100)

args = parser.parse_args()

original = Table.read(args.rootdir + 'prospects/targetids.csv', names=['TARGETID'])
rounds = np.arange(args.maxround)

print('\n\nOriginal.\n\n')
print(original)

print('\n\nSolving.\n\n')

for rround in rounds:
    outdir = args.rootdir + '/prospects/{:d}/'.format(rround)

    sample = np.random.choice(original['TARGETID'].data, size=len(original), replace=False)
    sample = sample.reshape(len(original), 1)

    sample = Table(data=sample, names=['# TARGETID'])
    sample = sample[:args.specperround]

    print('Populating {}.'.format(outdir + '/targetids.csv'))
    
    ascii.write(sample, outdir + '/targetids.csv', format='csv', fast_writer=False, overwrite=True)  

print('\n\nDone.\n\n')
