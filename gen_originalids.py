import os
import glob
import json
import argparse
import numpy  as np
import pandas as pd

from   astropy.io import ascii
from   astropy.table import Table, join, vstack, Row


root_dir = os.environ['CSCRATCH'] + '/speedz_notdating/'

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', default=root_dir, type=str)
parser.add_argument('--maxround', default=2, type=np.int)
parser.add_argument('--specperround', default=100, type=np.int)

args = parser.parse_args()

nids = args.maxround * args.specperround

# 'merger comment', 'all VI issues', 'all VI comments'
columns     = ['TARGETID', 'Redrock z', 'best z', 'best quality', 'Redrock spectype', 'best spectype', 'N_VI', 'DELTACHI2', 'ZWARN']

truth_qsos  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_QSO_v1.2.csv')[columns]
truth_lrgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_LRG_v1.2.csv')[columns]
truth_elgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_ELG_v1.2.csv')[columns]
truth_bgs   = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_BGS_v1.2.csv')[columns]

truth       = vstack((truth_qsos, truth_lrgs, truth_elgs, truth_bgs))
truth.sort('TARGETID')

truth.pprint(max_width=-1)  

sample = np.random.choice(truth['TARGETID'].data, size=nids, replace=False)
sample = sample.reshape(len(sample), 1)

sample = Table(data=sample, names=['# TARGETID'])

ascii.write(sample, args.rootdir + '/prospects/targetids.csv', format='csv', fast_writer=False, overwrite=True)

print('Finished writing: {}'.format(args.rootdir + '/prospects/targetids.csv'))

print('\n\nDone.\n\n')
