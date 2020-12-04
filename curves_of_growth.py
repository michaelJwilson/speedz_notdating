import os
import glob
import argparse
import matplotlib
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from   matplotlib.ticker import FormatStrFormatter
from   matplotlib.ticker import MaxNLocator
from   matplotlib import rcParams
from   astropy.table import Table


rcParams['font.family'] = 'Batang'
rcParams['font.sans-serif'] = ['Batang']

# font = {'size': 50}
# matplotlib.rc('font', **font)

score_types = ['How many?!?', 'Basics', 'Not your type?', 'Loss of confidence?', 'Arrogant much?', 'Round score']

root_dir = '/Users/MJWilson/Work/speedz_notdating/'

# '/Users/MJWilson/Work/speedz_notdating/test/'
parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', default=root_dir, type=str)
parser.add_argument('--inputdir', required=True, type=str)
parser.add_argument('--maxround', type=int, required=True)

args = parser.parse_args()

# assert  (args.maxround >= 2)

rounds = np.arange(args.maxround, dtype=np.int)

contestants = {}

for rround in rounds:
    scores_dir = args.inputdir + '/scores/{:d}/'.format(rround)
    scores     = glob.glob(scores_dir + '/*.json')

    print('Reducing {}'.format(scores_dir))
    '''
    if len(scores) == 0:
        raise ValueError('No entries in {}.'.format(scores_dir))
    '''
    for score in scores:
        author = score.split('/')[-1].split('.')[0]

        if author not in contestants.keys():
            contestants[author] = {}        
        
        score = pd.read_csv(score, sep='\s+', names=['type', 'score'])
        contestants[author][rround] = score

contestant_list = list(contestants.keys())
            
ladder = np.zeros(shape=(len(contestant_list), len(rounds), len(score_types)), dtype=np.int)            

for i, contestant in enumerate(contestant_list):
    for rround in rounds:
        for j, ttype in enumerate(score_types):
            if rround in contestants[contestant].keys():
                ladder[i, rround, j] = np.int(contestants[contestant][rround].iloc[j, 1])

ladder = np.cumsum(ladder, axis=1, dtype=np.int)

final_scores = Table(np.c_[np.array(contestant_list), ladder[:, -1, -1]], names=['ENTRANT', 'FINAL SCORE'], dtype=[np.str, np.int])
final_scores.sort('FINAL SCORE')
final_scores.reverse()

print('\n\n')

final_scores.pprint()

print('\n\n')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(1,6, figsize=(20,5))

plt.subplots_adjust(hspace=0.4)

for j, ttype in enumerate(score_types):
    ax[j].set_title(ttype)
    
    for i, contestant in enumerate(contestant_list):
        ax[j].plot(rounds, ladder[i, :, j], label=contestant)

    ax[j].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[j].yaxis.set_major_locator(MaxNLocator(integer=True))

    # ax[j].set_ylim(-1, np.maximum(5, ax[j].get_ylim()[1]))
    ax[j].set_xlabel('Round')
    
# pl.savefig('ladder_{:d}.pdf'.format(args.maxround), transparent=True)
ax[-1].legend(frameon=False, ncol=2, bbox_to_anchor=(1.025, 1))

ax[-1].set_yscale('log')

fig.suptitle('Curves of growth', fontsize=15)

pl.savefig('curve_of_growth_{:d}.png'.format(args.maxround))

