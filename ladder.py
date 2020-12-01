import os
import glob
import argparse
import matplotlib
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from   matplotlib import rcParams
from   astropy.table import Table


plt.style.use('dark_background')

rcParams['font.family'] = 'Batang'
rcParams['font.sans-serif'] = ['Batang']

font = {'size': 50}
matplotlib.rc('font', **font)

score_types = ['howmany?!?', 'basics', 'notyourtype', 'lossofconfidence', 'aggorantmuch', 'round_score']

root_dir = os.environ['CSCRATCH'] + '/speedz_notdating/dryrun/'

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', default=root_dir, type=str)
parser.add_argument('--maxround', type=int, default=2)

args = parser.parse_args()

rounds = np.arange(args.maxround)

contestants = {}

for rround in rounds:
    scores_dir = args.rootdir + '/scores/{:d}/'.format(rround)
    scores     = glob.glob(scores_dir + '/*.json')

    if len(scores) == 0:
        raise ValueError('No entries in {}.'.format(scores_dir))
    
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
                ladder[i, rround, j] = contestants[contestant][rround].iloc[j, 1]

ladder = np.cumsum(ladder, axis=1)

final_scores = Table(np.c_[np.array(contestant_list), ladder[:, -1, -1]], names=['ENTRANT', 'FINAL SCORE'])
final_scores.sort('FINAL SCORE')
final_scores.reverse()

print('\n\n')

final_scores.pprint()

print('\n\n')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(1,1, figsize=(15,30))

for i, row in enumerate(np.unique(final_scores['FINAL SCORE'])):
    this_score = final_scores[final_scores['FINAL SCORE'] == row]

    ax.axhline(row, xmin=0., xmax=1., c=colors[i], lw=1.0, alpha=0.75)

    if len(this_score) > 1:
        string = '-'.join([x for x in this_score['ENTRANT']])
        
        ax.text(0.5, row, string, verticalalignment='bottom')
        
    else:
        ax.text(0.5, row, this_score['ENTRANT'][0], verticalalignment='bottom')

ax.get_xaxis().set_visible(False)
ax.set_title('Final scores after {} rounds'.format(args.maxround))

ax.set_xlim(0.0, 1.0)

pl.savefig('ladder_{:d}.pdf'.format(args.maxround), transparent=True)
