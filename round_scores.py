import os
import glob
import json
import argparse
import numpy  as np
import pandas as pd

from   astropy.table import Table, join, vstack, Row


root_dir = os.environ['CSCRATCH'] + '/speedz_notdating/'

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', default=root_dir, type=str)
parser.add_argument('--round', type=int, default=0)

args = parser.parse_args()

# 'merger comment', 'all VI issues', 'all VI comments'
columns     = ['TARGETID', 'Redrock z', 'best z', 'best quality', 'Redrock spectype', 'best spectype', 'N_VI', 'DELTACHI2', 'ZWARN']

truth_qsos  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_QSO_v1.2.csv')[columns]
truth_lrgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_LRG_v1.2.csv')[columns]
truth_elgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_ELG_v1.2.csv')[columns]
truth_bgs   = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_BGS_v1.2.csv')[columns]

truth       = vstack((truth_qsos, truth_lrgs, truth_elgs, truth_bgs))
truth.sort('TARGETID')

truth.pprint(max_width=-1)  

##
rround      = 1

entries_dir = args.rootdir + '/entries/{:d}/'.format(args.round)
entries     = glob.glob(entries_dir + '/*.csv')

round_table = Table() # pd.read_csv(entries[0])

print('Reducing {} for round {}'.format(entries_dir, rround))

contestants = dict()

for x in entries:
    author      = x.split('_')[-1].replace('.csv', '')

    # TARGETID EXPID NIGHT TILEID Spec_version Redrock_version Template_version Redrock_spectype Redrock_z VI_scanner VI_quality VI_issue VI_z VI_spectype VI_comment 
    entry       = pd.read_csv(x)
    entry       = entry.sort_values(by=['TARGETID'])

    entry_columns = list(entry.columns)
    
    entry       = Table(entry.to_numpy(), names=entry_columns)
    entry['TARGETID'] = np.array(entry['TARGETID'], dtype=np.int64)
    entry['VI_z'] = np.array(entry['VI_z'], dtype=np.float64)
    entry['VI_spectype'] = np.array(entry['VI_spectype'].data, dtype=np.str)
        
    if author not in contestants:
        contestants[author] = {'entry': entry}
        round_table         = vstack((round_table, entry)) # round_table.append(entry)

print('\n\n')        
print(contestants.keys())

print('\n\n')

toloop = list(contestants.keys())

# In the absence of zerr.
dz = 50. / 2.9979e5

for author in toloop:
    merge = join(contestants[author]['entry'], truth['TARGETID', 'best z', 'best quality', 'best spectype'], join_type='left', keys='TARGETID')
    
    contestants[author]['merged_entry'] = merge

    scores = Table(np.array([len(merge)]), names=['howmany?!?'], meta={'name': author})
    
    # -1.5 points for every incorrect high-quality redshift (so the net reduction is 0.5).    
    redshifted = [~np.isnan(x) for x in merge['VI_z'].data]
    scores['basics'] = np.count_nonzero(redshifted & (merge['VI_quality'] >= 2.5) & (merge['best quality'] >= 2.5) & (np.logical_not(np.isclose(merge['VI_z'].data, merge['best z'].data, atol=dz))))
    
    # Incorrecly typed.
    typed = [x != 'nan' for x in merge['VI_spectype'].data]
    scores['notyourtype'] = np.count_nonzero((merge['VI_quality'] >= 2.5) & (merge['best quality'] >= 2.5) & typed & (merge['VI_spectype'].data != merge['best spectype'].data))
    
    # Every high-quality case the user gives low quality < 2.5.
    scores['lossofconfidence'] = np.count_nonzero((merge['VI_quality'] < 2.5) & (merge['best quality'] >= 2.5))
        
    # Every low-quality case the user gives quality >= 2.5.
    scores['arrogantmuch'] = np.count_nonzero((merge['VI_quality'] >= 2.5) & (merge['best quality'] < 2.5))

    scores['roundscore']  = scores['howmany?!?'][0] - scores['notyourtype'][0] - scores['lossofconfidence'][0] - 2 * scores['basics'][0] - 2 * scores['arrogantmuch'][0]
    scores['roundscore'] *= 5
    
    contestants[author]['score_breakdown'] = scores
       
    print('\n\t\t\t----  {}  ----'.format(author))

    scores.pprint()

    output_path = args.rootdir + '/scores/{:d}/{}.json'.format(args.round, author)

    towrite = list(scores.dtype.names)
    
    f = open(output_path, "w")

    for ttype in towrite:
        f.write('{}\t{}\n'.format(ttype, scores[ttype][0]))

    f.close()
    
print('\n\nDone.\n\n')
