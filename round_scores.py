import os
import glob
import json
import argparse
import numpy  as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt

from   astropy.table import Table, join, vstack, Row
from   pathlib import Path


# [dryrun, test], e.g. /global/cscratch1/sd/mjwilson/speedz_notdating/dryrun/entries/0/
input_dir = os.environ['CSCRATCH'] + '/speedz_notdating/ecs/'

print(input_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--inputdir', required=True, type=str)
parser.add_argument('--round', type=int, required=True)
parser.add_argument('--idcheck', type=int, default=0)

args = parser.parse_args()

round_ids   = Table.read(os.environ['CSCRATCH'] + '/speedz_notdating/prospects/{:d}/targetids.csv'.format(args.round))['# TARGETID'].data

# 'merger comment', 'all VI issues', 'all VI comments'
columns     = ['TARGETID', 'Redrock z', 'best z', 'best quality', 'Redrock spectype', 'best spectype', 'N_VI', 'DELTACHI2', 'ZWARN']

truth_qsos  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_QSO_v1.2.csv')[columns]
truth_lrgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_LRG_v1.2.csv')[columns]
truth_elgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_ELG_v1.2.csv')[columns]
truth_bgs   = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_BGS_v1.2.csv')[columns]
truth_bbgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/Andes_reinspection/BGS/Truth_table_Andes_reinspection_BGS_70500_20200303_v1.csv')

truth       = vstack((truth_qsos, truth_lrgs, truth_elgs, truth_bgs, truth_bbgs))
truth.sort('TARGETID')

# uids, cnts  = np.unique(truth['TARGETID'], return_counts=True) 
# assert  cnts.max() == 1

## truth.pprint(max_width=-1)  

##
entries_dir = args.inputdir + '/entries/{:d}/'.format(args.round)
entries     = glob.glob(entries_dir + '/*.csv')

round_table = Table() # pd.read_csv(entries[0])

print('\n\nReducing {} for round {}.'.format(entries_dir, args.round))

contestants = dict()

for x in entries:
    print('\t{}'.format(x))

    try:
        # desi-vi_speedz_notdating_1_MJW.csv
        author      = x.split('_')[-1].replace('.csv', '').upper()
        entry_num   = np.int(x.split('_')[-2])
    
        # TARGETID EXPID NIGHT TILEID Spec_version Redrock_version Template_version Redrock_spectype Redrock_z VI_scanner VI_quality VI_issue VI_z VI_spectype VI_comment 
        # entry     = pd.read_csv(x)
        # entry     = entry.sort_values(by=['TARGETID'])

        entry       = Table.read(x)
        entry.sort('TARGETID')

        print(entry)
        
        # entry_columns = list(entry.columns)
        # entry       = Table(entry.to_numpy(), names=entry_columns)

        entry['TARGETID'] = np.array(entry['TARGETID'], dtype=np.int64)
        entry['VI_z'] = np.array(entry['VI_z'], dtype=np.float64)
        entry['VI_spectype'] = np.array(entry['VI_spectype'].data, dtype=np.str)
        entry['ENTRY_NUM'] = entry_num
        entry['Redrock_z'] = np.array(entry['Redrock_z'], dtype=np.float64)

        del entry['VI_issue']
        del entry['VI_comment']
        
        if author not in contestants:
            contestants[author] = {'entry_0': entry, 'all_entries': entry, 'nentry': 1}

        else:
            contestants[author]['nentry'] += 1  
            contestants[author]['entry_{:d}'.format(entry_num)] = entry
            contestants[author]['all_entries'.format(entry_num)] = vstack((contestants[author]['all_entries'.format(entry_num)], entry))
            
        round_table = vstack((round_table, entry)) # round_table.append(entry)

    except Exception as e:
        print('Failure for {}'.format(x))

        print(e)
        
    if args.idcheck:
        assert  np.all(np.isin(entry['TARGETID'].data, round_ids))
        
print('\n\n')        
print(contestants.keys())

print('\n\n')

toloop = list(contestants.keys())

# In the absence of zerr.
dz   = 300. / 2.9979e5
atol = 3. * dz
atol = 1.e-2


labels   = ['basics_pos', 'basics_neg', 'basics_miss', 'notyourtype',    'lossofconfidence',    'arrogantmuch']
plabels  = ['How many?!?', 'Basics +',   'Basics -',   'Basics miss', 'Not your type?', 'Loss of confidence?', 'Arrogant much?']

plabels = np.array(plabels)

# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = pl.cm.tab20c(np.linspace(0, 1, len(labels)))

for author in toloop:
    merge = join(contestants[author]['all_entries'], truth['TARGETID', 'best z', 'best quality', 'best spectype'], join_type='left', keys='TARGETID')
    
    contestants[author]['merged_entry'] = merge

    scores = Table(np.array([5 * len(merge)]), names=['howmany?!?'], meta={'name': author})
    
    # Overrule redrock redshift & find correct redshift (as judged by truth).    
    redshifted = [~np.isnan(x) for x in merge['VI_z'].data]
    merge['basics_pos'] = redshifted & (merge['VI_quality'] >= 2.5) & (merge['best quality'] >= 2.5) & (np.isclose(merge['VI_z'].data, merge['best z'].data, atol=dz))
    scores['basics']  = 50 * np.count_nonzero(merge['basics_pos'])

    # Overrule redrock redshift & find incorrect redshift (as judged by truth).
    merge['basics_neg'] = redshifted & (merge['VI_quality'] >= 2.5) & (merge['best quality'] >= 2.5) & (np.logical_not(np.isclose(merge['VI_z'].data, merge['best z'].data, atol=dz)))
    scores['basics'] -= 50 * np.count_nonzero(merge['basics_neg'])

    # Missed correcting redshift.
    merge['basics_miss'] = (merge['best quality'] >= 2.5) & (np.logical_not(np.isclose(merge['Redrock_z'].data, merge['best z'].data, atol=atol))) & np.array([np.isnan(x) for x in merge['VI_z'].data])
    scores['basics'] -= 25 * np.count_nonzero(merge['basics_miss'])

    merge['basics'] = merge['basics_pos'] | merge['basics_neg'] | merge['basics_miss']
    
    # Incorrecly typed.
    typed = [x != 'nan' for x in merge['VI_spectype'].data]
    merge['notyourtype'] = (merge['VI_quality'] >= 2.5) & (merge['best quality'] >= 2.5) & typed & (merge['VI_spectype'].data != merge['best spectype'].data)
    scores['notyourtype'] = -15 * np.count_nonzero(merge['notyourtype'])

    # Correcly typed.                                                                                                                                                                                                                       
    typed = [x != 'nan' for x in merge['VI_spectype'].data]
    merge['notyourtype'] = (merge['VI_quality'] >= 2.5) & (merge['best quality'] >= 2.5) & typed & (merge['VI_spectype'].data == merge['best spectype'].data)
    scores['notyourtype'] = +15 * np.count_nonzero(merge['notyourtype'])

    # Missed type.                                                                                                                                                                                                                                                                                                                                    
    merge['notyourtype'] = (merge['best quality'] >= 2.5) & (merge['Redrock_spectype'].data != merge['best spectype'].data) &  [x == 'nan' for x in merge['VI_spectype'].data]
    scores['notyourtype'] = -15 * np.count_nonzero(merge['notyourtype'])
    
    # Every high-quality case the user gives low quality < 2.5.
    merge['lossofconfidence'] = (merge['VI_quality'] < 2.5) & (merge['best quality'] >= 2.5)
    scores['lossofconfidence'] = -5 * np.count_nonzero(merge['lossofconfidence'])
        
    # Every low-quality case the user gives quality >= 2.5.
    merge['arrogantmuch'] = (merge['VI_quality'] >= 2.5) & (merge['best quality'] < 2.5)
    scores['arrogantmuch'] = -5 * np.count_nonzero(merge['arrogantmuch'])

    scores['roundscore']  = scores['howmany?!?'][0] + scores['notyourtype'][0] + scores['lossofconfidence'][0] + scores['basics'][0] + scores['arrogantmuch'][0]
    
    contestants[author]['score_breakdown'] = scores
       
    print('\n\t\t\t----  {}  ----'.format(author))

    merge.pprint(max_width=-1, max_lines=-1)

    print('\n\n')
    
    scores.pprint()

    Path(args.inputdir + '/scores/{:d}/merge/'.format(args.round)).mkdir(parents=True, exist_ok=True)

    output_path = args.inputdir + '/scores/{:d}/merge/{}.fits'.format(args.round, author)

    merge.write(output_path, format='fits', overwrite=True)
    
    output_path = args.inputdir + '/scores/{:d}/{}.json'.format(args.round, author)

    towrite = list(scores.dtype.names)
    
    f = open(output_path, "w")

    for ttype in towrite:
        f.write('{}\t{}\n'.format(ttype, scores[ttype][0]))

    f.close()

    # basics_pos basics_neg basics_miss basics notyourtype lossofconfidence arrogantmuch    
    sizes  = [scores['howmany?!?'][0]] + [np.count_nonzero(merge[x]) for x in labels]
    sizes  = np.array(sizes)
    
    explode = np.array([0.0] * len(sizes))

    pl.clf()
    
    fig, ax  = plt.subplots(figsize=(7.5, 7.5))

    size     = 0.3
    
    ax.pie(sizes, labels=plabels, autopct='%1.1f%%', shadow=False, startangle=145, explode=explode, colors=colors, rotatelabels=True, wedgeprops={'alpha': 0.5, 'width': size, 'edgecolor': 'w'}, radius=1, textprops={'fontsize': 7})
    
    ax.set_title('Round {}: {}'.format(args.round, author))

    ax.axis('equal')

    Path(args.inputdir + '/scores/{:d}/plots/'.format(args.round)).mkdir(parents=True, exist_ok=True)
    
    fig.savefig(args.inputdir + '/scores/{:d}/plots/{}.png'.format(args.round, author))
    
print('\n\nScores generated at {}.'.format(args.inputdir + '/scores/{:d}/'.format(args.round)))
    
print('\n\nDone.\n\n')
