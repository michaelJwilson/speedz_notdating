import os
import glob
import json
import argparse
import numpy  as np
import pandas as pd

from   astropy.table import Table, join, vstack, Row

# [dryrun, test]
root_dir = os.environ['CSCRATCH'] + '/speedz_notdating/test/'

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', default=root_dir, type=str)
parser.add_argument('--round', type=int, default=0)
parser.add_argument('--idcheck', type=int, default=0)

args = parser.parse_args()

round_ids   = Table.read(os.environ['CSCRATCH'] + '/speedz_notdating/prospects/{:d}/targetids.csv'.format(args.round))['# TARGETID'].data

# 'merger comment', 'all VI issues', 'all VI comments'
columns     = ['TARGETID', 'Redrock z', 'best z', 'best quality', 'Redrock spectype', 'best spectype', 'N_VI', 'DELTACHI2', 'ZWARN']

truth_qsos  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_QSO_v1.2.csv')[columns]
truth_lrgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_LRG_v1.2.csv')[columns]
truth_elgs  = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_ELG_v1.2.csv')[columns]
truth_bgs   = Table.read('/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_BGS_v1.2.csv')[columns]

truth       = vstack((truth_qsos, truth_lrgs, truth_elgs, truth_bgs))
truth.sort('TARGETID')

## truth.pprint(max_width=-1)  

##
entries_dir = args.rootdir + '/entries/{:d}/'.format(args.round)
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
        entry       = pd.read_csv(x)
        entry       = entry.sort_values(by=['TARGETID'])

        entry_columns = list(entry.columns)
    
        entry       = Table(entry.to_numpy(), names=entry_columns)
        entry['TARGETID'] = np.array(entry['TARGETID'], dtype=np.int64)
        entry['VI_z'] = np.array(entry['VI_z'], dtype=np.float64)
        entry['VI_spectype'] = np.array(entry['VI_spectype'].data, dtype=np.str)
        entry['ENTRY_NUM'] = entry_num
        entry['Redrock_z'] = np.array(entry['Redrock_z'], dtype=np.float64)
        
        if author not in contestants:
            contestants[author] = {'entry_0': entry, 'all_entries': entry, 'nentry': 1}

        else:
            contestants[author]['nentry'] += 1  
            contestants[author]['entry_{:d}'.format(entry_num)] = entry
            contestants[author]['all_entries'.format(entry_num)] = vstack((contestants[author]['all_entries'.format(entry_num)], entry))
            
        round_table = vstack((round_table, entry)) # round_table.append(entry)

    except:
        print('Failure for {}'.format(x))

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

    output_path = args.rootdir + '/scores/{:d}/{}.json'.format(args.round, author)

    towrite = list(scores.dtype.names)
    
    f = open(output_path, "w")

    for ttype in towrite:
        f.write('{}\t{}\n'.format(ttype, scores[ttype][0]))

    f.close()

print('\n\nScores generated at {}.'.format(args.rootdir + '/scores/{:d}/'.format(args.round)))
    
print('\n\nDone.\n\n')
