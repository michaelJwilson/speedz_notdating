from astropy.table import Table


two = Table.read('finalscores_2.fits')
one = Table.read('finalscores_1.fits')

entrants = [x for x in two['ENTRANT'] if x in one['ENTRANT']]


for x in entrants:
    print('{}\t{: 5d}\t{: 5d}\t{: 5d}'.format(x, one['FINAL SCORE'][one['ENTRANT'] == x][0], two['FINAL SCORE'][two['ENTRANT'] == x][0], two['FINAL SCORE'][two['ENTRANT'] == x][0] - one['FINAL SCORE'][one['ENTRANT'] == x][0]))
