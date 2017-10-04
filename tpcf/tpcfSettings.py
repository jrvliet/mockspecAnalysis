
'''
Returns the settings used to run an interactive TPCF

Usages:
    python tpcfSettings.py <tpcf filename>
'''


from __future__ import print_function
from pandas import HDFStore
import pprint
import sys

if len(sys.argv)!=2:
    print('\nUsage: python tpcfSettings.py <tpcf filename>\n')
    sys.exit()
    
fname = sys.argv[1]

store = HDFStore(fname)
attrs = store.get_storer('data').attrs.metadata
pp = pprint.PrettyPrinter()
pp.pprint(attrs)

store.close()

