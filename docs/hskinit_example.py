"""
This file has to be imported first when you use Hisaki_py.
"""

## Add directories of Hisaki_py to Pythonpath
import sys
path_hisaki_py = 'path_to_your_hisaki_py_directory'
if path_hisaki_py not in sys.path:
    sys.path.append(path_hisaki_py)

## Add directories of your library to Pythonpath
## Add by yourself here.

## Add env paths
import os
envdic = {'dataloc_hsk':'Path_to_Hisaki_data_directory',
          'calloc_hsk':'Path_to_caldata_directory',
          'horizonsloc_hsk':'Path_to_horizons_data_directory'}
for ikey in envdic.keys():
    os.environ[ikey] = envdic[ikey]
