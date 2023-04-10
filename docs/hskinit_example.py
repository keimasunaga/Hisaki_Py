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
envdic = {'hsk_l2_data_loc':'Path_to_l2_data_directory',
          'hsk_l2p_data_loc':'Path_to_l2p_data_directory',
          'hsk_cal_data_loc':'Path_to_cal_data_directory',
          'hsk_horizons_data_loc':'Path_to_horizons_data_directory',

          'hsk_l2_data_url_pub': 'https://data.darts.isas.jaxa.jp/pub/hisaki/euv/',
          'hsk_l2_data_url': 'URL_of_l2_data (team-only site)',
          'hsk_l2p_data_url':'URL_of_l2p_data'}

for ikey in envdic.keys():
    os.environ[ikey] = envdic[ikey]
