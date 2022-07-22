# upwelling_process.py
# RTS 2022

# Processes daily CUTI upwelling data. Creates time lagged variables
# Note: must manually change the CIMIS output titles
# http://www.cimis.water.ca.gov/

import pandas as pd
import numpy as np
import os

# Inputs #
folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting/data/upwelling'
file = 'upwelling_CUTI_raw.csv'

df = pd.read_csv(os.path.join(folder, file), parse_dates=['date'], index_col=['date'])

stations = {
    'San Diego': '33N',
    'Santa Monica': '34N',
    'Santa Cruz': '37N',
    'San Francisco': '38N'
    }

for s in stations.keys():
    upwell = df[stations[s]]
    upwell.name = 'upwell'
    upwell = upwell.to_frame()
    upwell['upwell_bin'] = 0
    upwell.loc[upwell.upwell>0, 'upwell_bin'] = 1

    cols = upwell.columns
    for i in range(1, 4):
        for c in cols:
            upwell[c+str(i)] = upwell[c].shift(i, freq='D')

    # TODO: 
    # total_list = [14]
    # for j in total_list:  # rain2T-rain7T
    #     df['rain' + str(j) + 'T'] = 0.0
    #     for k in range(j, 0, -1):
    #         df['rain' + str(j) + 'T'] += df['rain'].shift(k, freq='D')
    #     df['lograin' + str(j) + 'T'] = round(np.log10(df['rain' + str(j) + 'T'] + 1), 2)
    
    # Save to file
    outfile = s.replace(' ','_') + '_upwelling.csv'
    upwell.to_csv(os.path.join(folder, outfile))  
