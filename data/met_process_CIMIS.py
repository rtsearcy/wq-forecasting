# get_process_CIMIS.py
# RTS 2022

# Processes daily met data stored in csv files. Creates time lagged variables
# Note: must manually change the CIMIS output titles
# http://www.cimis.water.ca.gov/

import pandas as pd
import numpy as np
import os

# Inputs #
folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting/data/met/CIMIS'
file = 'Irvine_CIMIS_daily.csv'

df = pd.read_csv(os.path.join(folder, file), parse_dates=['date'], index_col=['date'])
for i in range(1, 4):
    for c in ['atemp', 'rad']:
        df[c+str(i)] = df[c].shift(i, freq='D')

# rain
for i in range(1, 8):  # rain1 - rain7, lograin1-lograin7
    df['rain' + str(i)] = df['rain'].shift(i, freq='D')
    df['lograin' + str(i)] = round(np.log10(df['rain' + str(i)] + 1), 2)

total_list = list(range(2, 8)) + [14, 30]
for j in total_list:  # rain2T-rain7T
    df['rain' + str(j) + 'T'] = 0.0
    for k in range(j, 0, -1):
        df['rain' + str(j) + 'T'] += df['rain'].shift(k, freq='D')
    df['lograin' + str(j) + 'T'] = round(np.log10(df['rain' + str(j) + 'T'] + 1), 2)

# Save to file
df.to_csv(os.path.join(folder, file))  # PD Series
