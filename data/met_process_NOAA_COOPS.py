# get_process_CenCOOS.py
# RTS 2022

# Processes CenCOOS met data stored in csv files. Creates time lagged variables
# Note: must manually change the CIMIS output titles
# https://data.cencoos.org/#metadata/102580/station (SC Wharf)

import pandas as pd
import numpy as np
import os
import scipy.stats as stats

def mode(xs):
    try:
        return stats.mode(xs)[0][0]
    except IndexError:
        return np.nan

# Inputs #
folder = '/Users/rtsearcy/Box/water_quality_modeling/forecasting/data/met/CenCOOS'
file = 'sc_wharf_met.csv'
angle = 135  # beach angle, perpendicular, from N (Cowell = 135, OB SF = 270)

df = pd.read_csv(os.path.join(folder, file), parse_dates=['dt'], index_col=['dt'])
df = df.dropna(how='all')

## Convert UTC to LST
df = df.shift(-8, freq='H')

## Met Vars
wind = df[['wdir', 'wspd']].copy()
met = df[[c for c in df if c not in wind]].copy()

wind['owind'] = wind.wspd * np.cos((wind.wdir - angle) * np.pi / 180)  # offshore wspd
wind['awind'] = wind.wspd * np.sin((wind.wdir - angle) * np.pi / 180) # alongshore wspd
wind['wdir_offshore'] = 0  # binary offshore
wind.loc[((wind.wdir - angle) >= -90) & ((wind.wdir - angle) <= 90), 'wdir_offshore'] = 1
offshore = wind.wdir_offshore.copy()
wind.drop(['wspd', 'wdir','wdir_offshore'], axis=1, inplace=True)

offshore = offshore.resample('D').apply(mode)

df = pd.merge(met, wind, left_index=True, right_index=True)
df = df.resample('D').mean().round(2)
df = pd.merge(df, offshore, left_index=True, right_index=True)

cols = df.columns
for i in range(1, 4):
    for c in cols:
        df[c+str(i)] = df[c].shift(i, freq='D')
        

# # rain
# for i in range(1, 8):  # rain1 - rain7, lograin1-lograin7
#     df['rain' + str(i)] = df['rain'].shift(i, freq='D')
#     df['lograin' + str(i)] = round(np.log10(df['rain' + str(i)] + 1), 2)
# total_list = list(range(2, 8)) + [14, 30]
# for j in total_list:  # rain2T-rain7T
#     df['rain' + str(j) + 'T'] = 0.0
#     for k in range(j, 0, -1):
#         df['rain' + str(j) + 'T'] += df['rain'].shift(k, freq='D')
#     df['lograin' + str(j) + 'T'] = round(np.log10(df['rain' + str(j) + 'T'] + 1), 2)

# Save to file
df.index = df.index.date
df.index.rename('date', inplace=True)

outfile = file.replace('.csv', '_processed.csv')
df.to_csv(os.path.join(folder, outfile))  # PD Series
