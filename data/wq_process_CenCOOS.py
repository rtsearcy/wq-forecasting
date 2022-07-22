# get_process_CIMIS.py
# RTS 2022

# Processes water quality data from CenCOOS stored in csv files. Creates time lagged variables

import pandas as pd
import numpy as np
import os

# Inputs #
folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting/data/water_quality/CenCOOS/'
#file = 'sc_wharf_water_quality_raw.csv'
file = 'newport_water_quality_raw.csv'

### Read Raw Files
df = pd.read_csv(os.path.join(folder,'raw', file), parse_dates=['time'], index_col=['time'])

## UTC to Local
df.index.name = 'dt'
df.index = df.index.shift(-8, freq='h')

## Correct bad values
if 'pH' in df.columns:
    df.loc[df.pH < 5,'pH'] = np.nan
    df.loc[df.pH > 14,'pH'] = np.nan

if 'wtemp' in df.columns:
    df.loc[df.wtemp > 40,'wtemp'] = np.nan

## Daily mean/max value
df_daily = df.resample('D').mean()

# Save to file
df_daily.to_csv(os.path.join(folder, file.replace('raw','daily')))  # PD Series
