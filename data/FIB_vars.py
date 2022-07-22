# beach_init.py
# RTS - March 2018
# Update - RTS 2022

# 1.Creates directory for beaches on modeling list (if not already created)
#  In directory - variables folder, models folder, and saves raw FIB sample csv.
# 2. Adjusts Below detection limit
# 3. Creates FIB prevvariable dataset (save in var folder, date and time will be used to match enviro vars)

## TODO:  FIB1-FIB3 (FIB 1 - 3 days previous, NAN if no sample)

import pandas as pd
from numpy import log10
import numpy as np
import os
import shutil
from datetime import date


# def dls(df):
#     print('Adjusting sample times for DLS')
#     df_dls = pd.read_csv(dls_file)
#     df_dls.set_index('year', inplace=True)
#     dls_list = []
#     for i in df.index:
#         print(i.date())
#         if type(df['sample_time'].loc[i]) != str:
#             print('   No sample time for ' + str(i.date()))
#             dls_list.append(df['sample_time'].loc[i])
#             continue
#         year = i.year
#         drange = pd.date_range(start=df_dls.loc[year]['start'], 
#                                end = df_dls.loc[year]['end'], freq='D')
        
#         st = df.loc[i]['sample_time']
#         if i.date() in drange:
#             hour = pd.to_datetime(st).hour - 1
#             minute = pd.to_datetime(st).minute
#             if minute < 10:
#                 minute = '0' + str(minute)
#             st = str(hour) + ':' + str(minute)
#         dls_list.append(st)
#     df['dls'] = dls_list
#     return df

### Inputs #
base_folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting'
beach = 'Cowell' 

season = 'Summer'  # Summer, Winter , All
sd = '20070101'  # Start date (will parse out depending on season
ed = '20211231'  # End date

fib = ['EC', 'ENT']
thresholds = {'TC': 10000, 'EC': 400, 'ENT': 104}
LOQ = {'TC': 20, 'EC': 20, 'ENT': 10} # limit per standard methods
replace = 5  # Replacement value of 5 easy to distinguish

# dls_adjust = 1
# dls_file = '/Users/rtsearcy/Box/water_quality_modeling/data/daylight_savings.csv'

### Create Directories
beach_base_dir = os.path.join(base_folder,'beach')
beach_dir = os.path.join(beach_base_dir, beach.replace(' ','_'))
if beach.replace(' ','_') not in os.listdir(beach_base_dir):
    os.makedirs(beach_dir)
    os.makedirs(os.path.join(beach_dir,'variables'))
    os.makedirs(os.path.join(beach_dir,'models'))
    
### FIB Variables

## Load Raw FIB
fib_file = os.path.join(base_folder, 'data/FIB/', beach.replace(' ','_')+'_FIB.csv')
df_raw = pd.read_csv(fib_file, parse_dates=['date'], index_col=['date'])
df_raw.sort_index(inplace=True)  # ascending

df_raw = df_raw[~df_raw.index.duplicated()]  # drop duplicates (keep first sample of day)

## Process Censored Data
for f in fib:
    # All samples with qualifier < or <= to 10 (or less), replace with 5
    idx = df_raw[(df_raw[f+'_qual'].isin(['<','<='])) & (df_raw[f] <= LOQ[f])].index
    df_raw.loc[idx,f] = replace
    
    # All samples with qualifier < or <= to 10 (or less), replace with 5
    idx = df_raw[(df_raw[f] <= LOQ[f])].index
    df_raw.loc[idx,f] = replace

### Process Variables
df_vars = df_raw.copy()
df_vars = df_vars[['sample_time'] + fib]  # drop qualifying info

## Quant, Exceedaces, Antecedent, FIB 1-3 days previous
for f in fib:
    df_vars[f + '_ant'] = df_vars[f].dropna().shift(1)  # antecedent sample, skipping any missed samples in dataset
    
    for i in range(1,4): # for forecasts
        
        # FIB1-3
        temp = df_vars[f].shift(i, freq='D') # Shfits index back i days
        temp.rename(f+str(i), inplace=True, axis=1)
        df_vars = pd.merge(df_vars, temp, how='left', left_index=True, right_index=True)
        ## MAYBE: ADD FIB1_exc - FIB3_exc
        
        # antecedant for forecasts (most recent measurement at least i days before) [for persistence method]
        df_vars[f + '_ant'+ str(i)] = df_vars[f].dropna().resample('D').first().fillna(method='ffill').shift(i).reindex(index=df_vars.index)

    df_vars[f + '_quant'] = (df_vars[f] > replace).astype(int)  # = or exceeds LOQ? (binary)
    df_vars.loc[df_vars[f].isna(),f + '_quant'] = np.nan
    
    df_vars[f + '_ant_quant'] = (df_vars[f + '_ant'] > replace).astype(int)
    df_vars.loc[df_vars[f+'_ant'].isna(),f + '_ant_quant'] = np.nan
    # previous day quanitfied? (binary) 
    
    df_vars[f + '_exc'] = (df_vars[f] > thresholds[f]).astype(int)  # exceeds threshold? (binary)
    df_vars.loc[df_vars[f].isna(),f + '_exc'] = np.nan
    
    df_vars[f + '_ant_exc'] = (df_vars[f + '_ant'] > thresholds[f]).astype(int)
    df_vars.loc[df_vars[f+'_ant'].isna(),f + '_ant_exc'] = np.nan
    # previous day exceeds threshold? (binary)
    for i in range(1,4): # for forecasts
        df_vars[f + '_ant'+str(i)+'_exc'] = (df_vars[f + '_ant'+ str(i)] > thresholds[f]).astype(int)
        df_vars.loc[df_vars[f + '_ant'+ str(i)].isna(),f + '_ant'+str(i)+'_exc'] = np.nan
    
    # log10 transform
    df_vars['log' + f] = round(log10(df_vars[f] + 1), 2)
    df_vars['log' + f + '_ant'] = round(log10(df_vars[f + '_ant'] + 1), 2)
    for i in range(1,4): # FIB1-3
        df_vars['log' + f + str(i)] = round(log10(df_vars[f + str(i)] + 1), 2)
        df_vars['log' + f + '_ant' + str(i)] = round(log10(df_vars[f + '_ant' + str(i)] + 1), 2)

# var_order = fib + [f + '1' for f in fib] + [f + '_exc' for f in fib] \
#     + [f + '1_exc' for f in fib] + ['log' + f for f in fib] + ['log' + f + '1'for f in fib]
# df_vars = df_vars[var_order]

# Adjust for time range and season
df_vars = df_vars[sd:ed]
if season == 'Summer':
    df_vars = df_vars[(df_vars.index.month >= 4) & (df_vars.index.month < 11)]
elif season == 'Winter':
    df_vars = df_vars[(df_vars.index.month <= 3) | (df_vars.index.month >= 11)]

# ## Daylight Savings time adjustment
# # Assumes sample times provided HAVE NOT been adjusted to LST in the FIB database
# if dls_adjust == 1:
#     df_vars = dls(df_vars)
# else:
#     df_vars['dls'] = df_vars['sample_time']

# df_vars.reset_index(inplace=True)
# df_vars['dt'] = pd.to_datetime(df_vars['dt'].astype(str) + ' ' + df_vars['dls'], format='%Y-%m-%d %H:%M')
# df_vars.drop('dls',axis=1,inplace=True)
# df_vars.set_index('dt', inplace=True)

## Summary and Save variables
print(beach)
print('Variables (Season: ' + season + ', Sample Range: ' + str(df_vars.index.year[0]) + ' to '
      + str(df_vars.index.year[-1]) + ')')
print('Avg. Days Between Samples: ' + str((df_vars.index[1:] - df_vars.index[:-1]).mean().days) + ' days')
print('Avg. hour of sampling: ' + str(((pd.to_datetime(df_vars.sample_time).dt.hour*60 + pd.to_datetime(df_vars.sample_time).dt.minute)/60).median()))
print('\nNumber of Samples/Exceedance/BLOQ: ')
for f in fib:
    print(f + ' - ' + str(len(df_vars[f].dropna())) + '/' + str(df_vars[f+'_exc'].sum()) + '/' + str(len(df_vars[f].dropna()) - df_vars[f+'_quant'].sum()))
print('\nNum samples with samples taken 1-3 days previously:')
for i in range(1,4):
    print(str(i) + ' days: ' + str(len(df_vars[f+str(i)].dropna())))
var_file = os.path.join(beach_dir, 'variables', 'FIB_variables_' + beach.replace(' ', '_') + '.csv')
df_vars.to_csv(var_file)
print('\nSaved to : ' + var_file + '\n')
