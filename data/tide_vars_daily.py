# tideVarsDaily.py - Computes daily tidal variables from raw CO-OPS 6 min water level data
# RS - 1/5/2017
# RS - 3/14/2018 - Update
# RTS - 11/11/2019
# RTS - Feb 2-22

# Raw data source/description: https://tidesandcurrents.noaa.gov/tide_predictions.html

# NOTE: raw csv files should have timestamps in LST

# Tide Variables (Continuous): tide (level at 10), tide_max, tide_min, tide_range (tidal range = max - min), 
# tide_max1, tide_min1, tide_range1
# Tide Variables (Binary/Category): tide_stage (low, high, transition), tide_spring (spring/neap)

import pandas as pd
import os
import re

# Import raw data csv to pd DataFrame
path = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting/data/tide/'
raw_path = os.path.join(path, 'raw')

sd = '20000101'  # Start date (account for previous day, conservative)
ed = '20221231'  # End date

sample_hour = 10 # avg of most beaches

# Spring/Neap tide data
moon = pd.read_csv(os.path.join(path,'full_moon.csv'), index_col=['date'],parse_dates=['date'])


# TO PROCESS SINGLE FILE
# (change indent)
# file = 'NorthSplit_Tidal_Data_20080101_20201231.csv'
# infile = os.path.join(infolder, file)

# TO ITERATE THROUGH ALL FILES
for file in os.listdir(raw_path):
    if not file.endswith('.csv'):
        continue
    infile = os.path.join(raw_path, file)
    station = re.sub('_tide_.+', '', file).replace('_', ' ')  # find station name from filename
    print('\nProcessing tidal data for ' + station + ' ...')
    df_raw = pd.read_csv(infile, index_col=['dt'],parse_dates=['dt'])
    df_raw = df_raw[sd:ed]  # Only samples in time range (for speed)
    df_out = pd.DataFrame(index=df_raw.resample('D').mean().index)  # Preset index to days

    #%% Tide (Tide level at 10a PST) - MAY NEED TO VARY THIS IF SAMPLE TIME IS KNOWN
    df_out['tide'] = df_raw[(df_raw.index.hour == sample_hour) & (df_raw.index.minute == 0)].resample('D').mean()

    #%% Max/Min/Range
    df_out['tide_max'] = df_raw.resample('D').max()  # Max tide
    df_out['tide_min'] = df_raw.resample('D').min()  # Min tide
    df_out['tide_range'] = df_out.tide_max - df_out.tide_min  # Tidal range
    print('  Maximum: ' + str(df_out['tide_max'].max()) + ' ; Minimum: ' + str(df_out['tide_min'].min()) +
          ' ; Max Range: ' + str(df_out['tide_range'].max()))
    
    df_out['tide_max1'] = df_out['tide_max'].shift(1,freq='D') # Max tide prev day
    df_out['tide_min1'] = df_out['tide_min'].shift(1,freq='D')  # Min tide
    df_out['tide_range1'] = df_out['tide_range'].shift(1,freq='D')
    
    #%% Tide Stage (at 10a)
    
    #%% Hours since High/Low
    # high_time = df_raw.groupby(pd.Grouper(freq='D')).idxmax()
    # df_out['tide_time_since_high'] = high_time['tide'].dt.hour - sample_hour

    #%% Spring/Neap
    df_out = pd.merge(df_out, moon['tide_spring'], left_index=True, right_index=True)
    
    #%% Save to file
    of_name = station.replace(' ', '_') + '_tide_variables_' + sd + '_' + ed + '.csv'
    outfile = os.path.join(path, of_name)
    df_out.index.rename('date', inplace=True)
    df_out.to_csv(outfile)
