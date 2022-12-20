# getRawWaves.py
# RS - 10/6/2016
# RS - 03/15/2018 UPDATE
# RTS - Jan 2022 - adjusted for air temp

# Summary:
# - Find raw CDIP wave data for given time period, converts from UTC to PST time, save to raw file
# - Using raw data, calculate daily averages and max parameters

# NOTE: Output files may contain non-consecutive data, or data that does not go until the end date (decom. stations)
# NOTE: Output files may be offset from date in title do to UTC to PST shift of 8 hrs.

import requests
import os
import pandas as pd

# Inputs #
outfolder_vars = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting/data/waves'
#outfolder_vars = '/Volumes/GoogleDrive/My Drive/high_frequency_wq/harbor_study_2022/data/hindcast'
outfolder_raw = os.path.join(outfolder_vars,'raw')
sd = '20170101'  # start date, Must be YYYYMMDD format
ed = '20220804'  # end date, Because UTC time, include extra day

stations = {
    # 'Imperial Beach Nearshore': '155', # Lots of missing data
    # 'Point Loma South':	'191',
    #'Scripps Nearshore': '201', # Recently online
    # 'Torrey Pines Outer': '100',
    #'Oceanside Offshore': '045',
    #'San Pedro': '092',
#    'Santa Monica Bay':	'028',
#    'Anacapa Passage': '111',
#    'Harvest': '071',
#    'Diablo Canyon': '076',
#    'Point Sur': '157',
#    'Cabrillo Point Nearshore':	'158',
#    'Monterey Bay Outer': '156',
     # 'Pt Santa Cruz': '254',
     # 'Monterey Bay West': '185',
     'San Francisco Bar': '142',
#    'Point Reyes': '029',
#    'Cape Mendocino': '094',
#    'Humboldt Bay North Spit': '168',
#    'Ocean Station Papa': '166', # Offshore
     # 'Aunuu':'189' # Offshore
}

# Find Raw CDIP Data #
print('CDIP Wave Data\nRaw Directory: ' + outfolder_raw + '\nVariable Directory: ' + outfolder_vars)
df_missing = pd.DataFrame()
for key in stations:
    st_name = key
    st = stations[key]
    outname_raw = st_name.replace(' ', '_') + '_raw_wave_data_' + sd + '_' + ed + '.csv'
    out_file_raw = os.path.join(outfolder_raw, outname_raw)

    if outname_raw not in os.listdir(outfolder_raw):
        # Grab data from CDIP website #
        print('\nGrabbing CDIP wave data from ' + st_name + ' station (' + st + ')')
        url = 'http://cdip.ucsd.edu/data_access/ndar?' + st + '+pm+' + sd + '-' + ed
        # date range using CDIP NDAR, grab wave parameters, no header
        web = requests.get(url)
        try:
            web.raise_for_status()
        except Exception as exc:
            print('  There was a problem grabbing wave data for Station ' + st + ': %s' % exc)

        # Create DataFrame, index with timestamp, convert blanks/errors to NaN #
        data = [line.split() for line in web.text.splitlines()]
        data = data[:-1]  # exclude footer
        data[0][0] = data[0][0].replace('<pre>', '')  # remove header
        df_raw = pd.DataFrame(data)
        if len(df_raw.columns) == 11: 
            df_raw.drop(10,inplace=True, axis=1) # Remove air temp column (use met stations instead)
        df_raw.columns = ['year', 'month', 'day', 'hour', 'minute', 'WVHT', 'DPD', 'MWD', 'APD', 'wtemp_b']#,'atemp_b']
        df_raw['dt'] = pd.to_datetime(df_raw[['year', 'month', 'day', 'hour', 'minute']])
        df_raw.set_index('dt', inplace=True)
        df_raw.index = df_raw.index.shift(-8, freq='h')  # convert from UTC to PST (- 8 hrs)
        df_raw = df_raw[['WVHT', 'DPD', 'MWD', 'APD', 'wtemp_b']]#,'atemp_b']]
        df_raw = df_raw.apply(lambda x: pd.to_numeric(x, errors='coerce'))

        # Save to raw outfile #
        df_raw.to_csv(out_file_raw)
        print('  Raw wave data saved to ' + outname_raw)
    else:
        print('\nGrabbing CDIP wave data for ' + st_name + ' from ' + outname_raw)
        df_raw = pd.read_csv(out_file_raw)
        df_raw['dt'] = pd.to_datetime(df_raw['dt'])
        df_raw.set_index('dt', inplace=True)

    # Create DF for daily variables
    wave_round = {'WVHT': 2, 'DPD': 2, 'MWD': 0, 'APD': 2, 'wtemp_b': 1}#, 'atemp_b': 1}  # sig figs on CDIP
    df_vars = pd.DataFrame(index=df_raw.resample('D').mean().index)
    
    # Present observations (Measurement around 3a PST)
    # ** Waves change on the order of 12 hrs, so 3a is an OK estimate
    pres_obv = df_raw.groupby(df_raw.index.date).apply(lambda x: x.iloc[0])
    df_vars = df_vars.merge(pres_obv, how = 'left', left_index = True, right_index=True)
    
    # Previous Day Vars
    for i in range(1,6):
        for c in df_raw.columns:
            r = wave_round[c]
            df_vars[c + str(i)] = round(df_raw[c].resample('D').mean().shift(i, freq='D'), r)  # previous day mean
            #df_vars[c + '1_max'] = df_raw[c].resample('D').max().shift(1, freq='D')  # previous day max
            #df_vars[c + '1_min'] = df_raw[c].resample('D').min().shift(1, freq='D')  # previous day min
    
    
    # Impute
    df_vars = df_vars.fillna(method='ffill', limit=3)  # forward fill variables up to 3 days
    df_miss_temp = df_vars.isnull().sum().rename(st_name)
    df_miss_temp.loc['start'] = str(df_vars.first_valid_index())
    df_miss_temp.loc['end'] = str(df_vars.last_valid_index())
    df_missing = df_missing.append(df_miss_temp)

    # Save to variable outfile #
    outname_vars = st_name.replace(' ', '_') + '_wave_variables_' + sd + '_' + ed + '.csv'
    out_file_vars = os.path.join(outfolder_vars, outname_vars)
    df_vars.to_csv(out_file_vars)
    print(str(df_vars.index[0].date()) + ' - ' + str(df_vars.index[-1].date()))
    print('  Daily wave variables saved to ' + outname_vars)

# Save missing stats df
#df_missing.to_csv(os.path.join(outfolder_vars, 'missing_days_' + sd + '_' + ed + '.csv'))
