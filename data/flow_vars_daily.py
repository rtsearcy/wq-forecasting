# flowVarsDaily.py - Grabs daily data from USGS and calculates previous day flow variables
# Source: http://waterdata.usgs.gov/ca/nwis/current/?type=flow
# RTS - March 2018 UPDATE

import pandas as pd
import numpy as np
import requests
import os

# Inputs #
sd = '2000-01-01'  # YYYY-MM-DD format, include one day previous
ed = '2021-12-31'
outfolder = '/Users/rtsearcy/Box/water_quality_modeling/forecasting/data/flow'
outfolder_raw = os.path.join(outfolder, 'raw')
df_summary = pd.DataFrame()

stations = {
            'San Diego River': '11023000',  # SD (Ocean Beach, Mission Bay)
            #'San Luis Rey River': '11042000',  # SD - Oceanside
            'San Juan Creek': '11046530',   # OC - Doheny State
            'Arroyo Trabuco Creek': '11047300',  # OC - Doheny State
            'Santa Ana River': '11078000',  # OC - Huntington, Newport, SA Rivermouth
            #'Rio Hondo': '11102300',  # LA River/Long Beach
            #'San Gabriel River': '11087020',  # Seal Beach/LB
            #'Ventura River': '11118500',  # Emmawood/Ventura Point
            #'Carpinteria Creek': '11119500',  # Carp. State Beach
            #'Mission Creek Upper': '11119745',  # East Beach
            #'Mission Creek Lower': '11119750',  # East Beach
            #'Atascadero Creek': '11120000',  # Goleta
            #'Santa Ynez River': '11134000',  # Jalama
            'Carmel River': '11143250',  # Not near Carmel City Beach
            #'Salinas River': '11152500',  # Monterey/Marina
            'San Lorenzo River': '11161000',  # Cowell, Santa Cruz beaches
            'Soquel Creek': '11160000',  # Cowell, Santa Cruz beaches
            'Pescadero Creek': '11162500',  # Pescadero
            'Pilarcitos Creek': '11162630',  # Half moon bay
            #'Redwood Creek': '11460151',  # Muir Beach (Marin)
            #'Little River': '11481200'  # HB (Moonstone)
}

print('Flow Data\nDirectory: ' + outfolder)
for key in stations:
    station_name = key
    station_no = stations[key]
    print('\nGrabbing flow data for ' + station_name)
    outfile_raw = station_name.replace(' ', '_') + '_raw_flow_data_' + sd.replace('-', '') + '_' + ed.replace('-', '') + '.csv'
    if outfile_raw in os.listdir(outfolder_raw):
        df = pd.read_csv(os.path.join(outfolder_raw, outfile_raw),
                         index_col=['date'], parse_dates=['date'])
    else:

        # Grab USGS daily discharge data over the specified timeframe
        url = 'http://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=' + station_no + \
              '&referred_module=sw&period=&begin_date=' + sd + '&end_date=' + ed
    
        web = requests.get(url)
        try:
            web.raise_for_status()
        except Exception as exc:
            print('  There was a problem grabbing flow data: %s' % exc)
    
        data = [line.split() for line in web.text.splitlines()]
        while data[0][0].startswith('#'):  # delete comments from list
            del data[0]
    
        df = pd.DataFrame(data, columns=data[0]).drop([0, 1])  # Delete headers
        df = df[list(df.columns[2:])]  # Grab datetime, flow, and qualifier
        df.columns = ['date', 'flow', 'qual']
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        print(' Data found')
    
        # Save raw data
        df.to_csv(os.path.join(outfolder_raw, outfile_raw))
        print(' Raw data saved to ' + outfile_raw)

    ### Calculate variables 
    
    ## logflow1- logflow5
    df_vars = pd.DataFrame(index=df.index)
    df_vars['logflow'] = round(np.log10(df['flow'].astype(float)+1), 2)
    for i in range(1, 6):  # logflow1 - logflow5 (log10 + 1 transform)
        df_vars['logflow' + str(i)] = df_vars['logflow'].shift(i, freq='D')
        #df_vars['logflow' + str(i + 1)][np.isneginf(df_vars['logflow' + str(i+1)])] = round(np.log10(0.005), 5)
    
    # Flow spikes (>XX% percentile flow)
    for i in range(1,4):
        for n in [50, 75, 90]:
            df_vars['flow' + str(i) + '_q' + str(n)] = 1
            df_vars.loc[df_vars['logflow'+str(i)] < np.log10(df.flow.quantile(n/100)+1), 'flow' + str(i) + '_q' + str(n)] = 0
    

    # Save file to directory
    outfile = station_name.replace(' ', '_') + '_flow_variables_' + sd.replace('-', '') + '_' + ed.replace('-', '') + '.csv'
    df_vars.to_csv(os.path.join(outfolder, outfile))
    print('  Flow variables calculated and saved to ' + outfile)

    # # Summary of data
    missing = (pd.to_datetime(ed) - pd.to_datetime(sd)).days - len(df_vars)

    sum_dict = {
        'ID': station_no,
        'Start Date': str(df_vars.index[0].date()),
        'End Date': str(df_vars.index[-1].date()),
        'Missing Days': missing
    }
    df_summary = df_summary.append(pd.DataFrame(sum_dict, index=[station_name]))

df_summary = df_summary[['ID', 'Start Date', 'End Date', 'Missing Days']]
df_summary.index.rename('Station', inplace=True)
df_summary.to_csv(os.path.join(outfolder, 'data_summary.csv'))
