#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:53:40 2022

@author: rtsearcy

Uses the PyEphem packages to calculate days since full moon, and spring/neap 
tide determination
"""

import os
import pandas as pd
import numpy as np
import datetime
import ephem

save_folder = '/Users/rtsearcy/Box/water_quality_modeling/forecasting/data/tide/'

sd = '2000'
ed = '2022'

## Create Date Range
idx = pd.date_range(sd,ed, freq='1D')
df = pd.DataFrame(index=idx)
df['days_since_full'] = np.nan

for d in range(0,len(idx)):
    date=ephem.Date(datetime.date(idx[d].year,idx[d].month,idx[d].day))
    prev_full = ephem.previous_full_moon(date)
    df.loc[idx[d],'days_since_full'] = int(np.floor(date - prev_full))

df['tide_spring'] = 0  ## 0 = neap tide, 1 = spring tide
df.loc[df['days_since_full'].isin([0,1,2,3,12,13,14,15,15,17,18,26,27,28]),'tide_spring'] = 1

df.index.rename('date', inplace=True)
df.to_csv(os.path.join(save_folder,'full_moon.csv'))