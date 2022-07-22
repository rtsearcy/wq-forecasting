# currents_get_hf_radar.py
# RTS 0472022
# Grabs raw HF Radar currents from CenCOOS stations
# Code help from https://github.com/rowg/HFRnet-Thredds-support
# https://github.com/rowg/HFRnet-Thredds-support/blob/master/PythonNotebooks/TimeseriesRTVfromSIO_TDS-CSVout.ipynb


import pandas as pd
from datetime import datetime, timedelta
import time
import os
import netCDF4 as netcdf
import numpy as np
import matplotlib.pyplot as plt

def timeIndexToDatetime(baseTime,times):  # to convert netcdf time format
    newTimes=[]
    for ts in times:
        newTimes.append(baseTime+timedelta(hours=ts))

    return newTimes

### Inputs 
folder = '/Volumes/GoogleDrive/My Drive/water_quality_modeling/forecasting/data/currents/'

# Dates
startTime = "2010-01-01 00:00 UTC"
endTime = "2021-05-01 23:00 UTC"

# Location (center point, will select nearest points)

# # Manhattan Beach
# beach = 'Manhattan'
# site_lat = 33.886
# site_lon = -118.419
# beach_angle = 235   # Angle of the beach

# Huntington Beach
beach = 'Huntington'
site_lat = 33.620431 
site_lon = -117.960616
beach_angle = 225   # Angle of the beach

# # Cowell Beach
# site_lat = 33.886
# site_lon = -118.419

spatial_step = 1 # +/- steps away from center point (e.g. 1 yeilds a 3x3 grid, 2 yeilds a 5x5 grid)

# Data Source (resolution)
# 25h avg, 2km
# note: 2km more consistently available than 1km
#source = 'https://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/25hr/RTV/HFRADAR_US_West_Coast_2km_Resolution_25_Hour_Average_RTV_best.ncd'

# Hourly, 2km
source = 'https://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/USWC/2km/hourly/RTV/HFRADAR_US_West_Coast_2km_Resolution_Hourly_RTV_best.ncd'


### Get data from internet
netcdf_data = netcdf.Dataset(source)

for variable in netcdf_data.variables:
    print(variable,netcdf_data.variables[variable].shape)

## Get Lat/Lon/DT
lat=netcdf_data.variables['lat'][:]
lon=netcdf_data.variables['lon'][:]
time=netcdf_data.variables['time'][:]

## Convert Time data
# set the time base
baseTime=datetime.strptime(netcdf_data.variables['time'].units,"hours since %Y-%m-%d %H:%M:%S.%f %Z")
# Turn time index into timestamps
times=timeIndexToDatetime(baseTime,time)


## Find closest spatial data points
lonIdx=(np.abs(lon-site_lon)).argmin()
latIdx=(np.abs(lat-site_lat)).argmin()

print("\nUser input:",site_lon,site_lon)
print("Closest location in grid:", lon[lonIdx],lat[latIdx])

## TODO Print map of grid of points +/- 1 away from center point
u = []
v = []
end = 0

while end < len(times):
    start = end
    end += 100
    if end > len(times):
        end = len(times)+1
    
    u += list(netcdf_data.variables['u'][start:end,latIdx,lonIdx])
    v += list(netcdf_data.variables['v'][start:end,latIdx,lonIdx])
    print(start,end)
    

### Make into DataFrame
df = pd.DataFrame(index=times)
df.index.name = 'dt'

df['lat'] = lat[latIdx]
df['lon'] = lon[lonIdx]

u = list(u)
v = list(v)
df['u'] = u  # positive east, negative west
df['v'] = v  # positive north, negative south


### Daily mean velocities
df_hourly = df.copy()
df = df.resample('1D').mean()


### Plot U/V TS
plt.figure(figsize=(8,4))

plt.plot(df.u, 'k', label = 'u')
plt.plot(df.v, 'r', label = 'v')

plt.legend()
plt.title(beach + ' - East/North Currents')
plt.ylabel('m/s')
plt.xlabel('')


### Rotate velocity vectors to crosshore and alongshore
'''
Alongshore = + when flow towards wharf (to east)
Crossshore = + when offshore flow
'''

EN = df.u + df.v*1.0j # Create Complex Velocity Vector (EN)
rot_vec = np.cos(beach_angle * np.pi/180) + np.sin(beach_angle * np.pi/180)*1.0j

AC = EN * rot_vec
along = np.real(AC)
cross = -1 * np.imag(AC)

df['along'] = along # positive upshore, negative downshore
df['cross'] = cross # positive onshore, negative offshore


### Plot Cross/Alongshore TS
plt.figure(figsize=(8,4))

plt.plot(df.cross, 'b', label = 'cross')
plt.plot(df.along, 'g', label = 'along')

plt.legend()
plt.title(beach + ' - Alongshore/Crossshore Currents')
plt.ylabel('m/s')
plt.xlabel('')

### Save DF
df.to_csv(os.path.join(folder, beach + '_daily_currents.csv'))
df_hourly.to_csv(os.path.join(folder, beach + '_hourly_currents.csv'))
