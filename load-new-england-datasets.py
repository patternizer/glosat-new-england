#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: load-new-england-datasets.py
#------------------------------------------------------------------------------
# Version 0.11
# 6 July, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import numpy.ma as ma
import itertools
import pandas as pd
import xarray as xr
import pickle
from datetime import datetime
import nc_time_axis
import cftime

# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import cmocean
import seaborn as sns; sns.set()

# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# OS libraries:
import os
import os.path
from pathlib import Path
import sys
import subprocess
from subprocess import Popen
import time

# Stats libraries:
import random
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Maths libraries
import scipy

# Datetime libraries
import cftime
import calendar 
# print(calendar.calendar(2020))
# print(calendar.month(2020,2))
# calendar.isleap(2020)
from datetime import date, time, datetime, timedelta
#today = datetime.now()
#tomorrow = today + pd.to_timedelta(1,unit='D')
#tomorrow = today + timedelta(days=1)
#birthday = datetime(1970,11,1,0,0,0).strftime('%Y-%m-%d %H:%M')
#print('Week:',today.isocalendar()[1])

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16
color_palette = 'viridis_r'
use_fahrenheit = True

load_glosat = True
load_historical_observations = False
load_bho_observations = False
load_neighbouring_stations = False
load_ghcnm = False
load_20crv3 = False
load_hadcrut5 = False
load_cet = False
load_st_lawrence_valley = False
load_amherst2 = False

save_monthly_adjustments = True
save_glosat_adjustments = True

plot_historical = True
plot_differences = True
plot_differences_heatmap = True
plot_kde = True
plot_ghcn = True
plot_inventory = True
plot_glosat_neighbours = True
plot_bho_all_sources = True
plot_glosat_adjusted_vs_neighbours = True
plot_glosat_adjusted_with_back_extension = True
plot_glosat_adjusted_with_back_extension_vs_cet = True
plot_glosat_adjusted_with_back_extension_vs_cet_anomalies = True

if use_fahrenheit: 
    temperature_unit = 'F'
else:
    temperature_unit = 'C'

plot_temp = True

#------------------------------------------------------------------------------
# METHODS: 
#------------------------------------------------------------------------------

def linear_regression_ols(x,y):

    regr = linear_model.LinearRegression()
    # regr = TheilSenRegressor(random_state=42)
    # regr = RANSACRegressor(random_state=42)

    X = x[:, np.newaxis]    
    # X = x.values.reshape(len(x),1)
    t = np.linspace(X.min(),X.max(),len(X)) # dummy var spanning [xmin,xmax]        
    regr.fit(X, y)
    ypred = regr.predict(t.reshape(-1, 1))
    slope = regr.coef_
    intercept = regr.intercept_
    mse = mean_squared_error(y,ypred)
    r2 = r2_score(y,ypred) 
    
    return t, ypred, slope, intercept, mse, r2

def fahrenheit_to_centigrade(x):
    y = (5.0/9.0) * (x - 32.0)
    return y

def centigrade_to_fahrenheit(x):
    y = (x * (9.0/5.0)) + 32.0
    return y

def is_leap_and_29Feb(s):
    return (s.index.year % 4 == 0) & ((s.index.year % 100 != 0) | (s.index.year % 400 == 0)) & (s.index.month == 2) & (s.index.day == 29)

def convert_datetime_to_year_decimal(df, yearstr):

    if yearstr == 'datetime':
        t_monthly_xr = xr.cftime_range(start=str(df.index.year[0]), periods=len(df), freq='MS', calendar='all_leap')
    else:
        t_monthly_xr = xr.cftime_range(start=str(df[yearstr].iloc[0]), periods=len(df)*12, freq='MS', calendar='all_leap')
    year = [t_monthly_xr[i].year for i in range(len(t_monthly_xr))]
    year_frac = []
    for i in range(len(t_monthly_xr)):
        if i%12 == 0:
            istart = i
            iend = istart+11                  
            frac = np.cumsum([t_monthly_xr[istart+j].day for j in range(12)])               
            year_frac += list(frac/frac[-1])            
        else:                
            i += 1
    year_decimal = [float(year[i])+year_frac[i] for i in range(len(year))]    
    return year_decimal
    
#==============================================================================
# LOAD: Datasets
#==============================================================================

if load_glosat == True:
    
    #------------------------------------------------------------------------------    
    # LOAD: GloSAT absolute temperature archive: CRUTEM5.0.1.0
    #------------------------------------------------------------------------------
        
    print('loading temperatures ...')
        
    df_temp = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')    
    
    stationcode_amherst = '720218'
    stationcode_bedford = '720219'
    stationcode_blue_hill = '744920'
    stationcode_boston_city_wso = '725092'
    stationcode_kingston = '753011'
    stationcode_lawrence = '720222'
    stationcode_new_bedford = '720223'
    stationcode_new_haven = '725045'    
    stationcode_plymouth_kingston = '756213'
    stationcode_providence_wso = '725070'
    stationcode_provincetown = '725091'
    stationcode_reading = '725090'
    stationcode_taunton = '720225'
    stationcode_walpole_2 = '744900'
    stationcode_west_medway = '744902'

#   da_salem_cgar = df_temp.columns                                                         # USC00197124	SALEM COAST GUARD AIR STATION, MA US 1948-1967
#   da_salem_b = df_temp.columns                                                            # USC00197122	SALEM B, MA US 1885-1909
#   da_boston = df_temp.columns                                                             # USW00014739	BOSTON, MA US 1936-2021
#   da_new_salem = df_temp.columns                                                          # USC00195306	NEW SALEM, MA US 1897-1998

    da_amherst = df_temp[df_temp['stationcode']==stationcode_amherst]                       # USC00190120	AMHERST, MA US 1893-2021
    da_bedford = df_temp[df_temp['stationcode']==stationcode_bedford]                       # USC00190538	BEDFORD, MA US 1893-1923
    da_blue_hill = df_temp[df_temp['stationcode']==stationcode_blue_hill]                   # USC00190736	BLUE HILL COOP, MA US 1893-2021
    da_boston_city_wso = df_temp[df_temp['stationcode']==stationcode_boston_city_wso]       # USW00094701	BOSTON CITY WEATHER SERVICE OFFICE, MA US 1893-1935
    da_kingston = df_temp[df_temp['stationcode']==stationcode_kingston]                      
    da_lawrence = df_temp[df_temp['stationcode']==stationcode_lawrence]                     # USC00194105	LAWRENCE, MA US 1893-2021
    da_new_bedford = df_temp[df_temp['stationcode']==stationcode_new_bedford]            
    da_new_haven = df_temp[df_temp['stationcode']==stationcode_new_haven]                    
    da_plymouth_kingston = df_temp[df_temp['stationcode']==stationcode_plymouth_kingston]    
    da_providence_wso = df_temp[df_temp['stationcode']==stationcode_providence_wso]         # USC00376712	PROVIDENCE 2, RI US 1893-1913
    da_provincetown = df_temp[df_temp['stationcode']==stationcode_provincetown]             
    da_reading = df_temp[df_temp['stationcode']==stationcode_reading]                       
    da_taunton = df_temp[df_temp['stationcode']==stationcode_taunton]                       
    da_walpole_2 = df_temp[df_temp['stationcode']==stationcode_walpole_2]                   
    da_west_medway = df_temp[df_temp['stationcode']==stationcode_west_medway]               

    ts = np.array(da_amherst.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_amherst.year.iloc[0]), periods=len(ts), freq='MS')
    df_amherst = pd.DataFrame({'amherst':ts}, index=t) 
    ts = np.array(da_bedford.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_bedford.year.iloc[0]), periods=len(ts), freq='MS')
    df_bedford = pd.DataFrame({'bedford':ts}, index=t) 
    ts = np.array(da_blue_hill.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_blue_hill.year.iloc[0]), periods=len(ts), freq='MS')
    df_blue_hill = pd.DataFrame({'blue_hill':ts}, index=t)    
    ts = np.array(da_boston_city_wso.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_boston_city_wso.year.iloc[0]), periods=len(ts), freq='MS')
    df_boston_city_wso = pd.DataFrame({'boston_city_wso':ts}, index=t)        
    ts = np.array(da_kingston.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_kingston.year.iloc[0]), periods=len(ts), freq='MS')
    df_kingston = pd.DataFrame({'kingston':ts}, index=t) 
    ts = np.array(da_lawrence.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_lawrence.year.iloc[0]), periods=len(ts), freq='MS')
    df_lawrence = pd.DataFrame({'lawrence':ts}, index=t) 
    ts = np.array(da_new_bedford.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_new_bedford.year.iloc[0]), periods=len(ts), freq='MS')
    df_new_bedford = pd.DataFrame({'new_bedford':ts}, index=t) 
    ts = np.array(da_new_haven.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_new_haven.year.iloc[0]), periods=len(ts), freq='MS')
    df_new_haven = pd.DataFrame({'new_haven':ts}, index=t)     
    ts = np.array(da_plymouth_kingston.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_plymouth_kingston.year.iloc[0]), periods=len(ts), freq='MS')
    df_plymouth_kingston = pd.DataFrame({'plymouth_kingston':ts}, index=t)     
    ts = np.array(da_providence_wso.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_providence_wso.year.iloc[0]), periods=len(ts), freq='MS')
    df_providence_wso = pd.DataFrame({'providence_wso':ts}, index=t) 
    ts = np.array(da_provincetown.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_provincetown.year.iloc[0]), periods=len(ts), freq='MS')
    df_provincetown = pd.DataFrame({'provincetown':ts}, index=t) 
    ts = np.array(da_reading.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_reading.year.iloc[0]), periods=len(ts), freq='MS')
    df_reading = pd.DataFrame({'reading':ts}, index=t) 
    ts = np.array(da_taunton.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_taunton.year.iloc[0]), periods=len(ts), freq='MS')
    df_taunton = pd.DataFrame({'taunton':ts}, index=t) 
    ts = np.array(da_walpole_2.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_walpole_2.year.iloc[0]), periods=len(ts), freq='MS')
    df_walpole_2 = pd.DataFrame({'walpole_2':ts}, index=t) 
    ts = np.array(da_west_medway.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(da_west_medway.year.iloc[0]), periods=len(ts), freq='MS')
    df_west_medway = pd.DataFrame({'west_medway':ts}, index=t) 

#==============================================================================
    
if load_amherst2 == True:
    
    # LOAD: Amherst 1836-2021 data
        
    df_amherst2 = pd.read_csv('OUT/df_amherst2.csv', index_col=0)
    df_amherst2.index = pd.to_datetime(df_amherst2.index)
    
else:
    
    #-----------------------------------------------------------------------------
    # LOAD: Amherst 1836-2021 (monthly)
    #-----------------------------------------------------------------------------

    # DATA: Phil Jones

    nheader = 0
    f = open('DATA/tmean_Amherst_1836_2021.txt')
    lines = f.readlines()
    years = []
    vals = []
    for i in range(nheader,len(lines)):    
        if i > 0:
            year = lines[i][0:4]
            words = lines[i].split(',')
            val = 12*[None]
            for j in range(len(val)):
                val[j] = float(words[j+1])
            years.append(year)
            vals.append(val) 
    f.close()    
    years = np.array(years).astype('int')
    vals = np.array(vals)
    
    df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = years
    for j in range(12):   
        df[df.columns[j+1]] = [ float(vals[i][j]) for i in range(len(df)) ]
    da_amherst2 = df.replace(-99.9,np.nan)    
        
    ts = np.array(da_amherst2.groupby('year').mean().iloc[:,0:12]).ravel()
    if da_amherst2['year'].iloc[0] > 1678:
        t_monthly = pd.date_range(start=str(da_amherst2['year'].iloc[0]), periods=len(ts), freq='MS')          
    else:
        t_monthly = convert_datetime_to_year_decimal(da_amherst2, 'year')            
    df_amherst2 = pd.DataFrame({'amherst2':ts}, index=t_monthly) 
    df_amherst2.to_csv('df_amherst2.csv')
            
    #==============================================================================
    
if load_st_lawrence_valley == True:
    
    # LOAD: St Lawrence Valley 1742-2020 data
        
    df_st_lawrence_valley = pd.read_csv('OUT/df_st_lawrence_valley.csv', index_col=0)
    df_st_lawrence_valley.index = pd.to_datetime(df_st_lawrence_valley.index)
    
else:
    
    #-----------------------------------------------------------------------------
    # LOAD: St Lawrence Valley (monthly)
    #-----------------------------------------------------------------------------

    # DATA: Victoria Slonosky, Canada
            
    nheader = 0
    f = open('DATA/st-lawrence_valley_monthly_1742-2019_tmean.txt')
    lines = f.readlines()
    years = []
    vals = []
    for i in range(nheader,len(lines)):    
        if i > 0:
            year = lines[i][0:4]
            words = lines[i].split('\t')
            val = 12*[None]
            for j in range(len(val)):
                val[j] = float(words[j+1])
            years.append(year)
            vals.append(val) 
    f.close()    
    years = np.array(years).astype('int')
    vals = np.array(vals)
    
    df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = years
    for j in range(12):   
        df[df.columns[j+1]] = [ float(vals[i][j]) for i in range(len(df)) ]
    da_st_lawrence_valley = df.replace(-999.9,np.nan)    
        
    ts = np.array(da_st_lawrence_valley.groupby('year').mean().iloc[:,0:12]).ravel()
    if da_st_lawrence_valley['year'].iloc[0] > 1678:
        t_monthly = pd.date_range(start=str(da_st_lawrence_valley['year'].iloc[0]), periods=len(ts), freq='MS')          
    else:
        t_monthly = convert_datetime_to_year_decimal(da_st_lawrence_valley, 'year')            
    df_st_lawrence_valley = pd.DataFrame({'df_st_lawrence_valley':ts}, index=t_monthly) 
    df_st_lawrence_valley.to_csv('df_st_lawrence_valley.csv')

#==============================================================================
    
if load_cet == True:
    
    # LOAD: CET (NB: decimal year due to Pandas <1678 and >2262 calendar limit)
        
    df_cet = pd.read_csv('OUT/df_cet.csv', index_col=0)
    
else:
    
    #-----------------------------------------------------------------------------
    # LOAD: CET (monthly)
    #-----------------------------------------------------------------------------

    # DATA: https://www.metoffice.gov.uk/hadobs/hadcet/    
        
    stationcode_cet = '037401'
    da_cet = df_temp[df_temp['stationcode']==stationcode_cet]                       
    ts = np.array(da_cet.groupby('year').mean().iloc[:,0:12]).ravel()
    if da_cet['year'].iloc[0] > 1678:
        t_monthly = pd.date_range(start=str(da_cet['year'].iloc[0]), periods=len(ts), freq='MS')          
    else:
        t_monthly = convert_datetime_to_year_decimal(da_cet, 'year')            
    df_cet = pd.DataFrame({'cet':ts}, index=t_monthly) 
    df_cet.to_csv('df_cet.csv')
                
if load_hadcrut5 == True:
    
    # LOAD: HadCRUT5-analysis (5x5 absolutes)
        
    df_hadcrut5_bho = pd.read_csv('OUT/df_hadcrut5_bho.csv', index_col=0)
    df_hadcrut5_bho.index = pd.to_datetime(df_hadcrut5_bho.index)
    df_hadcrut5_new_haven = pd.read_csv('OUT/df_hadcrut5_new_haven.csv', index_col=0)
    df_hadcrut5_new_haven.index = pd.to_datetime(df_hadcrut5_bho.index)
    
else:
    
    #-----------------------------------------------------------------------------
    # LOAD: HadCRUT5-analysis (monthly) gridded 5x5 absolute temperature (1850-2020)
    #-----------------------------------------------------------------------------

    # Dataset: 

    ds_hadcrut5 = xr.open_dataset('DATA/hadcrut5-analysis-absolutes.nc', decode_cf=True)

    bho_idx_lat = 26
    bho_idx_lon = 21
    new_haven_idx_lat = 26
    new_haven_idx_lon = 21

    # EXTRACT: analysis

    dr_hadcrut5_bho = ds_hadcrut5.tas_mean[:,bho_idx_lat,bho_idx_lon]
    dr_hadcrut5_new_haven = ds_hadcrut5.tas_mean[:,new_haven_idx_lat,new_haven_idx_lon]

    # CLEAR: reanalysis dataframes that are no longer needed

    ds_hadcrut5 = []
    
    # CONSTRUCT: dataframes

    t = dr_hadcrut5_bho.time.values
    df_hadcrut5_bho = pd.DataFrame({'T(2m)':dr_hadcrut5_bho.values}, index=t)      
    df_hadcrut5_bho.index.name = 'datetime'
    df_hadcrut5_bho.to_csv('df_hadcrut5_bho.csv')

    t = dr_hadcrut5_new_haven.time.values
    df_hadcrut5_new_haven = pd.DataFrame({'T(2m)':dr_hadcrut5_new_haven.values}, index=t)      
    df_hadcrut5_new_haven.index.name = 'datetime'
    df_hadcrut5_new_haven.to_csv('df_hadcrut5_new_haven.csv')
        
    plt.plot(df_hadcrut5_bho)
    plt.plot(df_hadcrut5_new_haven)
       
if load_20crv3 == True:

    # LOAD: 20CRv3 for BHO at 2m and at 1000 hPa
        
    df_20CRv3_bho = pd.read_csv('OUT/df_20CRv3_bho.csv', index_col=0)
    df_20CRv3_bho.index = pd.to_datetime(df_20CRv3_bho.index)
    df_20CRv3_new_haven = pd.read_csv('OUT/df_20CRv3_new_haven.csv', index_col=0)
    df_20CRv3_new_haven.index = pd.to_datetime(df_20CRv3_new_haven.index)

else:
    
    #-----------------------------------------------------------------------------
    # LOAD: 20CRv3 (monthly) temperature at 2m and on pressure levels + spread = mean of SDs (1836-2015)
    #-----------------------------------------------------------------------------

    # Dataset: https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.pressure.html#caveat

    ds_20CR_2m = xr.open_dataset('../../BIGFILES/REPOS/glosat-new-england/DATA/air.2m.mon.mean.nc', decode_cf=True)
    ds_20CR_2m_spread = xr.open_dataset('../../BIGFILES/REPOS/glosat-new-england/DATA/air.2m.mon.mean.spread.nc', decode_cf=True)
    ds_20CR_hPa = xr.open_dataset('../../BIGFILES/REPOS/glosat-new-england/DATA/air.mon.mean.nc', decode_cf=True)
    ds_20CR_hPa_spread = xr.open_dataset('../../BIGFILES/REPOS/glosat-new-england/DATA/air.mon.mean.spread.nc', decode_cf=True)
#   ds_20CR_2m_tmin = xr.open_dataset('../../BIGFILES/REPOS/glosat-new-england/DATA/tmin.2m.mon.mean.nc', decode_cf=True)
#   ds_20CR_2m_tmax = xr.open_dataset('../../BIGFILES/REPOS/glosat-new-england/DATA/tmax.2m.mon.mean.nc', decode_cf=True)

    bho_idx_lat = 90+42
    bho_idx_lon = 360-71
    new_haven_idx_lat = 90+41
    new_haven_idx_lon = 360-73

    # EXTRACT: reanalysis at 2m and at 1000 hPa level and convert to degC

    dr_20CR_2m_bho = ds_20CR_2m.air[:,bho_idx_lat,bho_idx_lon]-273.15
    dr_20CR_2m_bho_spread = ds_20CR_2m_spread.air[:,bho_idx_lat,bho_idx_lon]
    dr_20CR_hPa_bho = ds_20CR_hPa.air[:,0,bho_idx_lat,bho_idx_lon]-273.15
    dr_20CR_hPa_bho_spread = ds_20CR_hPa_spread.air[:,0,bho_idx_lat,bho_idx_lon]

    dr_20CR_2m_new_haven = ds_20CR_2m.air[:,new_haven_idx_lat,new_haven_idx_lon]-273.15
    dr_20CR_2m_new_haven_spread = ds_20CR_2m_spread.air[:,new_haven_idx_lat,new_haven_idx_lon]
    dr_20CR_hPa_new_haven = ds_20CR_hPa.air[:,0,new_haven_idx_lat,new_haven_idx_lon]-273.15
    dr_20CR_hPa_new_haven_spread = ds_20CR_hPa_spread.air[:,0,new_haven_idx_lat,new_haven_idx_lon]

    # CLEAR: reanalysis dataframes that are no longer needed

    ds_20CR_2m = []
    ds_20CR_2m_spread = []
    ds_20CR_hPa = []
    ds_20CR_hPa_spread = []
    
    # CONSTRUCT: dataframes

    t = dr_20CR_2m_bho.time.values
    df_20CRv3_bho = pd.DataFrame({
        'T(1000hPa)':dr_20CR_hPa_bho.values, 
        'T(2m)':dr_20CR_2m_bho.values, 
        'T(1000hPa) spread':dr_20CR_hPa_bho_spread.values, 
        'T(2m) spread':dr_20CR_2m_bho_spread.values}, index=t)      
    df_20CRv3_bho.index.name = 'datetime'
    df_20CRv3_bho.to_csv('df_20CRv3_bho.csv')

    t = dr_20CR_2m_new_haven.time.values
    df_20CRv3_new_haven = pd.DataFrame({
        'T(1000hPa)':dr_20CR_hPa_new_haven.values, 
        'T(2m)':dr_20CR_2m_new_haven.values, 
        'T(1000hPa) spread':dr_20CR_hPa_new_haven_spread.values, 
        'T(2m) spread':dr_20CR_2m_new_haven_spread.values}, index=t)      
    df_20CRv3_new_haven.index.name = 'datetime'
    df_20CRv3_new_haven.to_csv('df_20CRv3_new_haven.csv')

#==============================================================================
                      
if load_ghcnm == True:
    
    # LOAD: GHCNM-v4 (QCU and QCF)
        
    df_ghcnmv4_qcu = pd.read_csv('OUT/df_ghcnmv4_qcu.csv', index_col=0)
    df_ghcnmv4_qcu.index = pd.to_datetime(df_ghcnmv4_qcu.index)
    df_ghcnmv4_qcf = pd.read_csv('OUT/df_ghcnmv4_qcf.csv', index_col=0)
    df_ghcnmv4_qcf.index = pd.to_datetime(df_ghcnmv4_qcf.index)
    
else:
    
    #------------------------------------------------------------------------------    
    # LOAD: GHCNM-v4 monthly adjusted and unadjusted data
    #------------------------------------------------------------------------------
        
    print('loading GHCNM-v4 temperatures ...')
    
    nheader = 0
    f = open('DATA/USC00190736-BHO/USC00190736-ghcnm-v4-qcu.dat')
    lines = f.readlines()
    dates = []
    vals = []
    for i in range(nheader,len(lines)):    
        date = lines[i][11:15]
        val = 12*[None]
        for j in range(len(val)):
            val[j] = lines[i][19+(j*8):19+(j*8)+5]
        dates.append(date)
        vals.append(val) 
    f.close()    
    years = np.array(years).astype('int')
    vals = np.array(vals)
    
    df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = dates
    for j in range(1,13):   
        df[df.columns[j]] = [ float(vals[i][j-1])/100.0 for i in range(len(df)) ]
    df_ghcnmv4_qcu = df.replace(-99.99,np.nan)
    
    ts = np.array(df_ghcnmv4_qcu.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(df_ghcnmv4_qcu.year.iloc[0]), periods=len(ts), freq='MS')
    df_ghcnmv4_qcu = pd.DataFrame({'df_ghcnmv4_qcu':ts}, index=t)     
    df_ghcnmv4_qcu.to_csv('df_ghcnmv4_qcu.csv')
    
    # LOAD: GHCNM-v4 (QCF)
    
    nheader = 0
    f = open('DATA/USC00190736-BHO/USC00190736-ghcnm-v4-qcf.dat')
    lines = f.readlines()
    dates = []
    vals = []
    for i in range(nheader,len(lines)):    
        date = lines[i][11:15]
        val = 12*[None]
        for j in range(len(val)):
            val[j] = lines[i][19+(j*8):19+(j*8)+5]
        dates.append(date)
        vals.append(val) 
    f.close()    
    years = np.array(years).astype('int')
    vals = np.array(vals)
    
    df = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df['year'] = dates
    for j in range(1,13):   
        df[df.columns[j]] = [ float(vals[i][j-1])/100.0 for i in range(len(df)) ]
    df_ghcnmv4_qcf = df.replace(-99.99,np.nan)
    
    ts = np.array(df_ghcnmv4_qcf.groupby('year').mean().iloc[:,0:12]).ravel()
    t = pd.date_range(start=str(df_ghcnmv4_qcf.year.iloc[0]), periods=len(ts), freq='MS')
    df_ghcnmv4_qcf = pd.DataFrame({'df_ghcnmv4_qcf':ts}, index=t) 
    df_ghcnmv4_qcf.to_csv('df_ghcnmv4_qcf.csv')
    
#==============================================================================
            
if load_historical_observations == True:
    
    # LOAD: Holyoke
    
    df_holyoke = pd.read_csv('OUT/df_holyoke.csv', index_col=0)
    df_holyoke.index = pd.to_datetime(df_holyoke.index)

    # LOAD: Wigglesworth

    df_wigglesworth = pd.read_csv('OUT/df_wigglesworth.csv', index_col=0)
    df_wigglesworth.index = pd.to_datetime(df_wigglesworth.index)

    # LOAD: Farrar (CRUTEM format)

    df_farrar = pd.read_csv('OUT/df_farrar.csv', index_col=0)
    df_farrar.index = pd.to_datetime(df_farrar.index)
               
else:

    #------------------------------------------------------------------------------
    # LOAD: Holyoke observations into dataframe
    #------------------------------------------------------------------------------
       
    nheader = 0
    f = open('DATA/holyoke.temperature.dat')
    lines = f.readlines()
    years = []
    
    T08_01 = []
    T08_02 = []
    T08_03 = []
    T08_04 = []
    T08_05 = []
    T08_06 = []
    T08_07 = []
    T08_08 = []
    T08_09 = []
    T08_10 = []
    T08_11 = []
    T08_12 = []
    
    T13_01 = []
    T13_02 = []
    T13_03 = []
    T13_04 = []
    T13_05 = []
    T13_06 = []
    T13_07 = []
    T13_08 = []
    T13_09 = []
    T13_10 = []
    T13_11 = []
    T13_12 = []
    
    T22_01 = []
    T22_02 = []
    T22_03 = []
    T22_04 = []
    T22_05 = []
    T22_06 = []
    T22_07 = []
    T22_08 = []
    T22_09 = []
    T22_10 = []
    T22_11 = []
    T22_12 = []
    
    Tsunset_01 = []
    Tsunset_02 = []
    Tsunset_03 = []
    Tsunset_04 = []
    Tsunset_05 = []
    Tsunset_06 = []
    Tsunset_07 = []
    Tsunset_08 = []
    Tsunset_09 = []
    Tsunset_10 = []
    Tsunset_11 = []
    Tsunset_12 = []
    
    season_counter = 0
    day_counter = 0
    
    for i in range(nheader,len(lines)):
        words = lines[i].split()
        if len(words) == 1:
            year = int(words[0])
            continue
        elif len(words) == 3:
            season_counter += 1
            continue
        if season_counter == 1:
            T08_01.append(words[1]) 
            T08_02.append(words[5]) 
            T08_03.append(words[9])
            T13_01.append(words[2]) 
            T13_02.append(words[6]) 
            T13_03.append(words[10])
            Tsunset_01.append(words[3]) 
            Tsunset_02.append(words[7]) 
            Tsunset_03.append(words[11])
            T22_01.append(words[4]) 
            T22_02.append(words[8]) 
            T22_03.append(words[12])
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif season_counter == 2:
            T08_04.append(words[1]) 
            T08_05.append(words[5]) 
            T08_06.append(words[9])
            T13_04.append(words[2]) 
            T13_05.append(words[6]) 
            T13_06.append(words[10])
            Tsunset_04.append(words[3]) 
            Tsunset_05.append(words[7]) 
            Tsunset_06.append(words[11])
            T22_04.append(words[4]) 
            T22_05.append(words[8]) 
            T22_06.append(words[12])
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif season_counter == 3:
            T08_07.append(words[1]) 
            T08_08.append(words[5]) 
            T08_09.append(words[9])
            T13_07.append(words[2]) 
            T13_08.append(words[6]) 
            T13_09.append(words[10])
            Tsunset_07.append(words[3]) 
            Tsunset_08.append(words[7]) 
            Tsunset_09.append(words[11])
            T22_07.append(words[4]) 
            T22_08.append(words[8]) 
            T22_09.append(words[12])
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif season_counter == 4:
            T08_10.append(words[1]) 
            T08_11.append(words[5]) 
            T08_12.append(words[9])
            T13_10.append(words[2]) 
            T13_11.append(words[6]) 
            T13_12.append(words[10])
            Tsunset_10.append(words[3]) 
            Tsunset_11.append(words[7]) 
            Tsunset_12.append(words[11])
            T22_10.append(words[4]) 
            T22_11.append(words[8]) 
            T22_12.append(words[12])
    
            years.append(year)
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
                season_counter = 0
           
    f.close()
    
    # CREATE: mask for 31-day month year for whole timeseries
    
    Nyears = len(np.unique(years))        
    monthdays = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    monthdiff = monthdays-31
    mask_372 = (12*31)*[True]
    for i in range(len(monthdiff)):
        mask_372[ (i+1)*31+monthdiff[i] : (i+1)*31 ] = np.abs( monthdiff[i] )*[False]
    mask = np.tile(mask_372, Nyears)
    
    # CONSTRUCT: timeseries
    
    T08 = []
    T13 = []
    Tsunset = []
    T22 = []
    for i in range(Nyears):
        T08 = T08 + list(T08_01[(i*31):(i+1)*31])
        T08 = T08 + list(T08_02[(i*31):(i+1)*31])
        T08 = T08 + list(T08_03[(i*31):(i+1)*31])
        T08 = T08 + list(T08_04[(i*31):(i+1)*31])
        T08 = T08 + list(T08_05[(i*31):(i+1)*31])
        T08 = T08 + list(T08_06[(i*31):(i+1)*31])
        T08 = T08 + list(T08_07[(i*31):(i+1)*31])
        T08 = T08 + list(T08_08[(i*31):(i+1)*31])
        T08 = T08 + list(T08_09[(i*31):(i+1)*31])
        T08 = T08 + list(T08_10[(i*31):(i+1)*31])
        T08 = T08 + list(T08_11[(i*31):(i+1)*31])
        T08 = T08 + list(T08_12[(i*31):(i+1)*31])
    
        T13 = T13 + list(T13_01[(i*31):(i+1)*31])
        T13 = T13 + list(T13_02[(i*31):(i+1)*31])
        T13 = T13 + list(T13_03[(i*31):(i+1)*31])
        T13 = T13 + list(T13_04[(i*31):(i+1)*31])
        T13 = T13 + list(T13_05[(i*31):(i+1)*31])
        T13 = T13 + list(T13_06[(i*31):(i+1)*31])
        T13 = T13 + list(T13_07[(i*31):(i+1)*31])
        T13 = T13 + list(T13_08[(i*31):(i+1)*31])
        T13 = T13 + list(T13_09[(i*31):(i+1)*31])
        T13 = T13 + list(T13_10[(i*31):(i+1)*31])
        T13 = T13 + list(T13_11[(i*31):(i+1)*31])
        T13 = T13 + list(T13_12[(i*31):(i+1)*31])
    
        Tsunset = Tsunset + list(Tsunset_01[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_02[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_03[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_04[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_05[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_06[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_07[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_08[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_09[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_10[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_11[(i*31):(i+1)*31])
        Tsunset = Tsunset + list(Tsunset_12[(i*31):(i+1)*31])
        
        T22 = T22 + list(T22_01[(i*31):(i+1)*31])
        T22 = T22 + list(T22_02[(i*31):(i+1)*31])
        T22 = T22 + list(T22_03[(i*31):(i+1)*31])
        T22 = T22 + list(T22_04[(i*31):(i+1)*31])
        T22 = T22 + list(T22_05[(i*31):(i+1)*31])
        T22 = T22 + list(T22_06[(i*31):(i+1)*31])
        T22 = T22 + list(T22_07[(i*31):(i+1)*31])
        T22 = T22 + list(T22_08[(i*31):(i+1)*31])
        T22 = T22 + list(T22_09[(i*31):(i+1)*31])
        T22 = T22 + list(T22_10[(i*31):(i+1)*31])
        T22 = T22 + list(T22_11[(i*31):(i+1)*31])
        T22 = T22 + list(T22_12[(i*31):(i+1)*31])
    
    T08_365 = np.array( [ float(T08[i]) for i in range(len(mask))] )[mask]
    T13_365 = np.array( [ float(T13[i]) for i in range(len(mask))] )[mask]
    Tsunset_365 = np.array( [ float(Tsunset[i]) for i in range(len(mask))] )[mask]
    T22_365 = np.array( [ float(T22[i]) for i in range(len(mask))] )[mask]
    
    # REPLACE: fill value = -99.00 with np.nan
    
    T08_365[T08_365==-99.00] = np.nan
    T13_365[T13_365==-99.00] = np.nan
    Tsunset_365[Tsunset_365==-99.00] = np.nan
    T22_365[T22_365==-99.00] = np.nan
        
    t = pd.date_range(start=str(years[0]), periods=len(T08_365), freq='D')
            
    # CONSTRUCT: dataframe & save
    
    df_holyoke = pd.DataFrame({'T(08:00)':T08_365, 'T(13:00)':T13_365, 'T(22:00)':T22_365, 'T(sunset)':Tsunset_365}, index=t)
    df_holyoke.to_csv('df_holyoke.csv')

    #------------------------------------------------------------------------------
    # LOAD: Wigglesworth observations into dataframe
    #------------------------------------------------------------------------------
    
    nheader = 0
    f = open('DATA/wigglesworth.temperature.dat')
    lines = f.readlines()
    years = []
    
    T08_01 = []
    T08_02 = []
    T08_03 = []
    T08_04 = []
    T08_05 = []
    T08_06 = []
    T08_07 = []
    T08_08 = []
    T08_09 = []
    T08_10 = []
    T08_11 = []
    T08_12 = []
    
    T13_01 = []
    T13_02 = []
    T13_03 = []
    T13_04 = []
    T13_05 = []
    T13_06 = []
    T13_07 = []
    T13_08 = []
    T13_09 = []
    T13_10 = []
    T13_11 = []
    T13_12 = []
    
    T21_01 = []
    T21_02 = []
    T21_03 = []
    T21_04 = []
    T21_05 = []
    T21_06 = []
    T21_07 = []
    T21_08 = []
    T21_09 = []
    T21_10 = []
    T21_11 = []
    T21_12 = []
    
    month_counter = 0
    day_counter = 0
    
    for i in range(nheader,len(lines)):
        words = lines[i].split()
        if len(words) == 1:
            if words[0].isnumeric():
                year = int(words[0])
            else:     
                month_counter += 1
            continue
        if month_counter == 1:
            T08_01.append(words[4]) 
            T13_01.append(words[5]) 
            T21_01.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 2:
            T08_02.append(words[4]) 
            T13_02.append(words[5]) 
            T21_02.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 3:
            T08_03.append(words[4]) 
            T13_03.append(words[5]) 
            T21_03.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 4:
            T08_04.append(words[4]) 
            T13_04.append(words[5]) 
            T21_04.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 5:
            T08_05.append(words[4]) 
            T13_05.append(words[5]) 
            T21_05.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 6:
            T08_06.append(words[4]) 
            T13_06.append(words[5]) 
            T21_06.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 7:
            T08_07.append(words[4]) 
            T13_07.append(words[5]) 
            T21_07.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 8:
            T08_08.append(words[4]) 
            T13_08.append(words[5]) 
            T21_08.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 9:
            T08_09.append(words[4]) 
            T13_09.append(words[5]) 
            T21_09.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 10:
            T08_10.append(words[4]) 
            T13_10.append(words[5]) 
            T21_10.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 11:
            T08_11.append(words[4]) 
            T13_11.append(words[5]) 
            T21_11.append(words[6]) 
    
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
    
        elif month_counter == 12:
            T08_12.append(words[4]) 
            T13_12.append(words[5]) 
            T21_12.append(words[6]) 
    
            years.append(year)
            day_counter += 1        
            if day_counter == 31:
                day_counter = 0        
                month_counter = 0
           
    f.close()
    
    # CREATE: mask for 31-day month year for whole timeseries
    
    Nyears = len(np.unique(years))        
    monthdays = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    monthdiff = monthdays-31
    mask_372 = (12*31)*[True]
    for i in range(len(monthdiff)):
        mask_372[ (i+1)*31+monthdiff[i] : (i+1)*31 ] = np.abs( monthdiff[i] )*[False]
    mask = np.tile(mask_372, Nyears)
    
    # CONSTRUCT: timeseries
    
    T08 = []
    T13 = []
    T21 = []
    for i in range(Nyears):
        T08 = T08 + list(T08_01[(i*31):(i+1)*31])
        T08 = T08 + list(T08_02[(i*31):(i+1)*31])
        T08 = T08 + list(T08_03[(i*31):(i+1)*31])
        T08 = T08 + list(T08_04[(i*31):(i+1)*31])
        T08 = T08 + list(T08_05[(i*31):(i+1)*31])
        T08 = T08 + list(T08_06[(i*31):(i+1)*31])
        T08 = T08 + list(T08_07[(i*31):(i+1)*31])
        T08 = T08 + list(T08_08[(i*31):(i+1)*31])
        T08 = T08 + list(T08_09[(i*31):(i+1)*31])
        T08 = T08 + list(T08_10[(i*31):(i+1)*31])
        T08 = T08 + list(T08_11[(i*31):(i+1)*31])
        T08 = T08 + list(T08_12[(i*31):(i+1)*31])
    
        T13 = T13 + list(T13_01[(i*31):(i+1)*31])
        T13 = T13 + list(T13_02[(i*31):(i+1)*31])
        T13 = T13 + list(T13_03[(i*31):(i+1)*31])
        T13 = T13 + list(T13_04[(i*31):(i+1)*31])
        T13 = T13 + list(T13_05[(i*31):(i+1)*31])
        T13 = T13 + list(T13_06[(i*31):(i+1)*31])
        T13 = T13 + list(T13_07[(i*31):(i+1)*31])
        T13 = T13 + list(T13_08[(i*31):(i+1)*31])
        T13 = T13 + list(T13_09[(i*31):(i+1)*31])
        T13 = T13 + list(T13_10[(i*31):(i+1)*31])
        T13 = T13 + list(T13_11[(i*31):(i+1)*31])
        T13 = T13 + list(T13_12[(i*31):(i+1)*31])
        
        T21 = T21 + list(T21_01[(i*31):(i+1)*31])
        T21 = T21 + list(T21_02[(i*31):(i+1)*31])
        T21 = T21 + list(T21_03[(i*31):(i+1)*31])
        T21 = T21 + list(T21_04[(i*31):(i+1)*31])
        T21 = T21 + list(T21_05[(i*31):(i+1)*31])
        T21 = T21 + list(T21_06[(i*31):(i+1)*31])
        T21 = T21 + list(T21_07[(i*31):(i+1)*31])
        T21 = T21 + list(T21_08[(i*31):(i+1)*31])
        T21 = T21 + list(T21_09[(i*31):(i+1)*31])
        T21 = T21 + list(T21_10[(i*31):(i+1)*31])
        T21 = T21 + list(T21_11[(i*31):(i+1)*31])
        T21 = T21 + list(T21_12[(i*31):(i+1)*31])
    
    T08_365 = np.array( [ float(T08[i]) for i in range(len(mask))] )[mask]
    T13_365 = np.array( [ float(T13[i]) for i in range(len(mask))] )[mask]
    T21_365 = np.array( [ float(T21[i]) for i in range(len(mask))] )[mask]
    
    # REPLACE: fill value = -99.00 with np.nan
    
    T08_365[T08_365==-99.9] = np.nan
    T13_365[T13_365==-99.9] = np.nan
    T21_365[T21_365==-99.9] = np.nan
        
    t = pd.date_range(start=str(years[0]), periods=len(T08_365), freq='D')
            
    # CONSTRUCT: dataframe & save
    
    df_wigglesworth = pd.DataFrame({'T(08:00)':T08_365, 'T(13:00)':T13_365, 'T(21:00)':T21_365}, index=t)
    df_wigglesworth.to_csv('df_wigglesworth.csv')

    #------------------------------------------------------------------------------
    # LOAD: Farrar observations into dataframe
    #------------------------------------------------------------------------------

    da = pd.read_csv('DATA/farrar.dat', index_col=0) # KEYED-IN by MT from the American Almanac, 1837 
    ts_monthly = []    
    for i in range(len(da)):                
        monthly = da.iloc[i,0:]
        ts_monthly = ts_monthly + monthly.to_list()    
    ts_monthly = np.array(ts_monthly)   
    ts_monthly = fahrenheit_to_centigrade(ts_monthly)   
    t_monthly = pd.date_range(start=str(da.index[0]), periods=len(ts_monthly), freq='MS')    
    df_farrar = pd.DataFrame({'Tmean':ts_monthly}, index=t_monthly)
    df_farrar.index.name = 'datetime'
    df_farrar.to_csv('df_farrar.csv')

#==============================================================================
    
if load_bho_observations == True:       
    
    df_bho_2828 = pd.read_csv('OUT/df_bho_2828.csv', index_col=0)
    df_bho_tg = pd.read_csv('OUT/df_bho_tg.csv', index_col=0)
    df_bho_daily = pd.read_csv('OUT/df_bho_daily.csv', index_col=0)
    df_bho_monthly = pd.read_csv('OUT/df_bho_monthly.csv', index_col=0)

    df_bho_2828.index = pd.to_datetime(df_bho_2828.index)
    df_bho_tg.index = pd.to_datetime(df_bho_tg.index)
            
else:
    
    #------------------------------------------------------------------------------
    # LOAD: BHO data from EXCEL
    #------------------------------------------------------------------------------
    
    da = pd.read_table('DATA/bho-2828-degF.dat', index_col=0) # '2828' monthly average       
    ts_monthly = []    
    for i in range(len(da)):                
        monthly = da.iloc[i,0:]
        ts_monthly = ts_monthly + monthly.to_list()    
    ts_monthly = np.array(ts_monthly)   
    t_monthly = pd.date_range(start=str(da.index[0]), periods=len(ts_monthly), freq='MS')    
    df_bho_2828 = pd.DataFrame({'T2828':ts_monthly}, index=t_monthly)
    df_bho_2828.index.name = 'datetime'

#   da = pd.read_table('DATA/bho-tg.dat', index_col=0) # Tg monthly average
    da = pd.read_table('DATA/bho-tg-degF.dat', index_col=0) # Tg monthly average
    ts_monthly = []    
    for i in range(len(da)):                
        monthly = da.iloc[i,0:]
        ts_monthly = ts_monthly + monthly.to_list()    
    ts_monthly = np.array(ts_monthly)   
    t_monthly = pd.date_range(start=str(da.index[0]), periods=len(ts_monthly), freq='MS')    
    df_bho_tg = pd.DataFrame({'Tg':ts_monthly}, index=t_monthly)
    df_bho_tg.index.name = 'datetime'

    da = pd.read_table('DATA/bho-max_01.dat', index_col=0) # daily
    for i in range(2,13):
        db = pd.read_table('DATA/bho-max_'+str(i).zfill(2)+'.dat', index_col=0) # daily 
        da = pd.concat([da,db],axis=1)    
    t = []        
    ts = []    
    for i in range(len(da)):                
        daily = da.iloc[i,0:]
        ts = ts + daily.to_list()    
    ts = np.array(ts)   

    # HANDLE: leap days

    t = xr.cftime_range(start=str(da.index[0])+'-01-01', periods=len(ts), freq='D', calendar="all_leap")   
    df_bho_tmax = pd.DataFrame({'Tmax':ts}, index=t)
    df_bho_tmax.index.name = 'datetime'
    
    da = pd.read_table('DATA/bho-min_01.dat', index_col=0) # daily
    for i in range(2,13):
        db = pd.read_table('DATA/bho-min_'+str(i).zfill(2)+'.dat', index_col=0) # daily 
        da = pd.concat([da,db],axis=1)        
    ts = []    
    for i in range(len(da)):                
        daily = da.iloc[i,0:]
        ts = ts + daily.to_list()    
    ts = np.array(ts)   

    # HANDLE: leap days
 
    t = xr.cftime_range(start=str(da.index[0])+'-01-01', periods=len(ts), freq='D', calendar="all_leap")   
    df_bho_tmin = pd.DataFrame({'Tmin':ts}, index=t)
    df_bho_tmin.index.name = 'datetime'

    # CALCULATE: Tg=(Tn+Tx)/2 and resample to monthly (and trim to TS end)
    
    df_bho_daily = pd.DataFrame({'Tmin':df_bho_tmin['Tmin'],'Tmax':df_bho_tmax['Tmax']},index=t)
    df_bho_daily['Tg'] = (df_bho_daily['Tmin']+df_bho_daily['Tmax'])/2.      

    # RESAMPLE: using xarray

    df_bho_daily_xr = df_bho_daily.to_xarray()    
    df_bho_daily_xr_resampled_Tmin = df_bho_daily_xr.Tmin.resample(datetime='MS').mean().to_dataset()    
    df_bho_daily_xr_resampled_Tmax = df_bho_daily_xr.Tmax.resample(datetime='MS').mean().to_dataset()    
    df_bho_daily_xr_resampled_Tg = df_bho_daily_xr.Tg.resample(datetime='MS').mean().to_dataset()    
    df_bho_monthly = df_bho_tg.copy()     
    df_bho_monthly['Tmin'] = df_bho_daily_xr_resampled_Tmin.Tmin.values
    df_bho_monthly['Tmax'] = df_bho_daily_xr_resampled_Tmax.Tmax.values
    df_bho_monthly['Tgm'] = df_bho_daily_xr_resampled_Tg.Tg.values
    
    # SAVE: dataframes

    df_bho_2828.to_csv('df_bho_2828.csv')
    df_bho_tg.to_csv('df_bho_tg.csv')
    df_bho_daily.to_csv('df_bho_daily.csv')
    df_bho_monthly.to_csv('df_bho_monthly.csv')
    
#==============================================================================
    
if load_neighbouring_stations == True:
          
    df_neighbouring_stations = pd.read_csv('OUT/df_neighbouring_stations.csv', index_col=0)
    df_neighbouring_stations.index = pd.to_datetime(df_neighbouring_stations.index)   

else:
    
    #------------------------------------------------------------------------------
    # LOAD: GHCND neighbouring station datasets (daily TMIN, TMAX ,TAVG)
    #------------------------------------------------------------------------------
    
    # NOAA NCEI LCD station group 1 
    #------------------------------
    #USC00197124	SALEM COAST GUARD AIR STATION, MA US
    #USC00197122	SALEM B, MA US
    #USW00014739	BOSTON, MA US
    #USC00195306	NEW SALEM, MA US
    #USW00094701	BOSTON CITY WEATHER SERVICE OFFICE, MA US
    
    # NOAA NCEI LCD station group 2
    #------------------------------
    #USC00190736	BLUE HILL COOP, MA US
    #USC00190538	BEDFORD, MA US
    #USC00194105	LAWRENCE, MA US
    #USC00190120	AMHERST, MA US
    #USC00376712	PROVIDENCE 2, RI US
    
    # NOAA NCEI LCD station group 3
    #------------------------------
    #USC00199928	WORCESTER, MA US

    # NOAA NCEI LCD station group 4
    #------------------------------
    #USC00065910	NORWICH, CT US

    # NOAA NCEI LCD station group 5
    #------------------------------
    #USW00014765	PROVIDENCE, RI US
    #USW00014758    NEW HAVEN TWEED AIRPORT, CT US
        
    df1 = pd.read_csv('DATA/2606266.csv') # NOAA NCEI LCD station group 1
    df2 = pd.read_csv('DATA/2606305.csv') # NOAA NCEI LCD station group 2
    df3 = pd.read_csv('DATA/2606326.csv') # NOAA NCEI LCD station group 3
    df4 = pd.read_csv('DATA/2629709.csv') # NOAA NCEI LCD station group 4
    df5 = pd.read_csv('DATA/2629892.csv') # NOAA NCEI LCD station group 5
    df = pd.concat([df1,df2,df3,df4,df5])
        
    uniquestations = df['STATION'].unique() 
    uniquenames = df['NAME'].unique() 
    Nstations = len(uniquestations)
    
    df_reordered = df.sort_index().reset_index(drop=True)
    datetimes = [ pd.to_datetime(df_reordered['DATE'][i]) for i in range(len(df_reordered)) ] 
    df_reordered['datetime'] = datetimes
    df_reordered['TAVG'] = (df_reordered['TMIN'] + df_reordered['TMAX'] )/2.
    df_reordered.index = df_reordered['datetime']
    del df_reordered['TAVG_ATTRIBUTES']
    del df_reordered['TMAX_ATTRIBUTES']
    del df_reordered['TMIN_ATTRIBUTES']
    del df_reordered['TOBS_ATTRIBUTES']
    del df_reordered['DATE']
    del df_reordered['datetime']
    
    df_neighbouring_stations = pd.DataFrame(columns = df_reordered.columns)    
    for i in range(Nstations):        
        stationcode = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['STATION']
        stationname = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['NAME']
        stationlat = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['LATITUDE']
        stationlon = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['LONGITUDE']
        stationelevation = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['ELEVATION']
        xmin = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['TMIN']
        xmax = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['TMAX']
        xavg = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['TAVG']    
        t = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]].index
        if ((len(t)/365) < 20.0):
             continue
        if np.nanmean(xmin) > 20: # Fahrenheit detection in Boston
            ymin = fahrenheit_to_centigrade(xmin)   
            ymax = fahrenheit_to_centigrade(xmax)           
            yavg = fahrenheit_to_centigrade(xavg)           
        else:
            ymin = xmin
            ymax = xmax
            yavg = xavg                        
        df_station = pd.DataFrame(columns = df_reordered.columns, index=t)    
        df_station['TMIN'] = ymin
        df_station['TMAX'] = ymax
        df_station['TAVG'] = yavg
        df_station['STATION'] = stationcode
        df_station['NAME'] = stationname
        df_station['LATITUDE'] = stationlat
        df_station['LONGITUDE'] = stationlon
        df_station['ELEVATION'] = stationelevation    
        
        # WRITE: station data to file
    
        df_station.to_csv(stationcode.unique()[0]+'.csv')
        
        # ADD: station data to neighbouring stations dataframe
    
    #    df_neighbouring_stations = pd.concat([df_neighbouring_stations,df_station], axis=0).reset_index(drop=True)
        df_neighbouring_stations = pd.concat([df_neighbouring_stations,df_station], axis=0)
    
    df_neighbouring_stations.index.name = 'datetime'
    df_neighbouring_stations.to_csv('df_neighbouring_stations.csv')

df_neighbouring_stations = df_neighbouring_stations.sort_index()
Nstations = df_neighbouring_stations.groupby('STATION').count().shape[0]
stationcodes = df_neighbouring_stations['STATION'].unique()

# SPLIT: into individual station dataframes: df_STATIONCODE and resample at monthly timescale

# ['USC00199928', 'USW00094701', 'USC00190736', 'USC00194105',
#  'USC00190120', 'USC00376712', 'USC00190538', 'USC00195306',
#  'USW00014739', 'USW00014765', 'USW00014758', 'USC00065910'],
            
for i in range(len(stationcodes)):    
        
    Tg = (df_neighbouring_stations[df_neighbouring_stations['STATION']==stationcodes[i]]['TAVG']).sort_index()
    Tg = pd.DataFrame({'Tg':Tg.values}, index=Tg.index)
    Tg_xr = Tg.to_xarray()    
    Tgm = Tg_xr['Tg'].resample(datetime='MS').mean().to_dataset() 
    globals()['df_' + stationcodes[i]] = pd.DataFrame({'Tgm':Tgm.Tg.values}, index=Tgm.datetime.values)

#==============================================================================

#------------------------------------------------------------------------------
# CONVERT: to Fahrenheit
#------------------------------------------------------------------------------

if use_fahrenheit == True:

    df_holyoke = centigrade_to_fahrenheit( df_holyoke )        
    df_wigglesworth = centigrade_to_fahrenheit( df_wigglesworth )        
    df_farrar = centigrade_to_fahrenheit( df_farrar )        
    
    df_amherst = pd.DataFrame({'amherst':centigrade_to_fahrenheit( df_amherst['amherst'] )})        
    df_bedford = pd.DataFrame({'bedford':centigrade_to_fahrenheit( df_bedford['bedford'] )})        
    df_blue_hill = pd.DataFrame({'blue_hill':centigrade_to_fahrenheit( df_blue_hill['blue_hill'] )})      
    df_boston_city_wso = pd.DataFrame({'boston_city_wso':centigrade_to_fahrenheit( df_boston_city_wso['boston_city_wso'] )})      
    df_lawrence = pd.DataFrame({'lawrence':centigrade_to_fahrenheit( df_lawrence['lawrence'] )})    
    df_kingston = pd.DataFrame({'kingston':centigrade_to_fahrenheit( df_kingston['kingston'] )})        
    df_new_bedford = pd.DataFrame({'new_bedford':centigrade_to_fahrenheit( df_new_bedford['new_bedford'] )})        
    df_new_haven = pd.DataFrame({'new_haven':centigrade_to_fahrenheit( df_new_haven['new_haven'] )})        
    df_plymouth_kingston = pd.DataFrame({'plymouth_kingston':centigrade_to_fahrenheit( df_plymouth_kingston['plymouth_kingston'] )})            
    df_providence_wso = pd.DataFrame({'providence_wso':centigrade_to_fahrenheit( df_providence_wso['providence_wso'] )})        
    df_provincetown = pd.DataFrame({'provincetown':centigrade_to_fahrenheit( df_provincetown['provincetown'] )})        
    df_reading = pd.DataFrame({'reading':centigrade_to_fahrenheit( df_reading['reading'] )})        
    df_taunton = pd.DataFrame({'taunton':centigrade_to_fahrenheit( df_taunton['taunton'] )})        
    df_walpole_2 = pd.DataFrame({'walpole_2':centigrade_to_fahrenheit( df_walpole_2['walpole_2'] )})       
    df_west_medway = pd.DataFrame({'west_medway':centigrade_to_fahrenheit( df_west_medway['west_medway'] )})        

    df_USC00199928 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USC00199928['Tgm'] )}) # WORCESTER, MA
    df_USW00094701 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USW00094701['Tgm'] )}) # BOSTON CITY WSO, MA
    df_USC00190736 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USC00190736['Tgm'] )}) # BLUE HILL, MA
    df_USC00194105 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USC00194105['Tgm'] )}) # LAWRENCE, MA
    df_USC00190120 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USC00190120['Tgm'] )}) # AMHERST, MA
    df_USC00376712 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USC00376712['Tgm'] )}) # PROVIDENCE 2, RI
    df_USC00190538 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USC00190538['Tgm'] )}) # BEDFORD, MA
    df_USC00195306 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USC00195306['Tgm'] )}) # NEW SALEM, MA
    df_USW00014739 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USW00014739['Tgm'] )}) # BOSTON, MA
    df_USW00014765 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USW00014765['Tgm'] )}) # PROVIDENCE, RI
    df_USW00014758 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USW00014758['Tgm'] )}) # NEW HAVEN TWEED AIRPORT, CT
    df_USC00065910 = pd.DataFrame({'Tgm':centigrade_to_fahrenheit( df_USC00065910['Tgm'] )}) # NORWICH, CT

    df_ghcnmv4_qcf = pd.DataFrame({'df_ghcnmv4_qcf':centigrade_to_fahrenheit( df_ghcnmv4_qcf['df_ghcnmv4_qcf'] )})        
    df_ghcnmv4_qcu = pd.DataFrame({'df_ghcnmv4_qcu':centigrade_to_fahrenheit( df_ghcnmv4_qcu['df_ghcnmv4_qcu'] )})     
    
    df_20CRv3_new_haven = pd.DataFrame({
        'T(2m)':centigrade_to_fahrenheit( df_20CRv3_new_haven['T(2m)'] ),
        'T(1000hPa)':centigrade_to_fahrenheit( df_20CRv3_new_haven['T(1000hPa)'] ),
        'T(2m) spread':centigrade_to_fahrenheit( df_20CRv3_new_haven['T(2m) spread'] ) - 32.0,
        'T(1000hPa) spread':centigrade_to_fahrenheit( df_20CRv3_new_haven['T(1000hPa) spread'] ) - 32.0,
        })
    df_20CRv3_bho = pd.DataFrame({
        'T(2m)':centigrade_to_fahrenheit( df_20CRv3_bho['T(2m)'] ),
        'T(1000hPa)':centigrade_to_fahrenheit( df_20CRv3_bho['T(1000hPa)'] ),
        'T(2m) spread':centigrade_to_fahrenheit( df_20CRv3_bho['T(2m) spread'] ) - 32.0,
        'T(1000hPa) spread':centigrade_to_fahrenheit( df_20CRv3_bho['T(1000hPa) spread'] ) - 32.0
        })

    df_hadcrut5_bho = pd.DataFrame({'df_hadcrut5_bho':centigrade_to_fahrenheit( df_hadcrut5_bho['T(2m)'] )})        
    df_hadcrut5_new_haven = pd.DataFrame({'df_hadcrut5_new_haven':centigrade_to_fahrenheit( df_hadcrut5_new_haven['T(2m)'] )})     
    df_cet = pd.DataFrame({'df_cet':centigrade_to_fahrenheit( df_cet['cet'] )})        
    df_st_lawrence_valley = pd.DataFrame({'df_st_lawrence_valley':centigrade_to_fahrenheit( df_st_lawrence_valley['df_st_lawrence_valley'] )})        
    
else:
    
    df_bho_2828 = pd.DataFrame({'T2828':fahrenheit_to_centigrade( df_bho_2828['T2828'] )})   
    df_bho_tg = pd.DataFrame({'Tg':fahrenheit_to_centigrade( df_bho_tg['Tg'] )})   
    df_bho_daily = pd.DataFrame({
        'Tmin':fahrenheit_to_centigrade( df_bho_daily['Tmin'] ),
        'Tmax':fahrenheit_to_centigrade( df_bho_daily['Tmax'] ),        
        'Tg':fahrenheit_to_centigrade( df_bho_daily['Tg'] )
        })   
    df_bho_monthly = pd.DataFrame({
        'Tg':fahrenheit_to_centigrade( df_bho_monthly['Tg'] ),
        'Tmin':fahrenheit_to_centigrade( df_bho_monthly['Tmin'] ),
        'Tmax':fahrenheit_to_centigrade( df_bho_monthly['Tmax'] ),        
        'Tgm':fahrenheit_to_centigrade( df_bho_monthly['Tgm'] )
        })   
            
#------------------------------------------------------------------------------
# CALCULATE: Tobs monthly adjustments using 60 years of data 1961-2020 Tobs = 00:00 - 00:00
#------------------------------------------------------------------------------

# SLICE: by time to calculate monthly adjustments and difference matrices

df_bho_tgm = pd.DataFrame({'Tgm':df_bho_monthly['Tgm'].values}, index=pd.to_datetime(df_bho_monthly.index))
df_bho_2828_1885_1959 = df_bho_2828[ (df_bho_2828.index>=pd.to_datetime('1885-01-01')) & (df_bho_2828.index<=pd.to_datetime('1959-05-01')) ]  
df_bho_2828_1961_2020 = df_bho_2828[ (df_bho_2828.index>=pd.to_datetime('1961-01-01')) & (df_bho_2828.index<=pd.to_datetime('2020-12-01')) ]
df_bho_2828_1885_2020 = df_bho_2828[ (df_bho_2828.index>=pd.to_datetime('1885-01-01')) & (df_bho_2828.index<=pd.to_datetime('2020-12-01')) ]
df_bho_tg_1885_2020 = df_bho_tg[ (df_bho_tg.index>=pd.to_datetime('1885-01-01')) & (df_bho_tg.index<=pd.to_datetime('2020-12-01')) ]
df_bho_tg_1961_2020 = df_bho_tg[ (df_bho_tg.index>=pd.to_datetime('1961-01-01')) & (df_bho_tg.index<=pd.to_datetime('2020-12-01')) ]
df_bho_tgm_1961_2020 = df_bho_tgm[ (df_bho_tgm.index>=pd.to_datetime('1961-01-01')) & (df_bho_tgm.index<=pd.to_datetime('2020-12-01')) ]
df_blue_hill_1961_2020 = df_blue_hill[ (df_blue_hill.index>=pd.to_datetime('1961-01-01')) & (df_blue_hill.index<=pd.to_datetime('2020-12-01')) ]
df_blue_hill_1811_1959 = df_blue_hill[ (df_blue_hill.index>=pd.to_datetime('1811-01-01')) & (df_blue_hill.index<=pd.to_datetime('1959-05-01')) ]
df_blue_hill_1885_1959 = df_blue_hill[ (df_blue_hill.index>=pd.to_datetime('1885-01-01')) & (df_blue_hill.index<=pd.to_datetime('1959-05-01')) ]
df_blue_hill_1959_2020 = df_blue_hill[ (df_blue_hill.index>=pd.to_datetime('1959-06-01')) & (df_blue_hill.index<=pd.to_datetime('2020-12-01')) ]
df_blue_hill_1811_2020 = df_blue_hill[ (df_blue_hill.index>=pd.to_datetime('1811-01-01')) & (df_blue_hill.index<=pd.to_datetime('2020-12-01')) ]
df_blue_hill_1811_1884 = df_blue_hill[ (df_blue_hill.index>=pd.to_datetime('1811-01-01')) & (df_blue_hill.index<=pd.to_datetime('1884-12-01')) ]
df_blue_hill_1885_2020 = df_blue_hill[ (df_blue_hill.index>=pd.to_datetime('1885-01-01')) & (df_blue_hill.index<=pd.to_datetime('2020-12-01')) ]

df_new_haven_1811_1959 = df_new_haven[ (df_new_haven.index>=pd.to_datetime('1811-01-01')) & (df_new_haven.index<=pd.to_datetime('1959-05-01')) ]
df_boston_city_wso_1811_1959 = df_boston_city_wso[ (df_boston_city_wso.index>=pd.to_datetime('1811-01-01')) & (df_boston_city_wso.index<=pd.to_datetime('1959-05-01')) ]
df_providence_wso_1811_1959 = df_providence_wso[ (df_providence_wso.index>=pd.to_datetime('1811-01-01')) & (df_providence_wso.index<=pd.to_datetime('1959-05-01')) ]
df_20CRv3_bho_1811_1959 = df_20CRv3_bho[ (df_20CRv3_bho.index>=pd.to_datetime('1811-01-01')) & (df_20CRv3_bho.index<=pd.to_datetime('1959-05-01')) ]
df_20CRv3_new_haven_1811_1959 = df_20CRv3_new_haven[ (df_20CRv3_new_haven.index>=pd.to_datetime('1811-01-01')) & (df_20CRv3_new_haven.index<=pd.to_datetime('1959-05-01')) ]

df_USC00199928_1959_2020 = df_USC00199928[ (df_USC00199928.index>=pd.to_datetime('1959-06-01')) & (df_USC00199928.index<=pd.to_datetime('2020-12-01')) ] # WORCESTER, MA
df_USW00094701_1959_2020 = df_USW00094701[ (df_USW00094701.index>=pd.to_datetime('1959-06-01')) & (df_USW00094701.index<=pd.to_datetime('2020-12-01')) ] # BOSTON CITY WSO, MA
df_USC00190736_1959_2020 = df_USC00190736[ (df_USC00190736.index>=pd.to_datetime('1959-06-01')) & (df_USC00190736.index<=pd.to_datetime('2020-12-01')) ] # BLUE HILL, MA
df_USC00194105_1959_2020 = df_USC00194105[ (df_USC00194105.index>=pd.to_datetime('1959-06-01')) & (df_USC00194105.index<=pd.to_datetime('2020-12-01')) ] # LAWRENCE, MA
df_USC00190120_1959_2020 = df_USC00190120[ (df_USC00190120.index>=pd.to_datetime('1959-06-01')) & (df_USC00190120.index<=pd.to_datetime('2020-12-01')) ] # AMHERST, MA
df_USC00376712_1959_2020 = df_USC00376712[ (df_USC00376712.index>=pd.to_datetime('1959-06-01')) & (df_USC00376712.index<=pd.to_datetime('2020-12-01')) ] # PROVIDENCE 2, RI
df_USC00190538_1959_2020 = df_USC00190538[ (df_USC00190538.index>=pd.to_datetime('1959-06-01')) & (df_USC00190538.index<=pd.to_datetime('2020-12-01')) ] # BEDFORD, MA
df_USC00195306_1959_2020 = df_USC00195306[ (df_USC00195306.index>=pd.to_datetime('1959-06-01')) & (df_USC00195306.index<=pd.to_datetime('2020-12-01')) ] # NEW SALEM, MA
df_USW00014739_1959_2020 = df_USW00014739[ (df_USW00014739.index>=pd.to_datetime('1959-06-01')) & (df_USW00014739.index<=pd.to_datetime('2020-12-01')) ] # BOSTON, MA
df_USW00014765_1959_2020 = df_USW00014765[ (df_USW00014765.index>=pd.to_datetime('1959-06-01')) & (df_USW00014765.index<=pd.to_datetime('2020-12-01')) ] # PROVIDENCE, RI
df_USW00014758_1959_2020 = df_USW00014758[ (df_USW00014758.index>=pd.to_datetime('1959-06-01')) & (df_USW00014758.index<=pd.to_datetime('2020-12-01')) ] # NEW HAVEN TWEED AIRPORT, CT
df_USC00065910_1959_2020 = df_USC00065910[ (df_USC00065910.index>=pd.to_datetime('1959-06-01')) & (df_USC00065910.index<=pd.to_datetime('2020-12-01')) ] # NORWICH, CT
    
# CALCULATE: monthly adjustments (1 per month) and differences [T2828 (monthly) - Tg (monthly)]

df_bho_differences_T2828_Tg = df_bho_2828_1961_2020['T2828'] - df_bho_tg_1961_2020['Tg']
years = df_bho_differences_T2828_Tg.index.year.unique()
differences_T2828_Tg = pd.DataFrame(index=years)
tobs_adjustment_T2828_Tg = []

for i in range(12):
    
    mask = df_bho_differences_T2828_Tg.index.month == i+1
    month = df_bho_differences_T2828_Tg.values[mask]
    differences_T2828_Tg[str(i+1)] = month
    month_adjustment = np.nanmean( df_bho_2828_1961_2020['T2828'][mask] - df_bho_tg_1961_2020['Tg'][mask] )        
    tobs_adjustment_T2828_Tg.append(month_adjustment)

# CALCULATE: differences [T2828 (monthly) - Tgm (i.e. from daily)]

df_bho_differences_T2828_Tgm = df_bho_2828_1961_2020['T2828'] - df_bho_tgm_1961_2020['Tgm']
years = df_bho_differences_T2828_Tgm.index.year.unique()
differences_T2828_Tgm = pd.DataFrame(index=years)
tobs_adjustment_T2828_Tgm = []

for i in range(12):
    
    mask = df_bho_differences_T2828_Tgm.index.month == i+1
    month = df_bho_differences_T2828_Tgm.values[mask]
    differences_T2828_Tgm[str(i+1)] = month
    month_adjustment = np.nanmean( df_bho_2828_1961_2020['T2828'][mask] - df_bho_tgm_1961_2020['Tgm'][mask] )        
    tobs_adjustment_T2828_Tgm.append(month_adjustment)

# CALCULATE: differences [Tg (monthly) - Tgm (i.e. from daily)]

df_bho_differences_Tg_Tgm = df_bho_tg_1961_2020['Tg'] - df_bho_tgm_1961_2020['Tgm']
years = df_bho_tg_1961_2020.index.year.unique()
differences_Tg_Tgm = pd.DataFrame(index=years)
tobs_adjustment_Tg_Tgm = []

for i in range(12):
    
    mask = df_bho_differences_Tg_Tgm.index.month == i+1
    month = df_bho_differences_Tg_Tgm.values[mask]
    differences_Tg_Tgm[str(i+1)] = month
    month_adjustment = np.nanmean( df_bho_tg_1961_2020['Tg'][mask] - df_bho_tgm_1961_2020['Tgm'][mask] )        
    tobs_adjustment_Tg_Tgm.append(month_adjustment)

if save_monthly_adjustments == True:

    df_tobs_adjustment_T2828_Tg = pd.DataFrame({'tobs_adjustment_T2828_Tg':tobs_adjustment_T2828_Tg}, index=list(np.arange(1,13)))
    df_tobs_adjustment_T2828_Tg.index.name = 'month'
    df_tobs_adjustment_T2828_Tg.to_csv('tobs_adjustment_T2828_Tg.csv')
    
    df_tobs_adjustment_T2828_Tgm = pd.DataFrame({'tobs_adjustment_T2828_Tgm':tobs_adjustment_T2828_Tgm}, index=list(np.arange(1,13)))
    df_tobs_adjustment_T2828_Tgm.index.name = 'month'
    df_tobs_adjustment_T2828_Tgm.to_csv('tobs_adjustment_T2828_Tgm.csv')
    
    df_tobs_adjustment_Tg_Tgm = pd.DataFrame({'tobs_adjustment_Tg_Tgm':tobs_adjustment_Tg_Tgm}, index=list(np.arange(1,13)))
    df_tobs_adjustment_Tg_Tgm.index.name = 'month'
    df_tobs_adjustment_Tg_Tgm.to_csv('tobs_adjustment_Tg_Tgm.csv')

# SELECT: monthly adjustment protocol

tobs_adjustment = tobs_adjustment_T2828_Tg

# TILE: monthly adjustments 1811-2020 with June 1959-2020 being 0.0

tobs_adjustment_1811_2020 = np.tile(tobs_adjustment, reps=(2020-1811+1))
tobs_adjustments = pd.Series(tobs_adjustment_1811_2020, index=pd.date_range(start='1811', periods=(2020-1811+1)*12, freq='MS'))
tobs_adjustments[tobs_adjustments.index>=pd.to_datetime('1959-06-01')] = 0.0                                     

# APPLY: monthly adjustments 

df_blue_hill_1811_1884.index.name = 'datetime'
df_bho_2828_1885_1959.rename(columns = {'T2828':'blue_hill'}, inplace=True)
df_blue_hill_1811_1959 = pd.DataFrame(df_blue_hill_1811_1884['blue_hill'].append(df_bho_2828_1885_1959['blue_hill']))
df_blue_hill_1811_2020 = pd.DataFrame(df_blue_hill_1811_1959['blue_hill'].append(df_blue_hill_1959_2020['blue_hill']))
df_blue_hill_1811_2020_tobs_adjusted = df_blue_hill_1811_2020.copy()
df_blue_hill_1811_2020_tobs_adjusted['blue_hill'] = df_blue_hill_1811_2020_tobs_adjusted['blue_hill'] + tobs_adjustments.values
df_blue_hill_1885_2020_tobs_adjusted = df_blue_hill_1811_2020_tobs_adjusted[ (df_blue_hill_1811_2020_tobs_adjusted.index>=pd.to_datetime('1885-01-01')) & (df_blue_hill.index<=pd.to_datetime('2020-12-01')) ]
df_blue_hill_1885_2020_tobs_adjusted.index.name = 'datetime'
df_blue_hill_1811_1959_tobs_adjusted = df_blue_hill_1811_1959.copy()
df_ghcnmv4_qcu_1811_1959 = df_ghcnmv4_qcu[ (df_ghcnmv4_qcu.index>=pd.to_datetime('1811-01-01')) & (df_ghcnmv4_qcu.index<=pd.to_datetime('1959-05-01'))] 
df_ghcnmv4_qcf_1811_1959 = df_ghcnmv4_qcf[ (df_ghcnmv4_qcf.index>=pd.to_datetime('1811-01-01')) & (df_ghcnmv4_qcf.index<=pd.to_datetime('1959-05-01'))] 

# CALCULATE: delta normals

normal_blue_hill = np.nanmean( df_blue_hill_1811_2020_tobs_adjusted[     
(df_blue_hill_1811_2020_tobs_adjusted.index>=pd.to_datetime('1961-01-01')) & 
(df_blue_hill_1811_2020_tobs_adjusted.index<=pd.to_datetime('1990-12-01')) ]['blue_hill'])

normal_cet = np.nanmean( df_cet[     
(df_cet.index>=1961.0) & 
(df_cet.index<1991.0) ]['df_cet'])
    
normal_boston_city_wso = np.nanmean( df_boston_city_wso[     
(df_boston_city_wso.index>=pd.to_datetime('1961-01-01')) & 
(df_boston_city_wso.index<=pd.to_datetime('1990-12-01')) ]['boston_city_wso'])
    
normal_new_haven = np.nanmean( df_new_haven[     
(df_new_haven.index>=pd.to_datetime('1961-01-01')) & 
(df_new_haven.index<=pd.to_datetime('1990-12-01')) ]['new_haven'])
    
normal_providence_wso = np.nanmean( df_providence_wso[     
(df_providence_wso.index>=pd.to_datetime('1961-01-01')) & 
(df_providence_wso.index<=pd.to_datetime('1990-12-01')) ]['providence_wso'])

normal_st_lawrence_valley = np.nanmean( df_st_lawrence_valley[     
(df_st_lawrence_valley.index>=pd.to_datetime('1961-01-01')) & 
(df_st_lawrence_valley.index<=pd.to_datetime('1990-12-01')) ]['df_st_lawrence_valley'])

normal_amherst = np.nanmean( df_amherst[     
(df_amherst.index>=pd.to_datetime('1961-01-01')) & 
(df_amherst.index<=pd.to_datetime('1990-12-01')) ]['amherst'])

normal_amherst2 = np.nanmean( df_amherst2[     
(df_amherst2.index>=pd.to_datetime('1961-01-01')) & 
(df_amherst2.index<=pd.to_datetime('1990-12-01')) ]['amherst2'])

# CALCULATE: Holyoke daily T(919) and Wigglesworth daily T(919) and reample to monthly timescale
     
df_holyoke_919 = (df_holyoke['T(08:00)']+df_holyoke['T(13:00)']+df_holyoke['T(22:00)']+df_holyoke['T(sunset)'])/4.0
df_holyoke_919 = pd.DataFrame({'T(919)':df_holyoke_919.values}, index=pd.to_datetime(df_holyoke_919.index))
df_holyoke_919.index.name = 'datetime'
df_holyoke_919_xr = df_holyoke_919.to_xarray()
df_holyoke_919_xr_resampled = df_holyoke_919_xr['T(919)'].resample(datetime='MS').mean().to_dataset()
df_holyoke_919 = pd.DataFrame({'T(919)':df_holyoke_919_xr_resampled['T(919)'].values}, index=df_holyoke_919_xr_resampled.datetime.values)
df_holyoke_919.index.name = 'datetime'
    
df_wigglesworth_919 = (df_wigglesworth['T(08:00)']+df_wigglesworth['T(13:00)']+df_wigglesworth['T(21:00)'])/3.0
df_wigglesworth_919 = pd.DataFrame({'T(919)':df_wigglesworth_919.values}, index=pd.to_datetime(df_wigglesworth_919.index))
df_wigglesworth_919.index.name = 'datetime'
df_wigglesworth_919_xr = df_wigglesworth_919.to_xarray()
df_wigglesworth_919_xr_resampled = df_wigglesworth_919_xr['T(919)'].resample(datetime='MS').mean().to_dataset()
df_wigglesworth_919 = pd.DataFrame({'T(919)':df_wigglesworth_919_xr_resampled['T(919)'].values}, index=df_wigglesworth_919_xr_resampled.datetime.values)
df_wigglesworth_919.index.name = 'datetime'
    
df_farrar_919 = df_farrar['Tmean']
df_farrar_919 = pd.DataFrame({'T(919)':df_farrar_919.values}, index=pd.to_datetime(df_farrar_919.index))
df_farrar_919.index.name = 'datetime'
    
# CALCULATE: back-extension using Farrar downshifted using Boston normal
    
df_farrar_aligned = (df_farrar_919.copy() - (normal_boston_city_wso - normal_blue_hill))
df_farrar_1790_1810 = df_farrar_aligned[df_farrar_aligned.index<pd.to_datetime('1811-01-01')]
df_farrar_1790_1810_shifted = df_farrar_1790_1810.rename(columns = {'T(919)':'blue_hill'})
df_blue_hill_1790_2020_tobs_adjusted = df_farrar_1790_1810_shifted.append(df_blue_hill_1811_2020_tobs_adjusted)

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SEABORN:
#
# sns.jointplot(x=x, y=y, kind='kde', color='blue', marker='+', fill=True)  # kind{ “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }            

# jointgrid = sns.JointGrid(x=x, y=y, data=df)
# jointgrid.plot_joint(sns.scatterplot)
# jointgrid.plot_marginals(sns.kdeplot)

# pairgrid = sns.PairGrid(data=iris)
# pairgrid = pairgrid.map_upper(sns.scatterplot)
# pairgrid = pairgrid.map_diag(plt.hist)
# pairgrid = pairgrid.map_lower(sns.kdeplot)

# sns.scatterplot(x=dx, y=y, s=5, color="blue")
# sns.histplot(x=x, y=y, bins=100, pthresh=0.01, cmap="Blues")
# sns.kdeplot(x=x, y=y, levels=10, color="k", linewidths=1)
# sns.kdeplot(x, color='blue', shade=True, alpha=0.2, legend=True, **kwargs, label='')
# sns.boxplot(data = x, orient = "v")
# sns.violinplot(data = x, orient = "v")
#------------------------------------------------------------------------------

#==============================================================================

if plot_historical == True:
    
    # PLOT: Holyoke + Wigglesworth daily observations + T919 (monthly) + Farrar monthly
    
    print('plotting Holyoke (daily) + Farrar monthly osbervations ...')
        
    figstr = 'salem(MA)-holyoke-cambridge(MA)-farrar.png'
    titlestr = 'Salem, MA: Holyoke (sub-daily) and Cambridge, MA: Farrar (monthly mean) observations'
            
    fig, axs = plt.subplots(figsize=(15,10))
    
    sns.lineplot(x=df_holyoke.index, y=(df_holyoke['T(08:00)']), ax=axs, marker='.', color='blue', alpha=1.0, label='Salem, MA: T(08:00)')
    sns.lineplot(x=df_holyoke.index, y=(df_holyoke['T(13:00)']), ax=axs, marker='.', color='red', alpha=1.0, label='Salem, MA: T(13:00)')
    sns.lineplot(x=df_holyoke.index, y=(df_holyoke['T(22:00)']), ax=axs, marker='.', color='purple', alpha=1.0, label='Salem, MA: T(22:00)')
    sns.lineplot(x=df_holyoke.index, y=(df_holyoke['T(sunset)']), ax=axs, marker='.', color='orange', alpha=1.0, label='Salem, MA: T(sunset)')
    sns.lineplot(x=df_farrar.index, y=(df_farrar['Tmean']), ax=axs, marker='.', color='navy', ls='-', lw=3, label='Cambridge, MA: T(mean)')
    if use_fahrenheit == True:
        axs.set_ylim(-20,110)
    else:
        axs.set_ylim(-30,40)
    axs.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs.set_xlim(pd.Timestamp('1785-01-01'),pd.Timestamp('1835-01-01'))
    axs.set_xlabel('Year', fontsize=fontsize)
    axs.set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs.set_title(titlestr, fontsize=fontsize)
    axs.tick_params(labelsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: Wigglesworth observations (daily)
    
    print('plotting Wigglesworth (+ Farrar) osbervations ...')
        
    figstr = 'salem(MA)-wigglesworth-cambridge(MA)-farrar.png'
    titlestr = 'Salem, MA: Wigglesworth (sub-daily) and Cambridge, MA: Farrar (monthly mean) observations'
    
    fig, axs = plt.subplots(figsize=(15,10))
    
    sns.lineplot(x=df_wigglesworth.index, y=(df_wigglesworth['T(08:00)']), ax=axs, marker='.', color='blue', alpha=1.0, label='Salem, MA: T(08:00)')
    sns.lineplot(x=df_wigglesworth.index, y=(df_wigglesworth['T(13:00)']), ax=axs, marker='.', color='red', alpha=1.0, label='Salem, MA: T(13:00)')
    sns.lineplot(x=df_wigglesworth.index, y=(df_wigglesworth['T(21:00)']), ax=axs, marker='.', color='purple', alpha=1.0, label='Salem, MA: T(21:00)')
    sns.lineplot(x=df_farrar.index, y=(df_farrar['Tmean']), ax=axs, marker='.', color='navy', ls='-', lw=3, label='Cambridge, MA: T(mean)')
    if use_fahrenheit == True:
        axs.set_ylim(-20,110)
    else:
        axs.set_ylim(-30,40)
    axs.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs.set_xlim(pd.Timestamp('1785-01-01'),pd.Timestamp('1835-01-01'))
    axs.set_xlabel('Year', fontsize=fontsize)
    axs.set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs.set_title(titlestr, fontsize=fontsize)
    axs.tick_params(labelsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

    # PLOT: Holyoke + Wigglesworth daily observations + T919 (monthly) + Farrar monthly

    print('plotting Holyoke + Wigglesworh + T919 + Farrar osbervations ...')
        
    figstr = 'salem(MA)-holyoke-wigglesworth-T919-cambridge(MA)-farrar.png'           
    titlestr = 'Salem, MA: Holyoke (sub-daily) and Cambridge, MA: Farrar (monthly mean) observations'
            
    fig, axs = plt.subplots(figsize=(15,10))    
    sns.lineplot(x=df_holyoke.index, y=(df_holyoke['T(08:00)']), ax=axs, marker='.', color='blue', alpha=1.0, label='Salem, MA (Holyoke): T(08:00)')
    sns.lineplot(x=df_holyoke.index, y=(df_holyoke['T(13:00)']), ax=axs, marker='.', color='red', alpha=1.0, label='Salem, MA (Holyoke): T(13:00)')
    sns.lineplot(x=df_holyoke.index, y=(df_holyoke['T(22:00)']), ax=axs, marker='.', color='purple', alpha=1.0, label='Salem, MA (Holyoke): T(22:00)')
    sns.lineplot(x=df_holyoke.index, y=(df_holyoke['T(sunset)']), ax=axs, marker='.', color='orange', alpha=1.0, label='Salem, MA (Holyoke): T(sunset)')
    sns.lineplot(x=df_holyoke_919.index, y=(df_holyoke_919['T(919)']), ax=axs, color='purple', ls='-', lw=3, alpha=1.0, label='Salem, MA (Holyoke): T(919) monthly')
    sns.lineplot(x=df_wigglesworth_919.index, y=(df_wigglesworth_919['T(919)']), ax=axs, color='teal', ls='-', lw=3, alpha=1.0, label='Salem, MA (Wigglesworth): T(919) monthly')
    sns.lineplot(x=df_farrar.index, y=(df_farrar['Tmean']), ax=axs, marker='.', color='navy', ls='-', lw=3, label='Cambridge, MA (Farrar): T(mean) monthly')    
    if use_fahrenheit == True:
        axs.set_ylim(-20,110)
    else:
        axs.set_ylim(-30,40)
    axs.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs.set_xlim(pd.Timestamp('1785-01-01'),pd.Timestamp('1835-01-01'))
    axs.set_xlabel('Year', fontsize=fontsize)
    axs.set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs.set_title(titlestr, fontsize=fontsize)
    axs.tick_params(labelsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
            
#==============================================================================

if plot_differences == True:
            
    # PLOT: BHO: Tg (from daily) versus Tg (monthly)
    
    print('plotting BHO: monthly Tg (from daily) vs Tg ...')
        
    figstr = 'bho-tg(from-daily)-vs-tg.png'
    titlestr = 'Blue Hill Observatory: monthly BHO $T_{g}$ (from daily) vs BHO $T_g$'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_bho_tg.index, y='Tgm', data=df_bho_monthly, ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ (from daily) BHO')
    sns.lineplot(x=df_bho_tg.index, y='Tg', data=df_bho_monthly, ax=axs[0], marker='.', color='blue', alpha=1.0, label='$T_{g}$ BHO')
    sns.lineplot(x=df_bho_tg.index, y=df_bho_monthly['Tgm']-df_bho_monthly['Tg'], data=df_bho_monthly, ax=axs[1], color='teal')
    
    mask_pre_1891 = df_bho_tg.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_bho_tg.index >= pd.Timestamp('1891-01-01')) & (df_bho_tg.index < pd.Timestamp('1959-06-01'))
    mask_post_1959 = df_bho_tg.index >= pd.Timestamp('1959-06-01')
    sns.lineplot(x=df_bho_tg.index[mask_pre_1891], y=mask_pre_1891.sum()*[ np.nanmean((df_bho_monthly['Tgm']-df_bho_monthly['Tg']).values[mask_pre_1891]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_tg.index[mask_1891_1959], y=mask_1891_1959.sum()*[ np.nanmean((df_bho_monthly['Tgm']-df_bho_monthly['Tg']).values[mask_1891_1959]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_tg.index[mask_post_1959], y=mask_post_1959.sum()*[ np.nanmean((df_bho_monthly['Tgm']-df_bho_monthly['Tg']).values[mask_post_1959]) ], ls='--', lw=1, color='black')            
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1891-01-01'), ls='--', lw=1, color='black')
    plt.axvline(x=pd.Timestamp('1959-06-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1891-01-01'), y=-0.1, s='1891-01')
    plt.text(x=pd.Timestamp('1959-06-01'), y=-0.1, s='1959-06')
    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)   
    axs[1].sharex(axs[0]) 
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'$T_{g}$ (from daily) - $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-0.1,0.1)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-0.1,0.1)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

    # PLOT: BHO: T2828 (monthly) versus Tg (monthly)
    
    print('plotting BHO: monthly T2828 vs Tg ...')
        
    figstr = 'bho-t2828-vs-tg.png'
    titlestr = 'Blue Hill Observatory: monthly BHO $T_{2828}$ vs BHO $T_g$'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_bho_2828.index, y=df_bho_2828['T2828'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{2828}$ BHO')
    sns.lineplot(x=df_bho_2828.index, y=df_bho_tg['Tg'], ax=axs[0], marker='.', color='blue', alpha=1.0, label='$T_{g}$ BHO')
    sns.lineplot(x=df_bho_2828.index, y=df_bho_2828['T2828'] - df_bho_tg['Tg'], ax=axs[1], color='teal')

    mask_pre_1891 = df_bho_2828.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_bho_2828.index >= pd.Timestamp('1891-01-01')) & (df_bho_2828.index <= pd.Timestamp('1959-05-01'))
    mask_post_1959 = df_bho_2828.index >= pd.Timestamp('1959-06-01')
    sns.lineplot(x=df_bho_2828.index[mask_pre_1891], y=mask_pre_1891.sum()*[ np.nanmean((df_bho_2828['T2828']-df_bho_tg['Tg']).values[mask_pre_1891]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_2828.index[mask_1891_1959], y=mask_1891_1959.sum()*[ np.nanmean((df_bho_2828['T2828']-df_bho_tg['Tg']).values[mask_1891_1959]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_2828.index[mask_post_1959], y=mask_post_1959.sum()*[ np.nanmean((df_bho_2828['T2828']-df_bho_tg['Tg']).values[mask_post_1959]) ], ls='--', lw=1, color='black')            
    plt.axvline(x=pd.Timestamp('1891-01-01'), ls='--', lw=1, color='black')
    plt.axvline(x=pd.Timestamp('1959-06-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1891-01-01'), y=-3, s='1891-01')
    plt.text(x=pd.Timestamp('1959-06-01'), y=-3, s='1959-06')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'$T_{2828}$ - $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-3,3)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: BHO: T2828 (monthly) vs GloSAT Tg (monthly)
    
    print('plotting BHO: monthly T2828 vs GloSAT Tg ...')
    
    figstr = 'bho-t2828-vs-glosat-tg.png'
    titlestr = 'Blue Hill Observatory: monthly BHO $T_{2828}$ vs GloSAT $T_g$'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_bho_2828_1885_2020.index, y=df_bho_2828_1885_2020['T2828'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{2828}$ BHO')
    sns.lineplot(x=df_bho_2828_1885_2020.index, y=df_blue_hill_1885_2020['blue_hill'], ax=axs[0], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GloSAT')
    sns.lineplot(x=df_bho_2828_1885_2020.index, y=df_bho_2828_1885_2020['T2828'] - df_blue_hill_1885_2020['blue_hill'], ax=axs[1], color='teal')

    mask_pre_1891 = df_bho_2828_1885_2020.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_bho_2828_1885_2020.index >= pd.Timestamp('1891-01-01')) & (df_bho_2828_1885_2020.index <= pd.Timestamp('1959-05-01'))
    mask_post_1959 = df_bho_2828_1885_2020.index >= pd.Timestamp('1959-06-01')
    sns.lineplot(x=df_bho_2828_1885_2020.index[mask_pre_1891], y=mask_pre_1891.sum()*[ np.nanmean((df_bho_2828_1885_2020['T2828']-df_blue_hill_1885_2020['blue_hill']).values[mask_pre_1891]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_2828_1885_2020.index[mask_1891_1959], y=mask_1891_1959.sum()*[ np.nanmean((df_bho_2828_1885_2020['T2828']-df_blue_hill_1885_2020['blue_hill']).values[mask_1891_1959]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_2828_1885_2020.index[mask_post_1959], y=mask_post_1959.sum()*[ np.nanmean((df_bho_2828_1885_2020['T2828']-df_blue_hill_1885_2020['blue_hill']).values[mask_post_1959]) ], ls='--', lw=1, color='black')            
    plt.axvline(x=pd.Timestamp('1891-01-01'), ls='--', lw=1, color='black')
    plt.axvline(x=pd.Timestamp('1959-06-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1891-01-01'), y=-3, s='1891-01')
    plt.text(x=pd.Timestamp('1959-06-01'), y=-3, s='1959-06')
    
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'$T_{2828}$ - $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-3,3)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: BHO: T2828 (monthly) vs GloSAT Tg (monthly) adjusted
    
    print('plotting BHO: monthly T2828 vs GloSAT Tg (adjusted) ...')
    
    figstr = 'bho-t2828-vs-glosat-tg-adjusted.png'
    titlestr = 'Blue Hill Observatory: monthly BHO $T_{2828}$ vs GloSAT $T_g$ (adjusted)'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1885_2020_tobs_adjusted.index, y=df_bho_2828_1885_2020['T2828'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{2828}$ BHO')
    sns.lineplot(x=df_blue_hill_1885_2020_tobs_adjusted.index, y=df_blue_hill_1885_2020_tobs_adjusted['blue_hill'], ax=axs[0], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GloSAT (adjusted)')
    sns.lineplot(x=df_blue_hill_1885_2020_tobs_adjusted.index, y=df_bho_2828_1885_2020['T2828'] - df_blue_hill_1885_2020_tobs_adjusted['blue_hill'], ax=axs[1], color='teal')

    mask_pre_1891 = df_blue_hill_1885_2020_tobs_adjusted.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_blue_hill_1885_2020_tobs_adjusted.index >= pd.Timestamp('1891-01-01')) & (df_blue_hill_1885_2020_tobs_adjusted.index <= pd.Timestamp('1959-05-01'))
    mask_post_1959 = df_blue_hill_1885_2020_tobs_adjusted.index >= pd.Timestamp('1959-06-01')
    sns.lineplot(x=df_blue_hill_1885_2020_tobs_adjusted.index[mask_pre_1891], y=mask_pre_1891.sum()*[ np.nanmean((df_bho_2828_1885_2020['T2828']-df_blue_hill_1885_2020_tobs_adjusted['blue_hill']).values[mask_pre_1891]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_blue_hill_1885_2020_tobs_adjusted.index[mask_1891_1959], y=mask_1891_1959.sum()*[ np.nanmean((df_bho_2828_1885_2020['T2828']-df_blue_hill_1885_2020_tobs_adjusted['blue_hill']).values[mask_1891_1959]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_blue_hill_1885_2020_tobs_adjusted.index[mask_post_1959], y=mask_post_1959.sum()*[ np.nanmean((df_bho_2828_1885_2020['T2828']-df_blue_hill_1885_2020_tobs_adjusted['blue_hill']).values[mask_post_1959]) ], ls='--', lw=1, color='black')            
    plt.axvline(x=pd.Timestamp('1891-01-01'), ls='--', lw=1, color='black')
    plt.axvline(x=pd.Timestamp('1959-06-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1891-01-01'), y=-3, s='1891-01')
    plt.text(x=pd.Timestamp('1959-06-01'), y=-3, s='1959-06')
            
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'$T_{2828}$ - $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-3,3)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: BHO: Tg (monthly) vs GloSAT Tg (monthly)
    
    print('plotting BHO: monthly Tg vs GloSAT Tg ...')
    
    figstr = 'bho-tg-vs-glosat-tg.png'
    titlestr = 'Blue Hill Observatory: monthly BHO $T_{g}$ vs GloSAT $T_g$'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_bho_tg_1885_2020.index, y=df_bho_tg_1885_2020['Tg'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ BHO')
    sns.lineplot(x=df_bho_tg_1885_2020.index, y=df_blue_hill_1885_2020['blue_hill'], ax=axs[0], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GloSAT')
    sns.lineplot(x=df_bho_tg_1885_2020.index, y=df_bho_tg_1885_2020['Tg'] - df_blue_hill_1885_2020['blue_hill'], ax=axs[1], color='teal')

    mask_pre_1891 = df_bho_tg_1885_2020.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_bho_tg_1885_2020.index >= pd.Timestamp('1891-01-01')) & (df_bho_tg_1885_2020.index <= pd.Timestamp('1959-05-01'))
    mask_post_1959 = df_bho_tg_1885_2020.index >= pd.Timestamp('1959-06-01')
    sns.lineplot(x=df_bho_tg_1885_2020.index[mask_pre_1891], y=mask_pre_1891.sum()*[ np.nanmean((df_bho_tg_1885_2020['Tg']-df_blue_hill_1885_2020['blue_hill']).values[mask_pre_1891]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_tg_1885_2020.index[mask_1891_1959], y=mask_1891_1959.sum()*[ np.nanmean((df_bho_tg_1885_2020['Tg']-df_blue_hill_1885_2020['blue_hill']).values[mask_1891_1959]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_tg_1885_2020.index[mask_post_1959], y=mask_post_1959.sum()*[ np.nanmean((df_bho_tg_1885_2020['Tg']-df_blue_hill_1885_2020['blue_hill']).values[mask_post_1959]) ], ls='--', lw=1, color='black')            
    plt.axvline(x=pd.Timestamp('1891-01-01'), ls='--', lw=1, color='black')
    plt.axvline(x=pd.Timestamp('1959-06-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1891-01-01'), y=-3, s='1891-01')
    plt.text(x=pd.Timestamp('1959-06-01'), y=-3, s='1959-06')
    
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - GloSAT $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-3,3)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: BHO: Tg (monthly) vs GloSAT Tg (monthly) adjusted
    
    print('plotting BHO: monthly Tg vs GloSAT Tg (adjusted) ...')
    
    figstr = 'bho-tg-vs-glosat-tg-adjusted.png'
    titlestr = 'Blue Hill Observatory: monthly BHO $T_{g}$ vs GloSAT $T_g$ (adjusted)'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_bho_tg_1885_2020.index, y=df_bho_tg_1885_2020['Tg'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ BHO')
    sns.lineplot(x=df_bho_tg_1885_2020.index, y=df_blue_hill_1885_2020_tobs_adjusted['blue_hill'].values, ax=axs[0], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GloSAT (adjusted)')
    sns.lineplot(x=df_bho_tg_1885_2020.index, y=df_bho_tg_1885_2020['Tg'] - df_blue_hill_1885_2020_tobs_adjusted['blue_hill'].values, ax=axs[1], color='teal')

    mask_pre_1891 = df_bho_tg_1885_2020.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_bho_tg_1885_2020.index >= pd.Timestamp('1891-01-01')) & (df_bho_tg_1885_2020.index <= pd.Timestamp('1959-05-01'))
    mask_post_1959 = df_bho_tg_1885_2020.index >= pd.Timestamp('1959-06-01')
    sns.lineplot(x=df_bho_tg_1885_2020.index[mask_pre_1891], y=mask_pre_1891.sum()*[ np.nanmean((df_bho_tg_1885_2020['Tg']-df_blue_hill_1885_2020_tobs_adjusted['blue_hill']).values[mask_pre_1891]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_tg_1885_2020.index[mask_1891_1959], y=mask_1891_1959.sum()*[ np.nanmean((df_bho_tg_1885_2020['Tg']-df_blue_hill_1885_2020_tobs_adjusted['blue_hill']).values[mask_1891_1959]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_bho_tg_1885_2020.index[mask_post_1959], y=mask_post_1959.sum()*[ np.nanmean((df_bho_tg_1885_2020['Tg']-df_blue_hill_1885_2020_tobs_adjusted['blue_hill']).values[mask_post_1959]) ], ls='--', lw=1, color='black')            
    plt.axvline(x=pd.Timestamp('1891-01-01'), ls='--', lw=1, color='black')
    plt.axvline(x=pd.Timestamp('1959-06-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1891-01-01'), y=-3, s='1891-01')
    plt.text(x=pd.Timestamp('1959-06-01'), y=-3, s='1959-06')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$- GloSAT $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-3,3)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs GCHNM-v4 QCU
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GHCNM-v4 QCU ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-ghcnmv4-qcu.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GHCNM-v4 QCU'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_ghcnmv4_qcu_1811_1959.index, df_ghcnmv4_qcu_1811_1959['df_ghcnmv4_qcu'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GHCNM-v4 QCU')
    axs[1].plot(df_blue_hill_1811_1959_tobs_adjusted.index, df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_ghcnmv4_qcu_1811_1959['df_ghcnmv4_qcu'], color='teal')

    mask_pre_1893 = df_blue_hill_1811_1959_tobs_adjusted.index < pd.Timestamp('1893-01-01')
    mask_1893_1959 = (df_blue_hill_1811_1959_tobs_adjusted.index >= pd.Timestamp('1893-01-01')) & (df_blue_hill_1811_1959_tobs_adjusted.index <= pd.Timestamp('1959-05-01'))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index[mask_pre_1893], y=mask_pre_1893.sum()*[ np.nanmean((df_blue_hill_1811_1959_tobs_adjusted['blue_hill']-df_ghcnmv4_qcu_1811_1959['df_ghcnmv4_qcu']).values[mask_pre_1893]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index[mask_1893_1959], y=mask_1893_1959.sum()*[ np.nanmean((df_blue_hill_1811_1959_tobs_adjusted['blue_hill']-df_ghcnmv4_qcu_1811_1959['df_ghcnmv4_qcu']).values[mask_1893_1959]) ], ls='--', lw=1, color='black')            
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.axvline(x=pd.Timestamp('1893-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-3, s='1885-01')
    plt.text(x=pd.Timestamp('1893-01-01'), y=-3, s='1893-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - GHCNM-v4 QCU $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-3,3)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs GCHNM-v4 QCF
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GHCNM-v4 QCF ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-ghcnmv4-qcf.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GHCNM-v4 QCF'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_ghcnmv4_qcf_1811_1959.index, df_ghcnmv4_qcf_1811_1959['df_ghcnmv4_qcf'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GHCNM-v4 QCF')
    axs[1].plot(df_blue_hill_1811_1959_tobs_adjusted.index, df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_ghcnmv4_qcf_1811_1959['df_ghcnmv4_qcf'], color='teal')

    mask_pre_1893 = df_blue_hill_1811_1959_tobs_adjusted.index < pd.Timestamp('1893-01-01')
    mask_1893_1959 = (df_blue_hill_1811_1959_tobs_adjusted.index >= pd.Timestamp('1893-01-01')) & (df_blue_hill_1811_1959_tobs_adjusted.index <= pd.Timestamp('1959-05-01'))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index[mask_pre_1893], y=mask_pre_1893.sum()*[ np.nanmean((df_blue_hill_1811_1959_tobs_adjusted['blue_hill']-df_ghcnmv4_qcf_1811_1959['df_ghcnmv4_qcf']).values[mask_pre_1893]) ], ls='--', lw=1, color='black')            
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index[mask_1893_1959], y=mask_1893_1959.sum()*[ np.nanmean((df_blue_hill_1811_1959_tobs_adjusted['blue_hill']-df_ghcnmv4_qcf_1811_1959['df_ghcnmv4_qcf']).values[mask_1893_1959]) ], ls='--', lw=1, color='black')            
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.axvline(x=pd.Timestamp('1893-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-3, s='1885-01')
    plt.text(x=pd.Timestamp('1893-01-01'), y=-3, s='1893-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - GHCNM-v4 QCF $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-3,3)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs GloSAT New Haven
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GloSAT New Haven ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-glosat-new-haven.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GloSAT New Haven'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    sns.lineplot(x=df_new_haven_1811_1959.index, y=df_new_haven_1811_1959['new_haven'], ax=axs[0], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GloSAT New Haven')
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_new_haven_1811_1959['new_haven'], ax=axs[1], color='teal')    

    mean_difference = np.nanmean( df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_new_haven_1811_1959['new_haven'] )
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=len(df_blue_hill_1811_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - New Haven $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs GloSAT Boston City WSO
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GloSAT Boston City WSO ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-glosat-boston-city-wso.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GloSAT Boston City WSO'
        
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_boston_city_wso_1811_1959.index, df_boston_city_wso_1811_1959['boston_city_wso'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GloSAT Boston City WSO')
    axs[1].plot(df_blue_hill_1811_1959_tobs_adjusted.index, df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_boston_city_wso_1811_1959['boston_city_wso'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_boston_city_wso_1811_1959['boston_city_wso'] )
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=len(df_blue_hill_1811_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - Boston City WSO $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs GloSAT Providence WSO
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GloSAT Providence WSO ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-glosat-providence-wso.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GloSAT Providence WSO'
    
    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_providence_wso_1811_1959.index, df_providence_wso_1811_1959['providence_wso'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ GloSAT Providence WSO')
    axs[1].plot(df_blue_hill_1811_1959_tobs_adjusted.index, df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_providence_wso_1811_1959['providence_wso'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_providence_wso_1811_1959['providence_wso'] )
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=len(df_blue_hill_1811_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - Providence WSO $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs 20CRv3 BHO T(2m)
    
    print('plotting BHO: GloSAT Tg (adjusted) vs 20CRv3 BHO T(2m) ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-20crv3-bho-t2m.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs 20CRv3 BHO T(2m)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_20CRv3_bho_1811_1959.index, df_20CRv3_bho_1811_1959['T(2m)'], marker='.', color='blue', alpha=1.0, label='$T_{2m}$ 20CRv3 BHO')
    axs[1].plot(df_blue_hill_1811_1959_tobs_adjusted.index, df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_20CRv3_bho_1811_1959['T(2m)'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_20CRv3_bho_1811_1959['T(2m)'] )
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=len(df_blue_hill_1811_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - 20CRv3 BHO $T_{2m}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
    
    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs 20CRv3 BHO T(1000hPa)
    
    print('plotting BHO: GloSAT Tg (adjusted) vs 20CRv3 BHO T(1000hPa) ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-20crv3-bho-1000hPa.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs 20CRv3 BHO T(1000hPa)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_20CRv3_bho_1811_1959.index, df_20CRv3_bho_1811_1959['T(1000hPa)'], marker='.', color='blue', alpha=1.0, label='$T_{1000hPa}$ 20CRv3 BHO')
    axs[1].plot(df_blue_hill_1811_1959_tobs_adjusted.index, df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_20CRv3_bho_1811_1959['T(1000hPa)'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_20CRv3_bho_1811_1959['T(1000hPa)'] )
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=len(df_blue_hill_1811_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - 20CRv3 BHO $T_{1000hPa}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')        
    
    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs 20CRv3 New Haven T(2m)
    
    print('plotting BHO: GloSAT Tg (adjusted) vs 20CRv3 New Haven T(2m) ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-20crv3-new-haven-t2m.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs 20CRv3 New Haven T(2m)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_20CRv3_new_haven_1811_1959.index, df_20CRv3_new_haven_1811_1959['T(2m)'], marker='.', color='blue', alpha=1.0, label='$T_{2m}$ 20CRv3 New Haven')
    axs[1].plot(df_blue_hill_1811_1959_tobs_adjusted.index, df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_20CRv3_new_haven_1811_1959['T(2m)'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_20CRv3_new_haven_1811_1959['T(2m)'] )
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=len(df_blue_hill_1811_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - 20CRv3 New Haven $T_{2m}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
    
    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs 20CRv3 New Haven T(1000hPa)
    
    print('plotting BHO: GloSAT Tg (adjusted) vs 20CRv3 New Haven T(1000hPa) ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-20crv3-new-haven-1000hPa.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs 20CRv3 New Haven T(1000hPa)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=df_blue_hill_1811_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_20CRv3_new_haven_1811_1959.index, df_20CRv3_new_haven_1811_1959['T(1000hPa)'], marker='.', color='blue', alpha=1.0, label='$T_{1000hPa}$ 20CRv3 New Haven')
    axs[1].plot(df_blue_hill_1811_1959_tobs_adjusted.index, df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_20CRv3_new_haven_1811_1959['T(1000hPa)'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1811_1959_tobs_adjusted['blue_hill'] - df_20CRv3_new_haven_1811_1959['T(1000hPa)'] )
    sns.lineplot(x=df_blue_hill_1811_1959_tobs_adjusted.index, y=len(df_blue_hill_1811_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - 20CRv3 New Haven $T_{1000hPa}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')            

    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs HadCRUT5-analysis BHO T(2m)
    
    print('plotting BHO: GloSAT Tg (adjusted) vs HadCRUT5-analysis BHO T(2m) ...')

    df_blue_hill_1850_1959_tobs_adjusted = df_blue_hill_1811_1959_tobs_adjusted[ df_blue_hill_1811_1959_tobs_adjusted.index>=pd.to_datetime('1850-01-01') ]
    df_hadcrut5_bho_1850_1959 = df_hadcrut5_bho[(df_hadcrut5_bho.index>=pd.to_datetime('1811-01-01')) & (df_hadcrut5_bho.index<=pd.to_datetime('1959-06-01'))] 
    
    figstr = 'bho-glosat-tg-adjusted-vs-hadcrut5-analysis-bho.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs HadCRUT5-analysis BHO T(2m)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1850_1959_tobs_adjusted.index, y=df_blue_hill_1850_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_hadcrut5_bho_1850_1959.index, df_hadcrut5_bho_1850_1959['df_hadcrut5_bho'], marker='.', color='blue', alpha=1.0, label='$T_{2m}$ HadCRUT5-analysis BHO')
    axs[1].plot(df_blue_hill_1850_1959_tobs_adjusted.index, df_blue_hill_1850_1959_tobs_adjusted['blue_hill'].values - df_hadcrut5_bho_1850_1959['df_hadcrut5_bho'].values, color='teal')

    mean_difference = np.nanmean( df_blue_hill_1850_1959_tobs_adjusted['blue_hill'].values - df_hadcrut5_bho_1850_1959['df_hadcrut5_bho'].values )
    sns.lineplot(x=df_blue_hill_1850_1959_tobs_adjusted.index, y=len(df_blue_hill_1850_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - HadCRUT5-analysis BHO $T_{2m}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
    
    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs 20CRv3 New Haven
    
    print('plotting BHO: GloSAT Tg (adjusted) vs 20CRv3 New Haven ...')
    
    df_blue_hill_1850_1959_tobs_adjusted = df_blue_hill_1811_1959_tobs_adjusted[ df_blue_hill_1811_1959_tobs_adjusted.index>=pd.to_datetime('1850-01-01') ]
    df_hadcrut5_new_haven_1850_1959 = df_hadcrut5_new_haven[(df_hadcrut5_new_haven.index>=pd.to_datetime('1811-01-01')) & (df_hadcrut5_new_haven.index<=pd.to_datetime('1959-06-01'))]
    
    figstr = 'bho-glosat-tg-adjusted-vs-hadcrut5-analysis-new-haven.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs HadCRUT5-analysis New Haven T(2m)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1850_1959_tobs_adjusted.index, y=df_blue_hill_1850_1959_tobs_adjusted['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO (adjusted)')
    axs[0].plot(df_hadcrut5_new_haven_1850_1959.index, df_hadcrut5_new_haven_1850_1959['df_hadcrut5_new_haven'], marker='.', color='blue', alpha=1.0, label='$T_{2m}$ HadCRUT5-analysis New Haven')
    axs[1].plot(df_blue_hill_1850_1959_tobs_adjusted.index, df_blue_hill_1850_1959_tobs_adjusted['blue_hill'].values - df_hadcrut5_new_haven_1850_1959['df_hadcrut5_new_haven'].values, color='teal')

    mean_difference = np.nanmean( df_blue_hill_1850_1959_tobs_adjusted['blue_hill'].values - df_hadcrut5_new_haven_1850_1959['df_hadcrut5_new_haven'].values )
    sns.lineplot(x=df_blue_hill_1850_1959_tobs_adjusted.index, y=len(df_blue_hill_1850_1959_tobs_adjusted)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    plt.axvline(x=pd.Timestamp('1885-01-01'), ls='--', lw=1, color='black')
    plt.text(x=pd.Timestamp('1885-01-01'), y=-15, s='1885-01')

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - HadCRUT5-analysis New Haven $T_{2m}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')            

    # PLOT: BHO: GloSAT Tg (monthly) adjusted vs GHCND Tgm (from daily)
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GHCND Tgm (from daily) USW00014739 (Boston) ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-ghcnd-tgm-boston.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GHCND $T_{g}$ (from daily) USW00014739 (Boston)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1959_2020.index, y=df_blue_hill_1959_2020['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO')
    axs[0].plot(df_USW00014739_1959_2020.index, df_USW00014739_1959_2020['Tgm'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ (from daily) GHCND Boston')
    axs[1].plot(df_blue_hill_1959_2020.index, df_blue_hill_1959_2020['blue_hill'] - df_USW00014739_1959_2020['Tgm'], color='teal')
    
    mean_difference = np.nanmean( df_blue_hill_1959_2020['blue_hill'] - df_USW00014739_1959_2020['Tgm'] )
    sns.lineplot(x=df_blue_hill_1959_2020.index, y=len(df_blue_hill_1959_2020)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - GHCND $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GHCND Tgm (from daily) USW00014765 (Providence, RI) ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-ghcnd-tgm-providence.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GHCND $T_{g}$ (from daily) USW00014739 (Providence)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1959_2020.index, y=df_blue_hill_1959_2020['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO')
    axs[0].plot(df_USW00014765_1959_2020.index, df_USW00014765_1959_2020['Tgm'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ (from daily) GHCND Providence')
    axs[1].plot(df_blue_hill_1959_2020.index, df_blue_hill_1959_2020['blue_hill'] - df_USW00014765_1959_2020['Tgm'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1959_2020['blue_hill'] - df_USW00014765_1959_2020['Tgm'] )
    sns.lineplot(x=df_blue_hill_1959_2020.index, y=len(df_blue_hill_1959_2020)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    

    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - GHCND $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GHCND Tgm (from daily) USW00014758 (New Haven Tweed Airport) ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-ghcnd-tgm-new-haven.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GHCND $T_{g}$ (from daily) USW00014758 (New Haven Tweed Airport)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1959_2020.index, y=df_blue_hill_1959_2020['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO')
    axs[0].plot(df_USW00014758_1959_2020.index, df_USW00014758_1959_2020['Tgm'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ (from daily) GHCND New Haven Tweed Airport')
    axs[1].plot(df_blue_hill_1959_2020.index, df_blue_hill_1959_2020['blue_hill'] - df_USW00014758_1959_2020['Tgm'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1959_2020['blue_hill'] - df_USW00014758_1959_2020['Tgm'] )
    sns.lineplot(x=df_blue_hill_1959_2020.index, y=len(df_blue_hill_1959_2020)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - GHCND $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    
    
    print('plotting BHO: GloSAT Tg (adjusted) vs GHCND Tgm (from daily) USC00065910 (Norwich) ...')
    
    figstr = 'bho-glosat-tg-adjusted-vs-ghcnd-tgm-norwich.png'
    titlestr = 'Blue Hill Observatory: monthly GloSAT $T_g$ (adjusted) vs GHCND $T_{g}$ (from daily) USC00065910 (Norwich)'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_blue_hill_1959_2020.index, y=df_blue_hill_1959_2020['blue_hill'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT BHO')
    axs[0].plot(df_USC00065910_1959_2020.index, df_USC00065910_1959_2020['Tgm'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ (from daily) GHCND Norwich')
    axs[1].plot(df_blue_hill_1959_2020.index, df_blue_hill_1959_2020['blue_hill'] - df_USC00065910_1959_2020['Tgm'], color='teal')

    mean_difference = np.nanmean( df_blue_hill_1959_2020['blue_hill'] - df_USC00065910_1959_2020['Tgm'] )
    sns.lineplot(x=df_blue_hill_1959_2020.index, y=len(df_blue_hill_1959_2020)*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'BHO $T_{g}$ - GHCND $T_{g}$, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    

    print('plotting Amherst: GloSAT (Amherst 1) vs Amherst 2 ...')

    df_amherst_diff = df_amherst2.copy()
    df_amherst_diff['amherst1'] = df_amherst
#   plt.plot(df_amherst_diff['amherst1']-df_amherst_diff['amherst2'])
    
    figstr = 'bho-glosat-tg-amherst1-vs-amherst2.png'
    titlestr = 'Amherst: monthly GloSAT $T_g$ vs Amherst 2'

    fig, axs = plt.subplots(2,1, figsize=(15,10))
    sns.lineplot(x=df_amherst_diff['amherst1'].index, y=df_amherst_diff['amherst1'], ax=axs[0], marker='o', color='red', alpha=1.0, label='$T_{g}$ GloSAT Amherst 1')
    axs[0].plot(df_amherst_diff['amherst2'].index, df_amherst_diff['amherst2'], marker='.', color='blue', alpha=1.0, label='$T_{g}$ Amherst 2')
    axs[1].plot(df_amherst_diff['amherst1'].index, df_amherst_diff['amherst1'] - df_amherst_diff['amherst2'], color='teal')

    mean_difference = np.nanmean( df_amherst_diff['amherst1'] - df_amherst_diff['amherst2'] )
    sns.lineplot(x=df_amherst_diff['amherst1'], y=len(df_amherst_diff['amherst1'])*[ mean_difference ], ls='--', lw=1, color='black', label='$\mu$=' + str(np.round(mean_difference,2)) + '$^{\circ}$F')            
    axs[1].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    
    axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    axs[0].set_xlabel('', fontsize=fontsize)
    axs[0].set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    axs[0].set_title(titlestr, fontsize=fontsize)
    axs[0].tick_params(labelsize=fontsize)    
    axs[1].sharex(axs[0])
    axs[1].tick_params(labelsize=fontsize)    
    axs[1].set_xlabel('Year', fontsize=fontsize)
    axs[1].set_ylabel(r'Amherst 1 - Amherst 2, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    if use_fahrenheit == True:
        axs[0].set_ylim(0,80)
        axs[1].set_ylim(-15,15)
    else:
        axs[0].set_ylim(-20,30)
        axs[1].set_ylim(-1.5,0.5)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')    


#==============================================================================
                      
if plot_differences_heatmap == True:
        
    print('plotting BHO: heatmaps of differences ... ')
            
    # PLOT: heatmap of differences: monthly T2828 - Tg
    
    Fmag = np.round( np.max([np.abs(np.nanmin(differences_T2828_Tg)), np.abs(np.nanmax(differences_T2828_Tg))]), 1)
         
    figstr = 'bho-differences-T2828-Tg-1961-2020.png'
    titlestr = 'BHO: monthly $T_{2828}$ - $T_{g}$ differences used to calculate monthly adjustments'
    
    fig,ax = plt.subplots(figsize=(15,10))
    g = sns.heatmap(differences_T2828_Tg, ax=ax, cmap='coolwarm', vmin=-Fmag, vmax=Fmag,
                cbar_kws={'drawedges': False, 'shrink':0.7, 'extend':'both', 'label':'difference, $^{\circ}F$'})
    ax.set_ylabel('Year', fontsize=fontsize)
    ax.set_xlabel('Month', fontsize=fontsize)
    ax.set_title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr , dpi=300)
    plt.close('all')
    
    # PLOT: heatmap of differences: monthly T2828 - Tgm (i.e. Tg from daily Tn and Tx)
    
    Fmag = np.round( np.max([np.abs(np.nanmin(differences_T2828_Tg)), np.abs(np.nanmax(differences_T2828_Tg))]), 1)
    
    figstr = 'bho-differences-T2828-Tgm-1961-2020.png'
    titlestr = 'BHO: monthly $T_{2828}$ - $T_{g}$ (from daily) differences used to calculate monthly adjustments'
    
    fig,ax = plt.subplots(figsize=(15,10))
    g = sns.heatmap(differences_T2828_Tgm, ax=ax, cmap='coolwarm', vmin=-Fmag, vmax=Fmag,
                cbar_kws={'drawedges': False, 'shrink':0.7, 'extend':'both', 'label':'difference, $^{\circ}F$'})
    ax.set_ylabel('Year', fontsize=fontsize)
    ax.set_xlabel('Month', fontsize=fontsize)
    ax.set_title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr , dpi=300)
    plt.close('all')
    
    # PLOT: heatmap of differences: monthly Tg - Tgm (i.e. Tg from daily Tn and Tx)
    
    Fmag = np.round( np.max([np.abs(np.nanmin(differences_Tg_Tgm)), np.abs(np.nanmax(differences_Tg_Tgm))]), 1)
    
    figstr = 'bho-differences-Tg-Tgm-1961-2020.png'
    titlestr = 'BHO: monthly $T_{g}$ (monthly) - $T_{g}$ (from daily) differences used to calculate monthly adjustments'
    
    fig,ax = plt.subplots(figsize=(15,10))
    g = sns.heatmap(differences_Tg_Tgm, ax=ax, cmap='coolwarm', vmin=-Fmag, vmax=Fmag,
                cbar_kws={'drawedges': False, 'shrink':0.7, 'extend':'both', 'label':'difference, $^{\circ}F$'})
    ax.set_ylabel('Year', fontsize=fontsize)
    ax.set_xlabel('Month', fontsize=fontsize)
    ax.set_title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr , dpi=300)
    plt.close('all')

#==============================================================================    
    
if plot_kde == True: 
    
    # PLOT: BHO: T2828 (monthly) versus Tg (monthly) KDE distribution
    
    print('plotting BHO: T2828 (monthly) versus Tg (monthly) distributions ...')
    
    figstr = 'bho-t2828(monthly)-vs-bho-tg(monthly)-degF-kde.png'
    titlestr = 'Blue Hill Observatory (BHO): $T_{2828}$ (monthly) versus $T_g$ (monthly) distributions'
    
    fig, ax = plt.subplots(figsize=(15,10))
    kwargs = {'levels': np.arange(0, 0.15, 0.01)}
    sns.kdeplot(df_bho_2828['T2828'], color='red', shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{2828}$')
    sns.kdeplot(df_bho_tg['Tg'], color='blue', shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$')
    if use_fahrenheit == True:
        ax.set_xlim(0,90)
        ax.set_ylim(0,0.03)
    else:
        ax.set_xlim(-20,40)
        ax.set_ylim(0,0.05)
    plt.xlabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.ylabel('KDE', fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

    # PLOT: BHO: Tg (from daily) versus Tg (monthly) KDE distribution
    
    print('plotting BHO: Tg (from daily) versus Tg (monthly) distributions ...')
    
    figstr = 'bho-tg(from daily)-vs-bho-tg(monthly)-degF-kde.png'
    titlestr = 'Blue Hill Observatory (BHO): $T_{g}$ (from daily) versus $T_g$ (monthly) distributions'
    
    fig, ax = plt.subplots(figsize=(15,10))
    kwargs = {'levels': np.arange(0, 0.15, 0.01)}
    sns.kdeplot(df_bho_monthly['Tgm'], color='red', shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$ (from daily)')
    sns.kdeplot(df_bho_monthly['Tg'], color='blue', shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$ monthly')
    if use_fahrenheit == True:
        ax.set_xlim(0,90)
        ax.set_ylim(0,0.03)
    else:
        ax.set_xlim(-20,40)
        ax.set_ylim(0,0.05)
    plt.xlabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.ylabel('KDE', fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================
    
if plot_ghcn == True:
    
    # PLOT: Distributions of Tn, Tg and Tx (daily) for neighbouring stations
    
    print('plotting neighbouring station distributions of Tn,Tg and Tx ...')
        
    figstr = 'neighbouring-stations-distributions.png'
    titlestr = 'GHCN-D stations in the Boston area: KDE distributions of monthly $T_{n}$, $T_{g}$ and $T_{x}$'
    
    #fig,ax = plt.subplots(figsize=(15,10))
    #g = sns.FacetGrid(df_neighbouring_stations, col='STATION', col_wrap=4)
    #g.map(sns.kdeplot, 'TMIN', color='blue', shade=True, alpha=0.5, legend=True, label=r'$T_{n}$')
    #g.map(sns.kdeplot, 'TAVG', color='purple', shade=True, alpha=0.5, legend=True, label=r'$T_{g}$')
    #g.map(sns.kdeplot, 'TMAX', color='red', shade=True, alpha=0.5, legend=True, label=r'$T_{x}$')
    #axs = g.axes.flatten()
    #for ax in axs:       
    #    ax.set_xlim(-30,40)
    #    ax.set_ylim(0,0.15)
    #    ax.set_xlabel(r'2m Temperature, $^{\circ}$C', fontsize=12)
    #    ax.set_ylabel('KDE', fontsize=12)
    #    ax.tick_params(labelsize=12)    
    #    ax.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)        
    #g.fig.subplots_adjust(top=0.9)
    #g.fig.suptitle(titlestr, fontsize=fontsize)
    
    ncols = 3; nrows = int(np.ceil(Nstations/ncols)); r = 0
    dg = df_neighbouring_stations.copy().sort_index()
    
    fig,axs = plt.subplots(nrows, ncols, figsize=(15,10))
    for i in range(nrows*ncols):    
        if i > (Nstations-1):
            axs[-1,-1].axis('off')
            continue
        if use_fahrenheit == True:
            ymin = centigrade_to_fahrenheit( dg[dg['STATION']==dg['STATION'].unique()[i]]['TMIN'] )
            ymax = centigrade_to_fahrenheit(dg[dg['STATION']==dg['STATION'].unique()[i]]['TMAX'] )
            yavg = centigrade_to_fahrenheit(dg[dg['STATION']==dg['STATION'].unique()[i]]['TAVG'] )
        else:
            ymin = dg[dg['STATION']==dg['STATION'].unique()[i]]['TMIN']
            ymax = dg[dg['STATION']==dg['STATION'].unique()[i]]['TMAX']
            yavg = dg[dg['STATION']==dg['STATION'].unique()[i]]['TAVG']
        stationcode = dg[dg['STATION']==dg['STATION'].unique()[i]]['STATION'][0]     
        t = dg[dg['STATION']==dg['STATION'].unique()[i]].index
        c = i%ncols
        if (i > 0) & (c == 0):
            r += 1     
        g = sns.kdeplot(ymin, ax=axs[r,c], color='blue', shade=True, alpha=0.5, legend=True, label=r'$T_{n}$')
        sns.kdeplot(yavg, ax=axs[r,c], color='purple', shade=True, alpha=0.5, legend=True, label=r'$T_{g}$')
        sns.kdeplot(ymax, ax=axs[r,c], color='red', shade=True, alpha=0.5, legend=True, label=r'$T_{x}$')   
        if use_fahrenheit == True:
            g.axes.set_xlim(-20,110)
            g.axes.set_ylim(0,0.03)
        else:
            g.axes.set_xlim(-20,40)
            g.axes.set_ylim(0,0.05)    
        if (r+1) == nrows:
            g.axes.set_xlabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
        else:
            g.axes.set_xlabel('', fontsize=fontsize)
            g.axes.set_xticklabels([])  
        if c == 0:
            g.axes.set_ylabel('KDE', fontsize=fontsize)
        else:
            g.axes.set_ylabel('', fontsize=fontsize)
            g.axes.set_yticklabels([])
        g.axes.set_title('STATION='+stationcode, fontsize=fontsize)
        g.axes.tick_params(labelsize=fontsize)    
        g.axes.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)    
        
    fig.subplots_adjust(top=0.9)
    fig.suptitle(titlestr, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: Timeseries of Tn, Tg and Tx (monthly) for neighbouring stations
    
    print('plotting neighbouring station monthly-averaged timeseries of Tn,Tg and Tx ...')
        
    figstr = 'neighbouring-stations-timeseries-1m-average.png'
    titlestr = 'GHCN-D stations in the Boston area: timeseries of monthly-averaged $T_{n}$, $T_{g}$ and $T_{x}$'
    
    ncols = 3; nrows = int(np.ceil(Nstations/ncols)); r = 0
    dg = df_neighbouring_stations.copy().sort_index()
    
    fig,axs = plt.subplots(nrows, ncols, figsize=(15,10))
    for i in range(nrows*ncols):    
        if i > (Nstations-1):
            axs[-1,-1].axis('off')
            continue
        if use_fahrenheit == True:
            ymin = centigrade_to_fahrenheit( dg[dg['STATION']==dg['STATION'].unique()[i]]['TMIN'] ) 
            ymax = centigrade_to_fahrenheit( dg[dg['STATION']==dg['STATION'].unique()[i]]['TMAX'] ) 
            yavg = centigrade_to_fahrenheit( dg[dg['STATION']==dg['STATION'].unique()[i]]['TAVG'] ) 
        else:
            ymin = dg[dg['STATION']==dg['STATION'].unique()[i]]['TMIN']
            ymax = dg[dg['STATION']==dg['STATION'].unique()[i]]['TMAX']
            yavg = dg[dg['STATION']==dg['STATION'].unique()[i]]['TAVG']
        stationcode = dg[dg['STATION']==dg['STATION'].unique()[i]]['STATION'][0]     
        t = dg[dg['STATION']==dg['STATION'].unique()[i]].index
        c = i%ncols
        if (i > 0) & (c == 0):
            r += 1     
        df_monthly = pd.DataFrame({'TMIN':ymin, 'TAVG':yavg, 'TMAX':ymax}, index=t)
        df_monthly_xr = df_monthly.to_xarray()    
        ymin_yearly = df_monthly_xr['TMIN'].resample(datetime='MS').mean().to_dataset() 
        yavg_yearly = df_monthly_xr['TAVG'].resample(datetime='MS').mean().to_dataset() 
        ymax_yearly = df_monthly_xr['TMAX'].resample(datetime='MS').mean().to_dataset() 
        t = pd.date_range(start=str(df_monthly.index[0].year), periods=len(ymin_yearly.TMIN.values), freq='MS')
        g = sns.lineplot(x=t, y=ymin_yearly.TMIN.values, ax=axs[r,c], marker='.', color='blue', alpha=0.5, label='$T_{n}$')
        sns.lineplot(x=t, y=yavg_yearly.TAVG.values, ax=axs[r,c], marker='.', color='purple', alpha=0.5, label='$T_{g}$')
        sns.lineplot(x=t, y=ymax_yearly.TMAX.values, ax=axs[r,c], marker='.', color='red', alpha=0.5, label='$T_{x}$')
        if use_fahrenheit == True:
            g.axes.set_ylim(0,90)
        else:
            g.axes.set_ylim(-20,40)    
        g.axes.set_xlim(pd.Timestamp('1880-01-01'),pd.Timestamp('2020-01-01'))
        if (r+1) == nrows:
            g.axes.set_xlabel('Year', fontsize=fontsize)
        else:
            g.axes.set_xticklabels([])  
        if c == 0:
            g.axes.set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
        else:
            g.axes.set_yticklabels([])
        g.axes.set_title('STATION='+stationcode, fontsize=fontsize)
        g.axes.tick_params(labelsize=fontsize)    
        g.axes.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)    
    
    fig.subplots_adjust(top=0.9)
    fig.suptitle(titlestr, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: Timeseries of Tn, Tg and Tx (2yr=average) for neighbouring stations
    
    print('plotting neighbouring station 2yr-averaged timeseries of Tn,Tg and Tx ...')
        
    figstr = 'neighbouring-stations-timeseries-24m-average.png'
    titlestr = 'GHCN-D stations in the Boston area: timeseries of 24m-averaged $T_{n}$, $T_{g}$ and $T_{x}$'
    
    ncols = 3; nrows = int(np.ceil(Nstations/ncols)); r = 0
    dg = df_neighbouring_stations.copy().sort_index()
    
    fig,axs = plt.subplots(nrows, ncols, figsize=(15,10))
    for i in range(nrows*ncols):    
        if i > (Nstations-1):
            axs[-1,-1].axis('off')
            continue
        if use_fahrenheit == True:
            ymin = centigrade_to_fahrenheit( dg[dg['STATION']==dg['STATION'].unique()[i]]['TMIN'] )
            ymax = centigrade_to_fahrenheit( dg[dg['STATION']==dg['STATION'].unique()[i]]['TMAX'] )
            yavg = centigrade_to_fahrenheit( dg[dg['STATION']==dg['STATION'].unique()[i]]['TAVG'] )
        else:            
            ymin = dg[dg['STATION']==dg['STATION'].unique()[i]]['TMIN']
            ymax = dg[dg['STATION']==dg['STATION'].unique()[i]]['TMAX']
            yavg = dg[dg['STATION']==dg['STATION'].unique()[i]]['TAVG']
        stationcode = dg[dg['STATION']==dg['STATION'].unique()[i]]['STATION'][0]     
        t = dg[dg['STATION']==dg['STATION'].unique()[i]].index
        c = i%ncols
        if (i > 0) & (c == 0):
            r += 1     
        df_monthly = pd.DataFrame({'TMIN':ymin, 'TAVG':yavg, 'TMAX':ymax}, index=t)
        df_monthly_xr = df_monthly.to_xarray()    
        ymin_yearly = df_monthly_xr['TMIN'].resample(datetime='2AS').mean().to_dataset() 
        yavg_yearly = df_monthly_xr['TAVG'].resample(datetime='2AS').mean().to_dataset() 
        ymax_yearly = df_monthly_xr['TMAX'].resample(datetime='2AS').mean().to_dataset() 
        t = pd.date_range(start=str(df_monthly.index[0].year), periods=len(ymin_yearly.TMIN.values), freq='2AS')
        g = sns.lineplot(x=t, y=ymin_yearly.TMIN.values, ax=axs[r,c], marker='.', color='blue', alpha=0.5, label='$T_{n}$')
        sns.lineplot(x=t, y=yavg_yearly.TAVG.values, ax=axs[r,c], marker='.', color='purple', alpha=0.5, label='$T_{g}$')
        sns.lineplot(x=t, y=ymax_yearly.TMAX.values, ax=axs[r,c], marker='.', color='red', alpha=0.5, label='$T_{x}$')
        if use_fahrenheit == True:
            g.axes.set_ylim(0,90)
        else:
            g.axes.set_ylim(-20,40)    
        g.axes.set_xlim(pd.Timestamp('1880-01-01'),pd.Timestamp('2020-01-01'))
        if (r+1) == nrows:
            g.axes.set_xlabel('Year', fontsize=fontsize)
        else:
            g.axes.set_xticklabels([])  
        if c == 0:
            g.axes.set_ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
        else:
            g.axes.set_yticklabels([])
        g.axes.set_title('STATION='+stationcode, fontsize=fontsize)
        g.axes.tick_params(labelsize=fontsize)    
        g.axes.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)    
    
    fig.subplots_adjust(top=0.9)
    fig.suptitle(titlestr, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================
    
if plot_inventory == True:

    # PLOT: inventory
    
    print('plotting station inventory chart ...')
        
    dg = df_neighbouring_stations.copy().sort_index()
    stationcodes = dg['STATION'].unique()
    categories = stationcodes

    sequential_colors = sns.color_palette(color_palette, 1) # single colour fill
    sequential_colors = sns.color_palette(color_palette, len(categories))
    sns.set_palette(sequential_colors)

    cat_dict = dict(zip(categories, range(1, len(categories)+1)))   # MAP: categories to y-values
    val_dict = dict(zip(range(1, len(categories)+1), categories))   # MAP: y-values to categories
    dates = dg.index
    values = [dg['STATION'][i] for i in range(len(dates))]
    #dates = pd.date_range(start='2021-01-01 00:00', end='2021-05-01 00:00', freq='1D')
    #random_sanple = [random.randint(1, len(categories)) for p in range(1,len(dates)+1)]
    #random_sample = [random.randint(1,2) for p in range(1,len(dates)+1)]
    #values = [val_dict[random_sample[i]] for i in range(len(dates))]
    df = pd.DataFrame(data=values, index=dates, columns=['category'])
    df['plotval'] = df['category'].apply(cat_dict.get) # get y-values from categories
    cmap = matplotlib.cm.get_cmap('viridis')                        # SAMPLE: discrete colours from colornap
    colsteps = np.linspace(0,1,len(categories))
    colors = [ cmap(colsteps[i]) for i in range(len(colsteps)) ]
    colors = len(colsteps)*[colors[len(colsteps)%2]]                # EXTRACT: single colour from middle of colormap
    col_dict = dict(zip(range(1, len(categories)+1), colors))       # MAP: y-values to categories
    #color_mapper = np.vectorize(lambda x: {1: 'red', 2: 'blue'}.get(x))
    color_mapper = np.vectorize(lambda x: col_dict.get(x))
    colors = [np.array(color_mapper(df['plotval'][i])) for i in range(len(df))]
    
    figstr = 'neighbouring-stations-inventory.png'
    titlestr = 'GHCN-D stations in the Boston area: inventory bar chart in the style of Havens (1958)'
        
    fig,ax = plt.subplots(figsize=(15,10))
    for i in np.arange(0,len(df),31):    
        plt.plot(df.index[i], df['plotval'][i], marker='s', markersize=10, color=colors[i])
    ax.set_yticks(range(len(categories)+1))        
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: val_dict.get(x))) # format y-ticks using category LUT        
    ax.invert_yaxis()
    plt.tick_params(labelsize=fontsize)    
    plt.title(titlestr, fontsize=fontsize)
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================

if plot_glosat_neighbours == True:

    # PLOT: Neighbouring stations in the Boston area: GloSAT (monthly) timeseries
    
    print('plotting GloSAT stations in the Boston area timeseries ... ')
    
    sequential_colors = sns.color_palette(color_palette, 14) # 14 stations
    sns.set_palette(sequential_colors)
    
    figstr = 'glosat-tg(monthly)-timeseries.png'
    titlestr = 'GloSAT stations in the Boston area: $T_g$ (monthly) and early observations'
    
    fig, ax = plt.subplots(figsize=(15,10))
        
    sns.lineplot(x=df_amherst.index, y=df_amherst['amherst'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_bedford.index, y=df_bedford['bedford'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_blue_hill.index, y=df_blue_hill['blue_hill'], marker='.', color='red', alpha=0.5, legend=False)
    sns.lineplot(x=df_boston_city_wso.index, y=df_boston_city_wso['boston_city_wso'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_kingston.index, y=df_kingston['kingston'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_lawrence.index, y=df_lawrence['lawrence'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_new_bedford.index, y=df_new_bedford['new_bedford'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_new_haven.index, y=df_new_haven['new_haven'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_plymouth_kingston.index, y=df_plymouth_kingston['plymouth_kingston'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_providence_wso.index, y=df_providence_wso['providence_wso'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_provincetown.index, y=df_provincetown['provincetown'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_reading.index, y=df_reading['reading'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_taunton.index, y=df_taunton['taunton'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_walpole_2.index, y=df_walpole_2['walpole_2'], marker='.', alpha=0.5, legend=False)
    sns.lineplot(x=df_west_medway.index, y=df_west_medway['west_medway'], marker='.', alpha=0.5, legend=False)
     
    sns.lineplot(x=df_amherst.index, y=(pd.Series(df_amherst['amherst']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True,  label=r'$T_{g}$ 2yr MA: Amherst')
    sns.lineplot(x=df_bedford.index, y=(pd.Series(df_bedford['bedford']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Bedford')
    sns.lineplot(x=df_blue_hill.index, y=(pd.Series(df_blue_hill['blue_hill']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, color='red', zorder=10, legend=True, label=r'$T_{g}$ 2yr MA: Blue Hill')
    sns.lineplot(x=df_boston_city_wso.index, y=(pd.Series(df_boston_city_wso['boston_city_wso']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Boston City WSO')
    sns.lineplot(x=df_kingston.index, y=(pd.Series(df_kingston['kingston']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Kingston')
    sns.lineplot(x=df_lawrence.index, y=(pd.Series(df_lawrence['lawrence']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True,  label=r'$T_{g}$ 2yr MA: Lawrence')
    sns.lineplot(x=df_new_bedford.index, y=(pd.Series(df_new_bedford['new_bedford']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: New Bedford')
    sns.lineplot(x=df_new_haven.index, y=(pd.Series(df_new_haven['new_haven']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: New Haven')
    sns.lineplot(x=df_plymouth_kingston.index, y=(pd.Series(df_plymouth_kingston['plymouth_kingston']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Plymouth-Kingston')
    sns.lineplot(x=df_providence_wso.index, y=(pd.Series(df_providence_wso['providence_wso']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Povidence WSO')
    sns.lineplot(x=df_provincetown.index, y=(pd.Series(df_provincetown['provincetown']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Provincetown')
    sns.lineplot(x=df_reading.index, y=(pd.Series(df_reading['reading']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Reading')
    sns.lineplot(x=df_taunton.index, y=(pd.Series(df_taunton['taunton']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Taunton')
    sns.lineplot(x=df_walpole_2.index, y=(pd.Series(df_walpole_2['walpole_2']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: Walpole 2')
    sns.lineplot(x=df_west_medway.index, y=(pd.Series(df_west_medway['west_medway']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, legend=True, label=r'$T_{g}$ 2yr MA: West Medway')
    if use_fahrenheit == True:
        ax.set_ylim(0,90)
    else:
        ax.set_ylim(-20,40)    
    plt.xlabel('', fontsize=fontsize)
    plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='lower right', ncol=3, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    # PLOT: Neighbouring stations in the Boston area: GloSAT (monthly) KDE distributions
    
    print('plotting neighbouring stations in the Boston area distributions ... ')
    
    sequential_colors = sns.color_palette(color_palette, 14) # 14 stations
    sns.set_palette(sequential_colors)
        
    figstr = 'glosat-tg(monthly)-kde.png'
    titlestr = 'GloSAT stations in the Boston area: $T_g$ (monthly)'
    
    fig, ax = plt.subplots(figsize=(15,10))
    kwargs = {'levels': np.arange(0, 0.15, 0.01)}
    sns.kdeplot(df_amherst['amherst'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Amherst')
    sns.kdeplot(df_bedford['bedford'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Bedford')
    sns.kdeplot(df_blue_hill['blue_hill'], shade=True, alpha=0.2, color = 'red', legend=True, zorder=10, **kwargs, label=r'$T_{g}$: Blue Hill')
    sns.kdeplot(df_boston_city_wso['boston_city_wso'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Boston City WSO')
    sns.kdeplot(df_kingston['kingston'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Kingston')
    sns.kdeplot(df_lawrence['lawrence'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Lawrence')
    sns.kdeplot(df_new_bedford['new_bedford'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: New Bedford')
    sns.kdeplot(df_new_haven['new_haven'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: New Haven')
    sns.kdeplot(df_plymouth_kingston['plymouth_kingston'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Plymouth-Kingston')
    sns.kdeplot(df_providence_wso['providence_wso'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Povidence WSO')
    sns.kdeplot(df_provincetown['provincetown'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Provincetown')
    sns.kdeplot(df_reading['reading'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Reading')
    sns.kdeplot(df_taunton['taunton'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Taunton')
    sns.kdeplot(df_walpole_2['walpole_2'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: Walpole 2')
    sns.kdeplot(df_west_medway['west_medway'], shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$: West Medway')
    if use_fahrenheit == True:
        ax.set_xlim(-20,110)
        ax.set_ylim(0,0.03)
    else:
        ax.set_xlim(-20,40)
        ax.set_ylim(0,0.05)    
    plt.xlabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.ylabel('KDE', fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================
    
if plot_bho_all_sources == True:
    
    # PLOT: BHO timeseries from all sources (no Tx or Tn)
    
    print('plotting BHO: timeseries (all sources) ... ')
    
    sequential_colors = sns.color_palette(color_palette, 5)
    sns.set_palette(sequential_colors)
    
    figstr = 'bho-timeseries-all-sources.png'
    titlestr = 'BHO: all sources'
    
    fig, ax = plt.subplots(figsize=(15,10))
    
    # PLOT: 2yr MA smoothed reanalysis
    
    plt.plot(df_20CRv3_new_haven.index, (pd.Series(df_20CRv3_new_haven['T(2m)']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='--', lw=1, color='black', alpha=1, zorder=5, label=r'$T_{g}$ 2yr MA: 20CRv3 (2m) at New Haven')
    
    # PLOT: 2yr MA smoothed timeseries
    
    plt.plot(df_new_haven.index, (pd.Series(df_new_haven['new_haven']).rolling(24,center=True).mean()), '.', color='lime', alpha=1, zorder=5, label=r'$T_{g}$ 2yr MA: New Haven (reference)')
    plt.plot(df_bho_2828.index, (pd.Series(df_bho_2828['T2828']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=5, color='red', alpha=1, zorder=5, label=r'$T_{2828}$ 2yr MA: BHO')
    plt.plot(df_bho_2828.index, (pd.Series(df_bho_tg['Tg']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=5, color='blue', alpha=1, zorder=5, label=r'$T_{g}$ 2yr MA: BHO')
    plt.plot(df_blue_hill.index, (pd.Series(df_blue_hill['blue_hill']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=5, color='green', alpha=1, zorder=5, label=r'$T_{g}$ 2yr MA: GloSAT')
    plt.plot(df_ghcnmv4_qcf.index, (pd.Series(df_ghcnmv4_qcf['df_ghcnmv4_qcf']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=2, color='black', zorder=10, label=r'$T_{g}$ 2yr MA: GHCNMv4 (QCF)')
    plt.fill_between(df_ghcnmv4_qcu.index, 6, (pd.Series(df_ghcnmv4_qcu['df_ghcnmv4_qcu']).rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), color='black', alpha=0.1, zorder=3, label=r'$T_{g}$ 2yr MA: GHCNMv4 (QCU)')
    
    # PLOT: segment means
    
    mask_pre_1891 = df_new_haven.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_new_haven.index >= pd.Timestamp('1891-01-01')) & (df_new_haven.index < pd.Timestamp('1959-06-01'))
    mask_post_1959 = df_new_haven.index >= pd.Timestamp('1959-06-01')
    plt.plot(df_new_haven.index[mask_pre_1891], mask_pre_1891.sum()*[df_new_haven['new_haven'][mask_pre_1891].dropna().mean()], ls='--', lw=1, color='lime')            
    plt.plot(df_new_haven.index[mask_1891_1959], mask_1891_1959.sum()*[df_new_haven['new_haven'][mask_1891_1959].dropna().mean()], ls='--', lw=1, color='lime')            
    plt.plot(df_new_haven.index[mask_post_1959], mask_post_1959.sum()*[df_new_haven['new_haven'][mask_post_1959].dropna().mean()], ls='--', lw=1, color='lime')            
    
    mask_pre_1891 = df_bho_2828.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_bho_2828.index >= pd.Timestamp('1891-01-01')) & (df_bho_2828.index < pd.Timestamp('1959-06-01'))
    mask_post_1959 = df_bho_2828.index >= pd.Timestamp('1959-06-01')
    plt.plot(df_bho_2828.index[mask_pre_1891], mask_pre_1891.sum()*[df_bho_2828['T2828'][mask_pre_1891].dropna().mean()], ls='--', lw=1, color='red')            
    plt.plot(df_bho_2828.index[mask_1891_1959], mask_1891_1959.sum()*[df_bho_2828['T2828'][mask_1891_1959].dropna().mean()], ls='--', lw=1, color='red')            
    plt.plot(df_bho_2828.index[mask_post_1959], mask_post_1959.sum()*[df_bho_2828['T2828'][mask_post_1959].dropna().mean()], ls='--', lw=1, color='red')            
    plt.plot(df_bho_2828.index[mask_pre_1891], mask_pre_1891.sum()*[df_bho_tg['Tg'][mask_pre_1891].dropna().mean()], ls='--', lw=1, color='blue')            
    plt.plot(df_bho_2828.index[mask_1891_1959], mask_1891_1959.sum()*[df_bho_tg['Tg'][mask_1891_1959].dropna().mean()], ls='--', lw=1, color='blue')            
    plt.plot(df_bho_2828.index[mask_post_1959], mask_post_1959.sum()*[df_bho_tg['Tg'][mask_post_1959].dropna().mean()], ls='--', lw=1, color='blue')            
    
    mask_pre_1891 = df_blue_hill.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_blue_hill.index >= pd.Timestamp('1891-01-01')) & (df_blue_hill.index < pd.Timestamp('1959-06-01'))
    mask_post_1959 = df_blue_hill.index >= pd.Timestamp('1959-06-01')
    plt.plot(df_blue_hill.index[mask_pre_1891], mask_pre_1891.sum()*[df_blue_hill['blue_hill'][mask_pre_1891].dropna().mean()], ls='--', lw=1, color='green')            
    plt.plot(df_blue_hill.index[mask_1891_1959], mask_1891_1959.sum()*[df_blue_hill['blue_hill'][mask_1891_1959].dropna().mean()], ls='--', lw=1, color='green')            
    plt.plot(df_blue_hill.index[mask_post_1959], mask_post_1959.sum()*[df_blue_hill['blue_hill'][mask_post_1959].dropna().mean()], ls='--', lw=1, color='green')            
    
    mask_pre_1891 = df_ghcnmv4_qcf.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_ghcnmv4_qcf.index >= pd.Timestamp('1891-01-01')) & (df_ghcnmv4_qcf.index < pd.Timestamp('1959-06-01'))
    mask_post_1959 = df_ghcnmv4_qcf.index >= pd.Timestamp('1959-06-01')
    plt.plot(df_ghcnmv4_qcf.index[mask_pre_1891], mask_pre_1891.sum()*[df_ghcnmv4_qcf['df_ghcnmv4_qcf'][mask_pre_1891].dropna().mean()], ls='--', lw=1, color='black')            
    plt.plot(df_ghcnmv4_qcf.index[mask_1891_1959], mask_1891_1959.sum()*[df_ghcnmv4_qcf['df_ghcnmv4_qcf'][mask_1891_1959].dropna().mean()], ls='--', lw=1, color='black')            
    plt.plot(df_ghcnmv4_qcf.index[mask_post_1959], mask_post_1959.sum()*[df_ghcnmv4_qcf['df_ghcnmv4_qcf'][mask_post_1959].dropna().mean()], ls='--', lw=1, color='black')            
    
    mask_pre_1891 = df_ghcnmv4_qcu.index < pd.Timestamp('1891-01-01')
    mask_1891_1959 = (df_ghcnmv4_qcu.index >= pd.Timestamp('1891-01-01')) & (df_ghcnmv4_qcu.index < pd.Timestamp('1959-06-01'))
    mask_post_1959 = df_ghcnmv4_qcu.index >= pd.Timestamp('1959-06-01')
    plt.plot(df_ghcnmv4_qcu.index[mask_pre_1891], mask_pre_1891.sum()*[df_ghcnmv4_qcu['df_ghcnmv4_qcu'][mask_pre_1891].dropna().mean()], ls='--', lw=1, color='black', alpha=0.3)            
    plt.plot(df_ghcnmv4_qcu.index[mask_1891_1959], mask_1891_1959.sum()*[df_ghcnmv4_qcu['df_ghcnmv4_qcu'][mask_1891_1959].dropna().mean()], ls='--', lw=1, color='black', alpha=0.3)            
    plt.plot(df_ghcnmv4_qcu.index[mask_post_1959], mask_post_1959.sum()*[df_ghcnmv4_qcu['df_ghcnmv4_qcu'][mask_post_1959].dropna().mean()], ls='--', lw=1, color='black', alpha=0.3)            
    
    # PLOT: breakpoints
    
    plt.axvline(x=pd.Timestamp('1891-01-01'), ls='--', lw=1, color='black', zorder=20)
    plt.axvline(x=pd.Timestamp('1959-06-01'), ls='--', lw=1, color='black', zorder=20)
    
    if use_fahrenheit == True:
        ax.set_ylim(40,60)
    else:
        ax.set_ylim(-20,40)
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================
    
if plot_glosat_adjusted_vs_neighbours == True:
        
    print('plotting BHO: Tobs-adjusted GloSAT versus neighbours ... ')

    # PLOT: BHO Tobs-adjusted GloSAT timeseries versus neighbours
                
    sequential_colors = sns.color_palette(color_palette, 3)
    sns.set_palette(sequential_colors)
    
    figstr = 'bho-glosat-tobs-adjusted-versus-glosat-unadjusted-neighbours.png'
    titlestr = 'BHO: $T_{obs}$-adjusted GloSAT versus GloSAT unadjusted neighbours'
    
    fig, ax = plt.subplots(figsize=(15,10))       
    plt.plot(df_boston_city_wso.index, pd.Series(df_boston_city_wso['boston_city_wso']).rolling(24,center=True).mean(), 'o', alpha=0.2, zorder=5, label=r'$T_{g}$ 2yr MA: Boston City WSO (reference)')
    plt.plot(df_new_haven.index, pd.Series(df_new_haven['new_haven']).rolling(24,center=True).mean(), 'o', alpha=0.2, zorder=5, label=r'$T_{g}$ 2yr MA: New Haven (reference)')
    plt.plot(df_providence_wso.index, pd.Series(df_providence_wso['providence_wso']).rolling(24,center=True).mean(), 'o', alpha=0.2, zorder=5, label=r'$T_{g}$ 2yr MA: Providence WSO (reference)') 
    sns.lineplot(x=df_blue_hill_1811_2020_tobs_adjusted.dropna().index, y=(pd.Series(df_blue_hill_1811_2020_tobs_adjusted['blue_hill'].values).dropna().rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean(), ls='-', lw=3, color='red', alpha=1, zorder=20, label='$T_{g}$: 2yr MA: Tobs-adjusted GloSAT: 1811-2020')
    sns.lineplot(x=df_blue_hill_1811_2020_tobs_adjusted.dropna().index, y=(pd.Series(df_blue_hill_1811_2020_tobs_adjusted['blue_hill'].values).dropna().rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean() + (normal_boston_city_wso-normal_blue_hill), ls='-', lw=3, color='lime', alpha=1, zorder=20, label='$T_{g}$: 2yr MA: Tobs-adjusted GloSAT: 1811-2020 (shifted to Boston City WSO normal)')
    sns.lineplot(x=df_blue_hill_1811_2020_tobs_adjusted.dropna().index, y=(pd.Series(df_blue_hill_1811_2020_tobs_adjusted['blue_hill'].values).dropna().rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean() + (normal_new_haven-normal_blue_hill), ls='-', lw=3, color='green', alpha=1, zorder=20, label='$T_{g}$: 2yr MA: Tobs-adjusted GloSAT: 1811-2020 (shifted to New Haven normal)')
    sns.lineplot(x=df_blue_hill_1811_2020_tobs_adjusted.dropna().index, y=(pd.Series(df_blue_hill_1811_2020_tobs_adjusted['blue_hill'].values).dropna().rolling(24,center=True).mean()).ewm(span=24, adjust=False).mean() + (normal_providence_wso-normal_blue_hill), ls='-', lw=3, color='navy', alpha=1, zorder=20, label='$T_{g}$: 2yr MA: Tobs-adjusted GloSAT: 1811-2020 (shifted to Providence WSO normal)')
    
    if use_fahrenheit == True:
        ax.set_ylim(40,60)
    else:
        ax.set_ylim(-20,40)
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
   
#==============================================================================
    
if plot_glosat_adjusted_with_back_extension == True:
        
    print('plotting GloSAT adjusted timeseries with back-extension ... ')   
 
    nsmooth = 60
       
    figstr = 'bho-glosat-tobs-adjusted-with-back-extension.png'
    titlestr = 'BHO: $T_{obs}$-adjusted GloSAT with back-extension'
               
    fig, ax = plt.subplots(figsize=(15,10))    
    plt.plot(df_boston_city_wso.index, pd.Series(df_boston_city_wso['boston_city_wso']).rolling(nsmooth,center=True).mean() - (normal_boston_city_wso - normal_blue_hill), ls='-', lw=3, color='lime', alpha=1, zorder=5, label=r'$T_{g}$ 2yr MA: Boston City WSO (reference) - $\Delta$(1961-1990)=' + str(np.round((normal_boston_city_wso - normal_blue_hill),2)) + '$^{\circ}$' + temperature_unit)
#    plt.plot(df_holyoke_919.index, pd.Series(df_holyoke_919['T(919)']).rolling(nsmooth,center=True).mean() - (normal_boston_city_wso - normal_blue_hill), 'o', markersize=10, color='green', alpha=0.2, zorder=3, label=r'$T_{g}$ 2yr MA: Salem, MA (Holyoke) - $\Delta$(1961-1990)=' + str(np.round((normal_boston_city_wso - normal_blue_hill),2)) + '$^{\circ}$' + temperature_unit)
#    plt.plot(df_wigglesworth_919.index, pd.Series(df_wigglesworth_919['T(919)']).rolling(nsmooth,center=True).mean() - (normal_boston_city_wso - normal_blue_hill), 'o', markersize=10, color='teal', alpha=0.2, zorder=3, label=r'$T_{g}$ 2yr MA: Salem, MA (Wigglesworth) - $\Delta$(1961-1990)=' + str(np.round((normal_boston_city_wso - normal_blue_hill),2)) + '$^{\circ}$' + temperature_unit)
    plt.plot(df_farrar_919.index, pd.Series(df_farrar_919['T(919)']).rolling(nsmooth,center=True).mean() - (normal_boston_city_wso - normal_blue_hill), 'o', markersize=10, color='navy', alpha=0.5, zorder=3, label=r'$T_{g}$ 2yr MA: Cambridge, MA (Farrar) - $\Delta$(1961-1990)=' + str(np.round((normal_boston_city_wso - normal_blue_hill),2)) + '$^{\circ}$' + temperature_unit)
    plt.plot(df_blue_hill_1790_2020_tobs_adjusted.index, (pd.Series(df_blue_hill_1790_2020_tobs_adjusted['blue_hill'].values).rolling(nsmooth,center=True).mean()), ls='-', lw=3, color='red', alpha=1, zorder=20, label='$T_{g}$: 2yr MA: Tobs-adjusted GloSAT: 1790-2020')
#    ax.set_xlim(pd.to_datetime('1743-01-01'),pd.to_datetime('1885-01-01'))    
    if use_fahrenheit == True:
        ax.set_ylim(40,55)
    else:
        ax.set_ylim(-20,40)
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================
    
if plot_glosat_adjusted_with_back_extension_vs_cet == True:
        
    print('plotting GloSAT adjusted timeseries with back-extension vs CET ... ')   
        
    nsmooth = 60
    
    figstr = 'bho-glosat-tobs-adjusted-with-back-extension-cet.png'
    titlestr = 'BHO: $T_{obs}$-adjusted GloSAT with back-extension vs CET and regional timeseries'
            
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(convert_datetime_to_year_decimal(df_boston_city_wso, 'datetime'), pd.Series(df_boston_city_wso['boston_city_wso']).rolling(nsmooth,center=True).mean() - (normal_boston_city_wso - normal_blue_hill), ls='-', lw=3, color='lime', alpha=1, zorder=5, label=r'$T_{g}$ 5yr MA: Boston City WSO (reference) - $\Delta$(1961-1990)=' + str(np.round((normal_boston_city_wso - normal_blue_hill),2)) + '$^{\circ}$' + temperature_unit)
    plt.plot(convert_datetime_to_year_decimal(df_st_lawrence_valley, 'datetime'), pd.Series(df_st_lawrence_valley['df_st_lawrence_valley']).rolling(nsmooth,center=True).mean() - (normal_st_lawrence_valley - normal_blue_hill), ls='-', lw=3, color='cyan', alpha=1, zorder=5, label=r'$T_{g}$ 5yr MA: St Lawrence Valley (reference) - $\Delta$(1961-1990)=' + str(np.round((normal_st_lawrence_valley - normal_blue_hill),2)) + '$^{\circ}$' + temperature_unit)
    plt.plot(convert_datetime_to_year_decimal(df_amherst, 'datetime'), pd.Series(df_amherst['amherst']).rolling(nsmooth,center=True).mean() - (normal_amherst - normal_blue_hill), ls='-', lw=3, color='blue', alpha=1, zorder=5, label=r'$T_{g}$ 5yr MA: Amherst 1 (reference) - $\Delta$(1961-1990)=' + str(np.round((normal_amherst - normal_blue_hill),2)) + '$^{\circ}$' + temperature_unit)
    plt.plot(convert_datetime_to_year_decimal(df_amherst2, 'datetime'), pd.Series(df_amherst2['amherst2']).rolling(nsmooth,center=True).mean() - (normal_amherst2 - normal_blue_hill), ls='-', lw=3, color='navy', alpha=1, zorder=5, label=r'$T_{g}$ 5yr MA: Amherst 2 (reference) - $\Delta$(1961-1990)=' + str(np.round((normal_amherst2 - normal_blue_hill),2)) + '$^{\circ}$' + temperature_unit)
    plt.plot(convert_datetime_to_year_decimal(df_blue_hill_1790_2020_tobs_adjusted, 'datetime'), (pd.Series(df_blue_hill_1790_2020_tobs_adjusted['blue_hill'].values).rolling(nsmooth,center=True).mean()), ls='-', lw=3, color='red', alpha=1, zorder=20, label='$T_{g}$: 5yr MA: CNET: 1790-2020')
    plt.plot(df_cet.index, pd.Series(df_cet['df_cet']).rolling(nsmooth,center=True).mean(), ls='-', lw=3, color='teal', label='$T_{g}$: 5yr MA: CET: 1659-2020')
    if use_fahrenheit == True:
        ax.set_ylim(40,55)
    else:
        ax.set_ylim(-20,40)
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#==============================================================================
    
if plot_glosat_adjusted_with_back_extension_vs_cet_anomalies == True:
        
    print('plotting GloSAT adjusted timeseries with back-extension vs CET and regional timeseries anomalies ... ')   
        
    nsmooth = 60
    
    figstr = 'bho-glosat-tobs-adjusted-with-back-extension-cet-anomalies.png'
    titlestr = 'BHO: $T_{obs}$-adjusted GloSAT with back-extension vs CET and regional timeseries'
            
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(convert_datetime_to_year_decimal(df_boston_city_wso, 'datetime'), pd.Series(df_boston_city_wso['boston_city_wso'] - normal_boston_city_wso).rolling(nsmooth,center=True).mean(), ls='-', lw=3, color='lime', alpha=1, zorder=5, label=r'$T_{g}$ 5yr MA: Boston City WSO (reference) anomalies (from 1961-1990)')
    plt.plot(convert_datetime_to_year_decimal(df_st_lawrence_valley, 'datetime'), pd.Series(df_st_lawrence_valley['df_st_lawrence_valley'] - normal_st_lawrence_valley).rolling(nsmooth,center=True).mean(), ls='-', lw=3, color='cyan', alpha=1, zorder=5, label=r'$T_{g}$ 5yr MA: St Lawrence Valley (reference) anaomlies (from 1961-1990)')
    plt.plot(convert_datetime_to_year_decimal(df_amherst, 'datetime'), pd.Series(df_amherst['amherst'] - normal_amherst).rolling(nsmooth,center=True).mean(), ls='-', lw=3, color='blue', alpha=1, zorder=5, label=r'$T_{g}$ 5yr MA: Amherst 1 (reference) anomalies (from 1961-1990)')
    plt.plot(convert_datetime_to_year_decimal(df_amherst2, 'datetime'), pd.Series(df_amherst2['amherst2'] - normal_amherst2).rolling(nsmooth,center=True).mean(), ls='-', lw=3, color='navy', alpha=1, zorder=5, label=r'$T_{g}$ 5yr MA: Amherst 2 (reference) anomalies (from 1961-1990)')
    plt.plot(convert_datetime_to_year_decimal(df_blue_hill_1790_2020_tobs_adjusted, 'datetime'), (pd.Series(df_blue_hill_1790_2020_tobs_adjusted['blue_hill'].values - normal_blue_hill).rolling(nsmooth,center=True).mean()), ls='-', lw=3, color='red', alpha=1, zorder=20, label='$T_{g}$: 5yr MA: CNET: 1790-2020 anomalies (from 1961-1990)')
    plt.plot(df_cet.index, pd.Series(df_cet['df_cet'] - normal_cet).rolling(nsmooth,center=True).mean(), ls='-', lw=3, color='teal', label='$T_{g}$: 5yr MA: CET: 1659-2020 anomalies (from 1961-1990)')
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'2m Temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper left', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
        
#==============================================================================
    
if save_glosat_adjustments == True:
        
    print('saving GloSAT timeseries in CRUTEM format ... ')
    
    #------------------------------------------------------------------------------
    # GENERATE: output files in CRUTEM format
    #------------------------------------------------------------------------------
    # FORMAT: station header components in CRUTEM format
    #
    # 37401 525  -17  100 HadCET on 29-11-19   UK            16592019  351721     NAN
    #2019   40   66   78   91  111  141  175  171  143  100 -999 -999
    #------------------------------------------------------------------------------
    
    # GloSAT: unadjusted 1811-2020 (degF)
    
    yearlist = df_blue_hill.index.year.unique()
    df_blue_hill_unadjusted = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df_blue_hill_unadjusted['year'] = yearlist
    for j in range(1,13):
        df_blue_hill_unadjusted[str(j)] = df_blue_hill[df_blue_hill.index.month == j]['blue_hill'].values        
    stationfile = 'bho-glosat-unadjusted-1811-2020.csv'
    station_data = df_blue_hill_unadjusted.iloc[:,range(0,13)].reset_index(drop=True)
    station_metadata = da_blue_hill.iloc[0,range(14,23)]
    stationcode = stationcode_blue_hill
    stationlat = "{:<4}".format(str(int(station_metadata[0]*10)))
    stationlon = "{:<4}".format(str(int(station_metadata[1]*10)))
    stationelevation = "{:<3}".format(str(station_metadata[2]))
    stationname = "{:<20}".format(station_metadata[3][:20])
    stationcountry = "{:<13}".format(station_metadata[4][:13])
    stationfirstlast = str(station_metadata[5]) + str(station_metadata[6])
    stationsourcefirst = "{:<8}".format(str(station_metadata[7]) + str(station_metadata[8]))
    stationgridcell = "{:<3}".format('NAN')
    station_header = ' ' + stationcode[0:] + ' ' + stationlat + ' ' + stationlon + ' ' + stationelevation + ' ' + stationname + ' ' + stationcountry + ' ' + stationfirstlast + '  ' + stationsourcefirst + '   ' + stationgridcell 
    with open(stationfile,'w') as f:
        f.write(station_header+'\n')
        for i in range(len(station_data)):  
            year = str(int(station_data.iloc[i,:][0]))
            rowstr = year
            for j in range(1,13):
                if np.isnan(station_data.iloc[i,:][j]):
                    monthstr = str(-99.9)
                else:
                    monthstr = str(np.round(station_data.iloc[i,:][j],1))
                rowstr += f"{monthstr:>5}"          
            f.write(rowstr+'\n')
    
    # GloSAT: monthly adjusted using T2828 (1961-2020): 1811-2020 (degF)
                    
    yearlist = df_blue_hill_1811_2020_tobs_adjusted.index.year.unique()
    df_blue_hill_adjusted = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df_blue_hill_adjusted['year'] = yearlist
    for j in range(1,13):
        df_blue_hill_adjusted[str(j)] = df_blue_hill_1811_2020_tobs_adjusted[df_blue_hill_1811_2020_tobs_adjusted.index.month == j]['blue_hill'].values        
    stationfile = 'bho-glosat-adjusted-1811-2020.csv'
    station_data = df_blue_hill_adjusted.iloc[:,range(0,13)].reset_index(drop=True)
    station_metadata = da_blue_hill.iloc[0,range(14,23)]
    stationcode = stationcode_blue_hill
    stationlat = "{:<4}".format(str(int(station_metadata[0]*10)))
    stationlon = "{:<4}".format(str(int(station_metadata[1]*10)))
    stationelevation = "{:<3}".format(str(station_metadata[2]))
    stationname = "{:<20}".format(station_metadata[3][:20])
    stationcountry = "{:<13}".format(station_metadata[4][:13])
    stationfirstlast = str(station_metadata[5]) + str(station_metadata[6])
    stationsourcefirst = "{:<8}".format(str(station_metadata[7]) + str(station_metadata[8]))
    stationgridcell = "{:<3}".format('NAN')
    station_header = ' ' + stationcode[0:] + ' ' + stationlat + ' ' + stationlon + ' ' + stationelevation + ' ' + stationname + ' ' + stationcountry + ' ' + stationfirstlast + '  ' + stationsourcefirst + '   ' + stationgridcell 
    with open(stationfile,'w') as f:
        f.write(station_header+'\n')
        for i in range(len(station_data)):  
            year = str(int(station_data.iloc[i,:][0]))
            rowstr = year
            for j in range(1,13):
                if np.isnan(station_data.iloc[i,:][j]):
                    monthstr = str(-99.9)
                else:
                    monthstr = str(np.round(station_data.iloc[i,:][j],1))
                rowstr += f"{monthstr:>5}"          
            f.write(rowstr+'\n')
                    
    # GloSAT: monthly adjusted differences GloSAT adjusted - GloSAT unadjusted: 1811-2020 (degF)
                    
    yearlist = df_blue_hill_1811_2020_tobs_adjusted.index.year.unique()
    df_blue_hill_difference = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df_blue_hill_difference['year'] = yearlist
    for j in range(1,13):
        df_blue_hill_difference[str(j)] = ( df_blue_hill_1811_2020_tobs_adjusted[df_blue_hill_1811_2020_tobs_adjusted.index.month == j]['blue_hill'].values - df_blue_hill[df_blue_hill.index.month == j]['blue_hill'].values )
        
    stationfile = 'bho-difference-glosat-adjusted-minus-unadjusted-1811-2020.csv'
    station_data = df_blue_hill_difference.iloc[:,range(0,13)].reset_index(drop=True)
    station_metadata = da_blue_hill.iloc[0,range(14,23)]
    stationcode = stationcode_blue_hill
    stationlat = "{:<4}".format(str(int(station_metadata[0]*10)))
    stationlon = "{:<4}".format(str(int(station_metadata[1]*10)))
    stationelevation = "{:<3}".format(str(station_metadata[2]))
    stationname = "{:<20}".format(station_metadata[3][:20])
    stationcountry = "{:<13}".format(station_metadata[4][:13])
    stationfirstlast = str(station_metadata[5]) + str(station_metadata[6])
    stationsourcefirst = "{:<8}".format(str(station_metadata[7]) + str(station_metadata[8]))
    stationgridcell = "{:<3}".format('NAN')
    station_header = ' ' + stationcode[0:] + ' ' + stationlat + ' ' + stationlon + ' ' + stationelevation + ' ' + stationname + ' ' + stationcountry + ' ' + stationfirstlast + '  ' + stationsourcefirst + '   ' + stationgridcell 
    with open(stationfile,'w') as f:
        f.write(station_header+'\n')
        for i in range(len(station_data)):  
            year = str(int(station_data.iloc[i,:][0]))
            rowstr = year
            for j in range(1,13):
                if np.isnan(station_data.iloc[i,:][j]):
                    monthstr = str(-99.9)
                else:
                    monthstr = str(np.round(station_data.iloc[i,:][j],1))
                rowstr += f"{monthstr:>5}"          
            f.write(rowstr+'\n')

    # GloSAT: monthly adjusted using T2828 (1961-2020) + Farrar Back-Extension: 1790-2020 (degF)
                    
    yearlist = df_blue_hill_1790_2020_tobs_adjusted.index.year.unique()
    df_blue_hill_adjusted = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12'])
    df_blue_hill_adjusted['year'] = yearlist
    for j in range(1,13):
        df_blue_hill_adjusted[str(j)] = df_blue_hill_1790_2020_tobs_adjusted[df_blue_hill_1790_2020_tobs_adjusted.index.month == j]['blue_hill'].values        
    stationfile = 'bho-glosat-adjusted-1790-2020.csv'
    station_data = df_blue_hill_adjusted.iloc[:,range(0,13)].reset_index(drop=True)
    station_metadata = da_blue_hill.iloc[0,range(14,23)]
    stationcode = stationcode_blue_hill
    stationlat = "{:<4}".format(str(int(station_metadata[0]*10)))
    stationlon = "{:<4}".format(str(int(station_metadata[1]*10)))
    stationelevation = "{:<3}".format(str(station_metadata[2]))
    stationname = "{:<20}".format(station_metadata[3][:20])
    stationcountry = "{:<13}".format(station_metadata[4][:13])
    stationfirstlast = str(station_metadata[5]) + str(station_metadata[6])
    stationsourcefirst = "{:<8}".format(str(station_metadata[7]) + str(station_metadata[8]))
    stationgridcell = "{:<3}".format('NAN')
    station_header = ' ' + stationcode[0:] + ' ' + stationlat + ' ' + stationlon + ' ' + stationelevation + ' ' + stationname + ' ' + stationcountry + ' ' + stationfirstlast + '  ' + stationsourcefirst + '   ' + stationgridcell 
    with open(stationfile,'w') as f:
        f.write(station_header+'\n')
        for i in range(len(station_data)):  
            year = str(int(station_data.iloc[i,:][0]))
            rowstr = year
            for j in range(1,13):
                if np.isnan(station_data.iloc[i,:][j]):
                    monthstr = str(-99.9)
                else:
                    monthstr = str(np.round(station_data.iloc[i,:][j],1))
                rowstr += f"{monthstr:>5}"          
            f.write(rowstr+'\n')

#------------------------------------------------------------------------------
print('** END')

#------------------------------------------------------------------------------
# BITS & BOBS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#if flag_stophere == True:
#    break
#else:
#    continue        
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------

if plot_temp == True:

#    print('')
     
    print('plotting Farrar vs Holyoke T919 and Wigglesworh T919 ...')

    #------------------------------------------------------------------------------
    # OLS: linear regression: Farrar (Tmean) vs Holyoke T919 and Wigglesworth T919
    #------------------------------------------------------------------------------

    df_regression = df_holyoke_919.copy()
    df_regression.rename(columns={"T(919)": "Holyoke"},inplace=True)
    df_regression['Wigglesworth'] = df_wigglesworth_919['T(919)']
    df_regression['Farrar'] = df_farrar['Tmean']    
    X = df_regression.Farrar
    Y_H = df_regression.Holyoke
    Y_W = df_regression.Holyoke

    minval = np.nanmin([np.nanmin(X), np.nanmin(Y_H), np.nanmin(Y_W)])
    maxval = np.nanmin([np.nanmax(X), np.nanmax(Y_H), np.nanmax(Y_W)])

    mask_H = np.isfinite(X) & np.isfinite(Y_H)
    mask_W = np.isfinite(X) & np.isfinite(Y_W)
    corrcoef_H = scipy.stats.pearsonr(X[mask_H], Y_H[mask_H])[0]
    corrcoef_W = scipy.stats.pearsonr(X[mask_W], Y_W[mask_W])[0]
    OLS_X_H, OLS_Y_H, OLS_slope_H, OLS_intercept_H, OLS_mse_H, OLS_r2_H = linear_regression_ols(X[mask_H], Y_H[mask_H])
    OLS_X_W, OLS_Y_W, OLS_slope_W, OLS_intercept_W, OLS_mse_W, OLS_r2_W = linear_regression_ols(X[mask_W], Y_W[mask_W])

    figstr = 'salem(MA)-holyoke-T919-wigglesworth-T919-cambridge(MA)-farrar-regression.png'
    titlestr = 'Linear regression (early observations 1790-1829): Farrar monthly $T_g$ vs Holyoke monthly $T_{919}$'
                                             
    fig,ax = plt.subplots(figsize=(15,10))    
#   sns.jointplot(x=X[mask_H], y=Y_W[mask_H], kind='kde', color='teal', marker='+', fill=True)  # kind{ “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }     
    plt.scatter(X[mask_H], Y_W[mask_H], marker='o', s=50, color='black', alpha=0.2) 
    plt.plot(OLS_X_H, OLS_Y_H, color='red', ls='-', lw=2, label=r'OLS $\rho$='+str(np.round(corrcoef_H,3))  + ' ' + r'$\alpha$=' + str(np.round(OLS_slope_H[0],3)))
    ax.plot([minval,maxval], [minval,maxval], color='black', ls='--', zorder=10)    
    ax.set_xlim(minval, maxval)
    ax.set_ylim(minval, maxval)
    ax.set_aspect('equal') 
    ax.xaxis.grid(True, which='minor')      
    ax.yaxis.grid(True, which='minor')  
    ax.xaxis.grid(True, which='major')      
    ax.yaxis.grid(True, which='major')  
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    leg = plt.legend(loc='lower right', ncol=1, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    for l in leg.get_lines(): 
        l.set_alpha(1)
        l.set_marker('.')
    plt.xlabel("Farrar monthly $T_g$, $\mathrm{\degree}$" + temperature_unit, fontsize=fontsize)
    plt.ylabel("Holyoke monthly $T_{919}$, $\mathrm{\degree}$" + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')
    
    figstr = 'salem(MA)-holyoke-T919-wigglesworth-T919-cambridge(MA)-farrar-timeseries.png'
    titlestr = 'Early observations 1786-1829: Farrar monthly $T_g$, Holyoke monthly $T_{919}$ and Wigglesworth monthly $T_{919}$'

    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(df_regression['Farrar'], label='Farrar $T_g$')
    plt.plot(df_regression['Holyoke'], label='Holyoke $T_{919}$')
    plt.plot(df_regression['Wigglesworth'], label='Wigglesworth $T_{919}$')
    plt.xlabel('Year', fontsize=fontsize)
    plt.ylabel(r'Air temperature, $^{\circ}$' + temperature_unit, fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)    
    plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
    fig.tight_layout()
    plt.savefig(figstr, dpi=300)
    plt.close('all')