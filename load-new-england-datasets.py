#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: load-new-england-datasets.py
#------------------------------------------------------------------------------
# Version 0.1
# 8 June, 2021
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
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import cmocean

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
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

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
load_stations = True
flag_stophere = False

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
    y = (5/9) * (x - 32)
    return y

def centigrade_to_fahrenheit(x):
    y = (x * (5/9)) + 32
    return y

def is_leap_and_29Feb(s):
    return (s.index.year % 4 == 0) & ((s.index.year % 100 != 0) | (s.index.year % 400 == 0)) & (s.index.month == 2) & (s.index.day == 29)

#------------------------------------------------------------------------------
#if flag_stophere == True:
#    break
#else:
#    continue        
#------------------------------------------------------------------------------
            
#------------------------------------------------------------------------------
# LOAD: Holyoke observations into dataframe
#------------------------------------------------------------------------------

if load_stations == True:
    
    df_holyoke = pd.read_csv('OUT/df_holyoke.csv', index_col=0)
    df_wigglesworth = pd.read_csv('OUT/df_wigglesworth.csv', index_col=0)
    df_neighbouring_stations = pd.read_csv('OUT/df_neighbouring_stations.csv', index_col=0)

    df_holyoke.index = pd.to_datetime(df_holyoke.index)
    df_wigglesworth.index = pd.to_datetime(df_wigglesworth.index)
    df_neighbouring_stations.index = pd.to_datetime(df_neighbouring_stations.index)

    # LOAD: Farrar (CRUTEM format)

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

    # LOAD: BHO (CRUTEM format)

#   da = pd.read_excel('DATA/BlueHillObservatory_Temperature_Mean_2828_Monthly_v2.0.xlsx', header=5, use_cols=['A':'N'], sheet_name='Mean Temperature deg C')    
    da = pd.read_table('DATA/bho-2828.dat', index_col=0) # '2828' monthly average
    ts_monthly = []    
    for i in range(len(da)):                
        monthly = da.iloc[i,0:]
        ts_monthly = ts_monthly + monthly.to_list()    
    ts_monthly = np.array(ts_monthly)   
    t_monthly = pd.date_range(start=str(da.index[0]), periods=len(ts_monthly), freq='MS')    
    df_bho_2828 = pd.DataFrame({'T2828':ts_monthly}, index=t_monthly)
    df_bho_2828.index.name = 'datetime'

    da = pd.read_table('DATA/bho-tg.dat', index_col=0) # Tg monthly average
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
    ts = fahrenheit_to_centigrade(ts)   

    # HANDLE: leap days (not working yet!)

    t = xr.cftime_range(start=str(da.index[0])+'-01-01', periods=len(ts), freq='D', calendar="all_leap")   
#   t = pd.date_range(start=str(da.index[0])+'-01-01', periods=len(ts), freq='D')    
    df_bho_tmax = pd.DataFrame({'Tmax':ts}, index=t)
    df_bho_tmax.index.name = 'datetime'
#    mask_leap = ~is_leap_and_29Feb(df_bho_tmax)
#    ts_noleap = df_bho_tmax['Tmax'][mask_leap]
#    t = pd.date_range(start=str(da.index[0])+'-01-01', end=str(da.index[-1])+'-12-31', freq='D')    
#    df_bho_tmax = pd.DataFrame({'Tmax':ts_noleap}, index=t)
#    df_bho_tmax.index.name = 'datetime'
    
    da = pd.read_table('DATA/bho-min_01.dat', index_col=0) # daily
    for i in range(2,13):
        db = pd.read_table('DATA/bho-min_'+str(i).zfill(2)+'.dat', index_col=0) # daily 
        da = pd.concat([da,db],axis=1)        
    ts = []    
    for i in range(len(da)):                
        daily = da.iloc[i,0:]
        ts = ts + daily.to_list()    
    ts = np.array(ts)   
    ts = fahrenheit_to_centigrade(ts)   

    # HANDLE: leap days (not working yet!)
 
    t = xr.cftime_range(start=str(da.index[0])+'-01-01', periods=len(ts), freq='D', calendar="all_leap")   
#   t = pd.date_range(start=str(da.index[0])+'-01-01', periods=len(ts), freq='D')    
    df_bho_tmin = pd.DataFrame({'Tmin':ts}, index=t)
    df_bho_tmin.index.name = 'datetime'
#    mask_leap = ~is_leap_and_29Feb(df_bho_tmin)
#    ts_noleap = df_bho_tmin['Tmin'][mask_leap]
#    t = pd.date_range(start=str(da.index[0])+'-01-01', end=str(da.index[-1])+'-12-31', freq='D')    
#    df_bho_tmin = pd.DataFrame({'Tmin':ts_noleap}, index=t)
#    df_bho_tmin.index.name = 'datetime'

    # CALCULATE: Tg=(Tn+Tx)/2 and resample to monthly (and trim to TS end)
    
    df_bho_daily = pd.DataFrame({'Tmin':df_bho_tmin['Tmin'],'Tmax':df_bho_tmax['Tmax']},index=t)
    df_bho_daily['Tg'] = (df_bho_daily['Tmin']+df_bho_daily['Tmax'])/2.      

    # RESAMPLE: using xarray

#   Tgm = df_bho_daily['Tg'].resample('M').agg('mean').values    
    df_bho_daily_xr = df_bho_daily.to_xarray()    
    df_bho_daily_xr_resampled = df_bho_daily_xr.Tg.resample(datetime='MS').mean().to_dataset()    

    df_bho_monthly = df_bho_tg.copy() 
    df_bho_monthly['Tgm'] = df_bho_daily_xr_resampled.Tg.values
               
else:
       
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
    # LOAD: neighbouring station datasets
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
    
    df1 = pd.read_csv('DATA/2606266.csv') # NOAA NCEI LCD station group 1
    df2 = pd.read_csv('DATA/2606305.csv') # NOAA NCEI LCD station group 2
    df3 = pd.read_csv('DATA/2606326.csv') # NOAA NCEI LCD station group 3
    df = pd.concat([df1,df2,df3])
        
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
    #    if i==9:
    #        continue        
        stationcode = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['STATION']
        stationname = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['NAME']
        stationlat = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['LATITUDE']
        stationlon = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['LONGITUDE']
        stationelevation = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['ELEVATION']
        xmin = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['TMIN']
        xmax = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['TMAX']
        xavg = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]]['TAVG']    
        t = df_reordered[df_reordered['STATION']==df_reordered['STATION'].unique()[i]].index
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
    df_neighbouring_stations.to_csv('neighbouring_stations.csv')

Nstations = df_neighbouring_stations.groupby('STATION').count().shape[0]
uniquestations = df_neighbouring_stations.groupby('STATION').mean().index
uniquenames = df_neighbouring_stations.groupby('NAME').mean().index

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

# PLOT: Holyoke observations (daily)

print('plotting Holyoke data ...')
    
figstr = 'salem-massechussets-holyoke.png'
titlestr = 'Salem, MA: Holyoke data rescue'

fig,ax = plt.subplots(figsize=(15,10))
plt.plot(df_holyoke.index, df_holyoke['T(08:00)'].rolling(7, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label='T(08:00)')
plt.plot(df_holyoke.index, df_holyoke['T(13:00)'].rolling(7, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label='T(13:00)')
plt.plot(df_holyoke.index, df_holyoke['T(22:00)'].rolling(7, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label='T(22:00)')
plt.plot(df_holyoke.index, df_holyoke['T(sunset)'].rolling(7, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label='T(sunset)')

plt.plot(df_farrar.index, df_farrar['Tmean'], ls='-', lw=3, color='navy', zorder=10, label='Cambridge, MA: Farrar T(mean)')

plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', ncol=2, fontsize=fontsize)
#plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'2m-Temperature, [$^{\circ}$C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.legend(loc='lower left', bbox_to_anchor=(0, -0.2), ncol=5, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=12)    
fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)   
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: Wigglesworth observations (daily)

print('plotting Wigglesworth data ...')
    
figstr = 'salem-massechussets-wigglesworth.png'
titlestr = 'Salem, MA: Wigglesworth data rescue'

fig,ax = plt.subplots(figsize=(15,10))
plt.plot(df_wigglesworth.index, df_wigglesworth['T(08:00)'].rolling(7, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label='T(08:00)')
plt.plot(df_wigglesworth.index, df_wigglesworth['T(13:00)'].rolling(7, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label='T(13:00)')
plt.plot(df_wigglesworth.index, df_wigglesworth['T(21:00)'].rolling(7, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label='T(21:00)')

plt.plot(df_farrar.index, df_farrar['Tmean'], ls='-', lw=3, color='navy', zorder=10, label='Cambridge, MA: Farrar T(mean)')

plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', ncol=2, fontsize=fontsize)
#plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'2m-Temperature, [$^{\circ}$C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.legend(loc='lower left', bbox_to_anchor=(0, -0.2), ncol=4, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=12)    
fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)   
plt.savefig(figstr, dpi=300)
plt.close('all')
    
# PLOT: Farrar observations (monthly)

print('plotting Farrar data ...')
    
figstr = 'cambridge-massechussets-farrar.png'
titlestr = 'Cambridge, MA: Farrar data rescue'

fig,ax = plt.subplots(figsize=(15,10))
plt.plot(df_farrar.index, df_farrar['Tmean'], '.', alpha=0.5, ls='-', lw=0.5, color='navy', label='T(mean)')
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='lower right', ncol=2, fontsize=fontsize)
#plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'2m-Temperature, [$^{\circ}$C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.legend(loc='lower left', bbox_to_anchor=(0, -0.2), ncol=4, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=12)    
fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)   
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: BHO: T2828 versus Tg (monthly)

print('plotting BHO: T(2828) versus Tg ...')
    
figstr = 'bho-2828-tg.png'
titlestr = 'Blue Hill Observatory (BHO): $T_{2828}$ versus $T_g$'

fig, ax = plt.subplots(2,1, figsize=(15,10))
ax[0].plot(df_bho_2828.index, df_bho_2828['T2828'], '.', alpha=0.5, ls='-', lw=0.5, color='red', label='$T_{2828}$')
ax[0].plot(df_bho_2828.index, df_bho_tg['Tg'], '.', alpha=0.5, ls='-', lw=0.5, color='blue', label='$T_{g}$')
ax[1].step(x=df_bho_2828.index, y=df_bho_2828['T2828'] - df_bho_tg['Tg'], ls='-', lw=0.5, color='teal')
ax[1].sharex(ax[0])
ax[0].tick_params(labelsize=16)    
ax[1].tick_params(labelsize=16)    
ax[0].legend(loc='lower right', ncol=1, markerscale=2, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
ax[0].set_ylabel(r'2m-Temperature, [$^{\circ}$C]', fontsize=fontsize)
ax[1].set_ylabel(r'$T_{2828}$-$T_{g}$, [$^{\circ}$C]', fontsize=fontsize)
ax[0].set_title(titlestr, fontsize=fontsize)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: BHO: Tg (daily) 1m-MA versus Tg (monthly)

print('plotting BHO: Tg(daily) versus Tg(monthly) ...')
    
figstr = 'bho-tg-daily-monthly.png'
titlestr = 'Blue Hill Observatory (BHO): $T_{g}$ from daily (1m-MA) versus $T_g$ monthly'

fig, ax = plt.subplots(2,1, figsize=(15,10))
ax[0].plot(df_bho_monthly.index, df_bho_monthly['Tgm'], '.', alpha=0.5, ls='-', lw=0.5, color='red', label='$T_{g}$ from daily')
ax[0].plot(df_bho_monthly.index, df_bho_monthly['Tg'], '.', alpha=0.5, ls='-', lw=0.5, color='blue', label='$T_{g}$ monthly')
ax[1].step(x=df_bho_monthly.index, y=(df_bho_monthly['Tgm']-df_bho_monthly['Tg']), ls='-', lw=0.5, color='teal')
ax[1].sharex(ax[0])
ax[0].tick_params(labelsize=16)    
ax[1].tick_params(labelsize=16)    
ax[0].legend(loc='lower right', ncol=1, markerscale=2, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
ax[0].set_ylabel(r'2m-Temperature, [$^{\circ}$C]', fontsize=fontsize)
ax[1].set_ylabel(r'$T_{g}$ (from daily) - $T_{g}$ (monthly), [$^{\circ}$C]', fontsize=fontsize)
ax[0].set_title(titlestr, fontsize=fontsize)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: Tg (hourly): 1m-MA

print('plotting neighbouring stations: Tmin ...')
    
figstr = 'neighbouring-stations-tmin.png'
titlestr = 'GHCN-D stations within 1 degree of Boston: $T_n$'

fig,ax = plt.subplots(figsize=(15,10))
for i in range(Nstations):        
#    if i==9:
#        continue    
    ymin = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMIN']
    ymax = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMAX']
    yavg = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TAVG']    
    t = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]].index
    plt.plot(t, pd.Series(ymin).rolling(24*31, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label=uniquestations[i]+': '+uniquenames[i])

plt.tick_params(labelsize=16)    
plt.legend(loc='lower right', ncol=2, fontsize=10)
#plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'2m-Temperature, [$^{\circ}$C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.legend(loc='lower left', bbox_to_anchor=(0, -0.5), ncol=2, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=12)    
fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)   
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: Tx (hourly): 1m-MA

print('plotting neighbouring stations: Tmax ...')
    
figstr = 'neighbouring-stations-tmax.png'
titlestr = 'GHCN-D stations within 1 degree of Boston: $T_x$'

fig,ax = plt.subplots(figsize=(15,10))
for i in range(Nstations):        
#    if i==9:
#        continue    
    ymin = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMIN']
    ymax = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMAX']
    yavg = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TAVG']    
    t = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]].index
    plt.plot(t, pd.Series(ymax).rolling(24*31, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label=uniquestations[i]+': '+uniquenames[i])

plt.tick_params(labelsize=16)    
#plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'2m-Temperature, [$^{\circ}$C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.legend(loc='lower left', bbox_to_anchor=(0, -0.5), ncol=2, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=12)    
fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)             
plt.savefig(figstr, dpi=300)
plt.close('all')

print('plotting neighbouring stations: Tmean ...')
    
# PLOT: Tg (hourly) + Salem (daily)

figstr = 'neighbouring-stations-tmean.png'
titlestr = 'GHCN-D stations within 1 degree of Boston: hourly $T_g$ (1m-MA) versus daily obs from Holyoke record'

fig,ax = plt.subplots(figsize=(15,10))
for i in range(Nstations):        
#    if i==9:
#        continue    
    ymin = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMIN']
    ymax = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMAX']
    yavg = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TAVG']    
    t = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]].index
    plt.plot(t, pd.Series(yavg).rolling(24*31, center=True).mean(), '.', alpha=0.5, ls='-', lw=0.5, label=uniquestations[i]+': '+uniquenames[i])
    
plt.plot(df_holyoke.index, df_holyoke['T(08:00)'], '.', alpha=0.1, ls='-', lw=0.5, label='Holyoke: daily T(08:00)')
plt.plot(df_holyoke.index, df_holyoke['T(13:00)'], '.', alpha=0.1, ls='-', lw=0.5, label='Holyoke: daily T(13:00)')
plt.plot(df_holyoke.index, df_holyoke['T(22:00)'], '.', alpha=0.1, ls='-', lw=0.5, label='Holyoke: daily T(22:00)')
plt.plot(df_holyoke.index, df_holyoke['T(sunset)'], '.', alpha=0.1, ls='-', lw=0.5, label='Holyoke: daily T(sunset)')    
plt.axhline(y=np.nanmean(df_holyoke['T(08:00)']), ls='dashed', color='blue')
plt.axhline(y=np.nanmean(df_holyoke['T(13:00)']), ls='dashed', color='orange')
plt.axhline(y=np.nanmean(df_holyoke['T(22:00)']), ls='dashed', color='green')
plt.axhline(y=np.nanmean(df_holyoke['T(sunset)']), ls='dashed', color='red')
        
plt.tick_params(labelsize=16)    
#plt.xlabel('Year', fontsize=fontsize)
plt.ylabel(r'2m-Temperature, [$^{\circ}$C]', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.legend(loc='lower left', bbox_to_anchor=(0, -0.6), ncol=2, markerscale=3, facecolor='lightgrey', framealpha=1, fontsize=12)    
fig.subplots_adjust(left=None, bottom=0.4, right=None, top=None, wspace=None, hspace=None)             
plt.savefig(figstr, dpi=300)
plt.close('all')

#------------------------------------------------------------------------------
print('** END')

