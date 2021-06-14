#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: load-new-england-datasets.py
#------------------------------------------------------------------------------
# Version 0.2
# 14 June, 2021
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
load_historical_observations = True
load_bho_observations = True
load_neighbouring_stations = True

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


#==============================================================================
            
if load_historical_observations == True:
    
    # LOAD: Holyoke
    
    df_holyoke = pd.read_csv('OUT/df_holyoke.csv', index_col=0)
    df_holyoke.index = pd.to_datetime(df_holyoke.index)

    # LOAD: Wigglesworth

    df_wigglesworth = pd.read_csv('OUT/df_wigglesworth.csv', index_col=0)
    df_wigglesworth.index = pd.to_datetime(df_wigglesworth.index)

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
    ts = fahrenheit_to_centigrade(ts)   

    # HANDLE: leap days
 
    t = xr.cftime_range(start=str(da.index[0])+'-01-01', periods=len(ts), freq='D', calendar="all_leap")   
    df_bho_tmin = pd.DataFrame({'Tmin':ts}, index=t)
    df_bho_tmin.index.name = 'datetime'

    # CALCULATE: Tg=(Tn+Tx)/2 and resample to monthly (and trim to TS end)
    
    df_bho_daily = pd.DataFrame({'Tmin':df_bho_tmin['Tmin'],'Tmax':df_bho_tmax['Tmax']},index=t)
    df_bho_daily['Tg'] = (df_bho_daily['Tmin']+df_bho_daily['Tmax'])/2.      

    # RESAMPLE: using xarray

    df_bho_daily_xr = df_bho_daily.to_xarray()    
    df_bho_daily_xr_resampled = df_bho_daily_xr.Tg.resample(datetime='MS').mean().to_dataset()    
    df_bho_monthly = df_bho_tg.copy() 
    df_bho_monthly['Tgm'] = df_bho_daily_xr_resampled.Tg.values
    
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
    # LOAD: neighbouring station datasets
    #------------------------------------------------------------------------------
    
    # NOAA NCEI LCD station group 1 
    #------------------------------
    #USC00197124	SALEM COAST GUARD AIR STATION, MA US
    #USC00197122	SALEM B, MA US
    #USW00014739	BOSTON, MA US
    #USC00195306	NEW SALEM, MA US
    #USW00094701	BOSTON CITY WEATHER SERVICE OFFICE, MA US (short --> exclude)
    
    # NOAA NCEI LCD station group 2
    #------------------------------
    #USC00190736	BLUE HILL COOP, MA US
    #USC00190538	BEDFORD, MA US
    #USC00194105	LAWRENCE, MA US (short --> exclude)
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
        if (i==3) | (i==5): # exclude short series: USC00194105 (Boston WSO) and USC00197122 (Lawrence)
            continue        
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
    df_neighbouring_stations.to_csv('df_neighbouring_stations.csv')

Nstations = df_neighbouring_stations.groupby('STATION').count().shape[0]
uniquestations = df_neighbouring_stations.groupby('STATION').mean().index
uniquenames = df_neighbouring_stations.groupby('NAME').mean().index

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
# SEABORN:
#
# sns.jointplot(x=x y=y, kind='kde', color='b', marker='+', fill=True)  # kind{ “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }            

# jointgrid = sns.JointGrid(x=x, y=y, data=df)
# jointgrid.plot_joint(sns.scatterplot)
# jointgrid.plot_marginals(sns.kdeplot)

# pairgrid = sns.PairGrid(data=iris)
# pairgrid = pairgrid.map_upper(sns.scatterplot)
# pairgrid = pairgrid.map_diag(plt.hist)
# pairgrid = pairgrid.map_lower(sns.kdeplot)

# sns.scatterplot(x=dx, y=y, s=5, color="b")
# sns.histplot(x=x, y=y, bins=100, pthresh=0.01, cmap="Blues")
# sns.kdeplot(x=x, y=y, levels=10, color="k", linewidths=1)
# sns.kdeplot(x, color='b', shade=True, alpha=0.2, legend=True, **kwargs, label='')
# sns.boxplot(data = df_bho_monthly, orient = "v")
# sns.violinplot(data = df_bho_monthly, orient = "v")

#------------------------------------------------------------------------------

# PLOT: Holyoke observations (daily)

print('plotting Holyoke (+ Farrar) osbervations ...')
    
figstr = 'salem(MA)-holyoke-cambridge(MA)-farrar.png'
titlestr = 'Salem, MA: Holyoke (sub-daily) and Cambridge, MA: Farrar (monthly mean) observations'

fig, axs = plt.subplots(figsize=(15,10))
sns.lineplot(x=df_holyoke.index, y='T(08:00)', data=df_holyoke, ax=axs, marker='.', color='b', alpha=1.0, label='Salem, MA: T(08:00)')
sns.lineplot(x=df_holyoke.index, y='T(13:00)', data=df_holyoke, ax=axs, marker='.', color='r', alpha=1.0, label='Salem, MA: T(13:00)')
sns.lineplot(x=df_holyoke.index, y='T(22:00)', data=df_holyoke, ax=axs, marker='.', color='purple', alpha=1.0, label='Salem, MA: T(22:00)')
sns.lineplot(x=df_holyoke.index, y='T(sunset)', data=df_holyoke, ax=axs, marker='.', color='orange', alpha=1.0, label='Salem, MA: T(sunset)')
sns.lineplot(x=df_farrar.index, y=df_farrar['Tmean'], ax=axs, marker='.', color='navy', ls='-', lw=3, label='Cambridge, MA: T(mean)')
axs.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs.set_xlim(pd.Timestamp('1785-01-01'),pd.Timestamp('1835-01-01'))
axs.set_ylim(-30,40)
axs.set_xlabel('Year', fontsize=fontsize)
axs.set_ylabel(r'2m Temperature, $^{\circ}$C', fontsize=fontsize)
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
sns.lineplot(x=df_wigglesworth.index, y='T(08:00)', data=df_wigglesworth, ax=axs, marker='.', color='b', alpha=1.0, label='Salem, MA: T(08:00)')
sns.lineplot(x=df_wigglesworth.index, y='T(13:00)', data=df_wigglesworth, ax=axs, marker='.', color='r', alpha=1.0, label='Salem, MA: T(13:00)')
sns.lineplot(x=df_wigglesworth.index, y='T(21:00)', data=df_wigglesworth, ax=axs, marker='.', color='purple', alpha=1.0, label='Salem, MA: T(21:00)')
sns.lineplot(x=df_farrar.index, y=df_farrar['Tmean'], ax=axs, marker='.', color='navy', ls='-', lw=3, label='Cambridge, MA: T(mean)')
axs.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs.set_xlim(pd.Timestamp('1785-01-01'),pd.Timestamp('1835-01-01'))
axs.set_ylim(-30,40)
axs.set_xlabel('Year', fontsize=fontsize)
axs.set_ylabel(r'2m Temperature, $^{\circ}$C', fontsize=fontsize)
axs.set_title(titlestr, fontsize=fontsize)
axs.tick_params(labelsize=fontsize)    
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')
    
# PLOT: BHO: T2828 (monthly) versus Tg (monthly)

print('plotting BHO: T2828 (monthly) versus Tg (monthly)...')
    
figstr = 'bho-t2828(monthly)-vs-tg(monthly).png'
titlestr = 'Blue Hill Observatory (BHO): $T_{2828}$ (monthly) versus $T_g$ (monthly)'

fig, axs = plt.subplots(2,1, figsize=(15,10))
sns.lineplot(x=df_bho_2828.index, y=df_bho_2828['T2828'], ax=axs[0], marker='o', color='r', alpha=1.0, label='$T_{2828}$')
sns.lineplot(x=df_bho_2828.index, y=df_bho_tg['Tg'], ax=axs[0], marker='.', color='b', alpha=1.0, label='$T_{g}$')
sns.lineplot(x=df_bho_2828.index, y=df_bho_2828['T2828'] - df_bho_tg['Tg'], ax=axs[1], color='teal')
axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs[0].set_xlabel('', fontsize=fontsize)
axs[0].set_ylabel(r'2m Temperature, $^{\circ}$C', fontsize=fontsize)
axs[0].set_title(titlestr, fontsize=fontsize)
axs[0].tick_params(labelsize=fontsize)    
axs[0].set_ylim(-20,30)
axs[1].sharex(axs[0])
axs[1].tick_params(labelsize=fontsize)    
axs[1].set_xlabel('Year', fontsize=fontsize)
axs[1].set_ylabel(r'$T_{2828}$-$T_{g}$, $^{\circ}$C', fontsize=fontsize)
axs[1].set_ylim(-3,3)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')

figstr = 'bho-t2828(monthly)-vs-tg(monthly)-kde.png'
titlestr = 'Blue Hill Observatory (BHO): $T_{2828}$ (monthly) versus $T_g$ (monthly) distributions'

fig, ax = plt.subplots(figsize=(15,10))
kwargs = {'levels': np.arange(0, 0.15, 0.01)}
sns.kdeplot(df_bho_2828['T2828'], color='r', shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{2828}$')
sns.kdeplot(df_bho_tg['Tg'], color='b', shade=True, alpha=0.2, legend=True, **kwargs, label=r'$T_{g}$')
ax.set_xlim(-20,30)
ax.set_ylim(0,0.05)
plt.xlabel(r'2m Temperature, $^{\circ}$C', fontsize=fontsize)
plt.ylabel('KDE', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: BHO: Tg (from daily) versus Tg (monthly)

print('plotting BHO: Tg (from daily) versus Tg (monthly) ...')
    
figstr = 'bho-tg(from-daily)-vs-tg(monthly).png'
titlestr = 'Blue Hill Observatory (BHO): $T_{g}$ (from daily) versus $T_g$ monthly'

fig, axs = plt.subplots(2,1, figsize=(15,10))
sns.lineplot(x=df_bho_tg.index, y='Tgm', data=df_bho_monthly, ax=axs[0], marker='o', color='r', alpha=1.0, label='$T_{g}$ (from daily)')
sns.lineplot(x=df_bho_tg.index, y='Tg', data=df_bho_monthly, ax=axs[0], marker='.', color='b', alpha=1.0, label='$T_{g}$ monthly')
sns.lineplot(x=df_bho_tg.index, y=df_bho_monthly['Tgm']-df_bho_monthly['Tg'], data=df_bho_monthly, ax=axs[1], color='teal')
axs[0].legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=fontsize)    
axs[0].set_xlabel('', fontsize=fontsize)
axs[0].set_ylabel(r'2m Temperature, $^{\circ}$C', fontsize=fontsize)
axs[0].set_title(titlestr, fontsize=fontsize)
axs[0].tick_params(labelsize=fontsize)   
axs[0].set_ylim(-20,30)
axs[1].sharex(axs[0]) 
axs[1].tick_params(labelsize=fontsize)    
axs[1].set_xlabel('Year', fontsize=fontsize)
axs[1].set_ylabel(r'$T_{g}$ (from daily) - $T_{g}$ (monthly), $^{\circ}$C', fontsize=fontsize)
axs[1].set_ylim(-3,3)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: Distributions of Tn, Tg and Tx (daily) for neighbouring stations

print('plotting neighbouring station distributions of Tn,Tg and Tx ...')
    
figstr = 'neighbouring-stations-distributions.png'
titlestr = 'GHCN-D stations in the Boston, MA area: KDE distributions of monthly $T_{n}$, $T_{g}$ and $T_{x}$'

# DEDUCE: row and column index for loop over subplots

nrows = 3
ncols = 3
nr = int(np.ceil(Nstations/ncols))
r = 0

#fig,ax = plt.subplots(figsize=(15,10))
#g = sns.FacetGrid(df_neighbouring_stations, col='STATION', col_wrap=4)
#g.map(sns.kdeplot, 'TMIN', color='b', shade=True, alpha=0.5, legend=True, label=r'$T_{n}$')
#g.map(sns.kdeplot, 'TAVG', color='purple', shade=True, alpha=0.5, legend=True, label=r'$T_{g}$')
#g.map(sns.kdeplot, 'TMAX', color='r', shade=True, alpha=0.5, legend=True, label=r'$T_{x}$')
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

fig,axs = plt.subplots(nrows, ncols, figsize=(15,10))
for i in range(nrows*ncols):    
    if i > (Nstations-1):
        axs[-1,-1].axis('off')
        continue
    ymin = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMIN']
    ymax = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMAX']
    yavg = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TAVG']    
    t = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]].index
    c = i%ncols
    if (i > 0) & (c == 0):
        r += 1     
    print(i,r,c)
    g = sns.kdeplot(ymin, ax=axs[r,c], color='b', shade=True, alpha=0.5, legend=True, label=r'$T_{n}$')
    sns.kdeplot(yavg, ax=axs[r,c], color='purple', shade=True, alpha=0.5, legend=True, label=r'$T_{g}$')
    sns.kdeplot(ymax, ax=axs[r,c], color='r', shade=True, alpha=0.5, legend=True, label=r'$T_{x}$')   
    g.axes.set_xlim(-30,40)
    g.axes.set_ylim(0,0.05)
    if (r+1) == nrows:
        g.axes.set_xlabel(r'2m Temperature, $^{\circ}$C', fontsize=fontsize)
    else:
        g.axes.set_xlabel('', fontsize=fontsize)
        g.axes.set_xticklabels([])  
    if c == 0:
        g.axes.set_ylabel('KDE', fontsize=fontsize)
    else:
        g.axes.set_ylabel('', fontsize=fontsize)
        g.axes.set_yticklabels([])
    g.axes.set_title('STATION='+uniquestations[i], fontsize=fontsize)
    g.axes.tick_params(labelsize=fontsize)    
    g.axes.legend(loc='upper right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)    
    
fig.subplots_adjust(top=0.9)
fig.suptitle(titlestr, fontsize=fontsize)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: Timeseries of Tn, Tg and Tx (monthly) for neighbouring stations

print('plotting neighbouring station monthly-averaged timeseries of Tn,Tg and Tx ...')
    
figstr = 'neighbouring-stations-timeseries-1m-average.png'
titlestr = 'GHCN-D stations in the Boston, MA area: timeseries of monthly-averaged $T_{n}$, $T_{g}$ and $T_{x}$'

# DEDUCE: row and column index for loop over subplots

nrows = 3
ncols = 3
nr = int(np.ceil(Nstations/ncols))
r = 0

fig,axs = plt.subplots(nrows, ncols, figsize=(15,10))
for i in range(nrows*ncols):    
    if i > (Nstations-1):
        axs[-1,-1].axis('off')
        continue
    ymin = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMIN']
    ymax = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMAX']
    yavg = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TAVG']    
    t = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]].index
    c = i%ncols
    if (i > 0) & (c == 0):
        r += 1     
    print(i,r,c)
    df_monthly = pd.DataFrame({'TMIN':ymin, 'TAVG':yavg, 'TMAX':ymax}, index=t)
    df_monthly_xr = df_monthly.to_xarray()    
    ymin_yearly = df_monthly_xr['TMIN'].resample(datetime='MS').mean().to_dataset() 
    yavg_yearly = df_monthly_xr['TAVG'].resample(datetime='MS').mean().to_dataset() 
    ymax_yearly = df_monthly_xr['TMAX'].resample(datetime='MS').mean().to_dataset() 
    t = pd.date_range(start=str(df_monthly.index[0].year), periods=len(ymin_yearly.TMIN.values), freq='MS')
    g = sns.lineplot(x=t, y=ymin_yearly.TMIN.values, ax=axs[r,c], marker='.', color='b', alpha=0.5, label='$T_{n}$')
    sns.lineplot(x=t, y=yavg_yearly.TAVG.values, ax=axs[r,c], marker='.', color='purple', alpha=0.5, label='$T_{g}$')
    sns.lineplot(x=t, y=ymax_yearly.TMAX.values, ax=axs[r,c], marker='.', color='r', alpha=0.5, label='$T_{x}$')
    g.axes.set_ylim(-30,40)
    g.axes.set_xlim(pd.Timestamp('1880-01-01'),pd.Timestamp('2020-01-01'))
    if (r+1) == nrows:
        g.axes.set_xlabel('Year', fontsize=fontsize)
    else:
        g.axes.set_xticklabels([])  
    if c == 0:
        g.axes.set_ylabel(r'2m Temperature, $^{\circ}$C', fontsize=fontsize)
    else:
        g.axes.set_yticklabels([])
    g.axes.set_title('STATION='+uniquestations[i], fontsize=fontsize)
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
titlestr = 'GHCN-D stations in the Boston, MA area: timeseries of 24m-averaged $T_{n}$, $T_{g}$ and $T_{x}$'

# DEDUCE: row and column index for loop over subplots

nrows = 3
ncols = 3
nr = int(np.ceil(Nstations/ncols))
r = 0

fig,axs = plt.subplots(nrows, ncols, figsize=(15,10))
for i in range(nrows*ncols):    
    if i > (Nstations-1):
        axs[-1,-1].axis('off')
        continue
    ymin = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMIN']
    ymax = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TMAX']
    yavg = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]]['TAVG']    
    t = df_neighbouring_stations[df_neighbouring_stations['STATION']==df_neighbouring_stations['STATION'].unique()[i]].index
    c = i%ncols
    if (i > 0) & (c == 0):
        r += 1     
    print(i,r,c)
    df_monthly = pd.DataFrame({'TMIN':ymin, 'TAVG':yavg, 'TMAX':ymax}, index=t)
    df_monthly_xr = df_monthly.to_xarray()    
    ymin_yearly = df_monthly_xr['TMIN'].resample(datetime='2AS').mean().to_dataset() 
    yavg_yearly = df_monthly_xr['TAVG'].resample(datetime='2AS').mean().to_dataset() 
    ymax_yearly = df_monthly_xr['TMAX'].resample(datetime='2AS').mean().to_dataset() 
    t = pd.date_range(start=str(df_monthly.index[0].year), periods=len(ymin_yearly.TMIN.values), freq='2AS')
    g = sns.lineplot(x=t, y=ymin_yearly.TMIN.values, ax=axs[r,c], marker='.', color='b', alpha=0.5, label='$T_{n}$')
    sns.lineplot(x=t, y=yavg_yearly.TAVG.values, ax=axs[r,c], marker='.', color='purple', alpha=0.5, label='$T_{g}$')
    sns.lineplot(x=t, y=ymax_yearly.TMAX.values, ax=axs[r,c], marker='.', color='r', alpha=0.5, label='$T_{x}$')
    g.axes.set_ylim(-30,40)
    g.axes.set_xlim(pd.Timestamp('1880-01-01'),pd.Timestamp('2020-01-01'))
    if (r+1) == nrows:
        g.axes.set_xlabel('Year', fontsize=fontsize)
    else:
        g.axes.set_xticklabels([])  
    if c == 0:
        g.axes.set_ylabel(r'2m Temperature, $^{\circ}$C', fontsize=fontsize)
    else:
        g.axes.set_yticklabels([])
    g.axes.set_title('STATION='+uniquestations[i], fontsize=fontsize)
    g.axes.tick_params(labelsize=fontsize)    
    g.axes.legend(loc='lower right', ncol=1, markerscale=1, facecolor='lightgrey', framealpha=0.5, fontsize=12)    

fig.subplots_adjust(top=0.9)
fig.suptitle(titlestr, fontsize=fontsize)
fig.tight_layout()
plt.savefig(figstr, dpi=300)
plt.close('all')

# PLOT: Tg (hourly) + Salem (daily)

print('plotting neighbouring stations: Tmean ...')
    
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

