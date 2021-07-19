#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:34:19 2021

@author: cqz20mbu
"""

import numpy as np
import pandas as pd

station_name = 'Berlin_Dahlem'

#------------------------------------------------------------------------------    
# LOAD: GHCNM-v4 monthly adjusted and unadjusted data
#------------------------------------------------------------------------------
        
print('loading GHCNM-v4 temperatures ...')
    
# LOAD: GHCNMv4 unadjusted monthly data --> df_u

nheader = 0
f = open('DATA/ghcnm.tavg.v4.0.1.20210618.qcu.dat')
lines = f.readlines()
dates = []
ids = []
vals = []
for i in range(nheader,len(lines)):    
    date = lines[i][11:15]
    stationcode = lines[i][0:11]
    val = 12*[None]
    for j in range(len(val)):
        val[j] = lines[i][19+(j*8):19+(j*8)+5]
    dates.append(date)
    ids.append(stationcode)
    vals.append(val) 
f.close()    
dates = np.array(dates).astype('int')
ids = np.array(ids)
vals = np.array(vals)
    
df_u = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12','stationcode'])
df_u['year'] = dates
for j in range(1,13):   
    df_u[df_u.columns[j]] = [ float(vals[i][j-1])/100.0 for i in range(len(df_u)) ]
df_u = df_u.replace(-99.99,np.nan)
df_u['stationcode'] = ids

# LOAD: GHCNMv4 adjusted monthly data --> df_a

nheader = 0
f = open('DATA/ghcnm.tavg.v4.0.1.20210618.qcf.dat')
lines = f.readlines()
dates = []
ids = []
vals = []
for i in range(nheader,len(lines)):    
    date = lines[i][11:15]
    stationcode = lines[i][0:11]
    val = 12*[None]
    for j in range(len(val)):
        val[j] = lines[i][19+(j*8):19+(j*8)+5]
    dates.append(date)
    ids.append(stationcode)
    vals.append(val) 
f.close()    
dates = np.array(dates).astype('int')
ids = np.array(ids)
vals = np.array(vals)
    
df_a = pd.DataFrame(columns=['year','1','2','3','4','5','6','7','8','9','10','11','12','stationcode'])
df_a['year'] = dates
for j in range(1,13):   
    df_a[df_a.columns[j]] = [ float(vals[i][j-1])/100.0 for i in range(len(df_a)) ]
df_a = df_a.replace(-99.99,np.nan)
df_a['stationcode'] = ids

#------------------------------------------------------------------------------    
# LOAD: GHCNM-v4 monthly adjusted and unadjusted inventories
#------------------------------------------------------------------------------

# LOAD: inventory for unadjusted stations --> df_u_inv

nheader = 0
f = open('DATA/ghcnm.tavg.v4.0.1.20210618.qcu.inv')
lines = f.readlines()
vals = []
for i in range(nheader,len(lines)):    
    val = 5*[None]
    for j in range(len(val)):
        val[j] = lines[i].split()[j]
    vals.append(val) 
f.close()    
vals = np.array(vals)    
df_u_inv = pd.DataFrame(columns=['stationcode','stationlat','stationlon','stationelevation','stationname'])
df_u_inv['stationcode'] = vals[:,0]
df_u_inv['stationlat'] = vals[:,1]
df_u_inv['stationlon'] = vals[:,2]
df_u_inv['stationelevation'] = vals[:,3]
df_u_inv['stationname'] = vals[:,4]

# LOAD: inventory for adjusted stations --> df_a_inv

nheader = 0
f = open('DATA/ghcnm.tavg.v4.0.1.20210618.qcf.inv')
lines = f.readlines()
vals = []
for i in range(nheader,len(lines)):    
    val = 5*[None]
    for j in range(len(val)):
        val[j] = lines[i].split()[j]
    vals.append(val) 
f.close()    
vals = np.array(vals)
df_a_inv = pd.DataFrame(columns=['stationcode','stationlat','stationlon','stationelevation','stationname'])
df_a_inv['stationcode'] = vals[:,0]
df_a_inv['stationlat'] = vals[:,1]
df_a_inv['stationlon'] = vals[:,2]
df_a_inv['stationelevation'] = vals[:,3]
df_a_inv['stationname'] = vals[:,4]

#------------------------------------------------------------------------------    
# EXTRACT: selected station e.g. 'Berlin_Dahlem'
#------------------------------------------------------------------------------

station_code = df_u_inv[df_u_inv['stationname'].str.contains( station_name, case = False )].reset_index(drop=True)['stationcode'][0]

da = df_u[df_u['stationcode'] == station_code]   
ts = np.array(da.groupby('year').mean().iloc[:,0:12]).ravel()
t = pd.date_range(start=str(da.year.iloc[0]), periods=len(ts), freq='MS')
df_u_station = pd.DataFrame({'t2m':ts}, index=t)     
df_u_station.index.name = 'datetime'
df_u_station.to_csv( station_name + '.csv')
   
#------------------------------------------------------------------------------    
print('** END') 

   