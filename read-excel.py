#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 21:06:15 2021

@author: cqz20mbu
"""

import numpy as np
import pandas as pd
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

workbook1 = xlrd.open_workbook('DATA/BlueHillObservatory_Temperature_Mean_MxMn2_Monthly_v2.0.xlsx')

worksheet = workbook1.sheet_by_index(1)
years = [ int(worksheet.cell(i, 0).value) for i in range(5,141) ]
dF1 = pd.DataFrame(index=years)
for j in range(12):
    month = [ float(worksheet.cell(i, j+1).value) for i in range(5,141) ]
    dF1[str(j+1)] = month
    
worksheet = workbook1.sheet_by_index(2)
years = [ int(worksheet.cell(i, 0).value) for i in range(5,141) ]
dC1 = pd.DataFrame(index=years)
for j in range(12):
    month = [ float(worksheet.cell(i, j+1).value) for i in range(5,141) ]
    dC1[str(j+1)] = month
    
workbook2 = xlrd.open_workbook('DATA/BlueHillObservatory_Temperature_Mean_MxMn2_Monthly_v2.0_20210626.xlsx')

worksheet = workbook2.sheet_by_index(1)
years = [ int(worksheet.cell(i, 0).value) for i in range(5,141) ]
dF2 = pd.DataFrame(index=years)
for j in range(12):
    month = [ float(worksheet.cell(i, j+1).value) for i in range(5,141) ]
    dF2[str(j+1)] = month
    
worksheet = workbook2.sheet_by_index(2)
years = [ int(worksheet.cell(i, 0).value) for i in range(5,141) ]
dC2 = pd.DataFrame(index=years)
for j in range(12):
    month = [ float(worksheet.cell(i, j+1).value) for i in range(5,141) ]
    dC2[str(j+1)] = month

Cdiff21 = dC2-dC1    
Fdiff21 = dF2-dF1    

Cmag = np.round( np.max([np.abs(np.nanmin(Cdiff21)), np.abs(np.nanmax(Cdiff21))]), 1)
Fmag = np.round( np.max([np.abs(np.nanmin(Fdiff21)), np.abs(np.nanmax(Fdiff21))]), 1)

sequential_colors = sns.color_palette('coolwarm', 10)
sns.set_palette(sequential_colors)

fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(Cdiff21, cmap='coolwarm', vmin=-Cmag, vmax=Cmag)
plt.savefig('Cdiff21.png', dpi=300)

fig,ax = plt.subplots(figsize=(15,10))
sns.heatmap(Fdiff21, cmap='coolwarm', vmin=-Fmag, vmax=Fmag)
plt.savefig('Fdiff21.png', dpi=300)

