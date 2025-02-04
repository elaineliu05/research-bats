import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/matched_data_from_BATS.xlsx'
df = pd.read_excel(file_path)
df = df.apply(pd.to_numeric, errors='coerce').astype('float64')
df['yymmdd'] = pd.to_datetime(df['yymmdd_in'], format='%Y%m%d') 
df['day_of_year'] = df['yymmdd'].dt.dayofyear
df['PP'] = df['pp'] * 12 #converting from mmolC to mgC
df['TON'] = df['TN']
df['TOP'] = df['TDP']
df['BAC'] = df['Bact']
print("Number of rows in original dataset:", len(df))

df = df[["yymmdd", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "TOP", "BAC", "PP"]]
#df = df.dropna() #drop NaNs
#print("Number of rows after dropping NaNs:", len(df))
df = df.drop_duplicates(subset=['yymmdd', 'Depth'], keep='first') #drop duplicates
print("Number of rows after dropping duplicates:", len(df))

arr_names = ["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "TOP", "BAC", "PP"]
arr_units = ["", " (m)", " (mg/m3)", " (C)", " (umol/kg)", " (umol/kg)", " (umol/kg)", " (ug/kg)", " (ug/kg)", " (umol/kg)", " (umol/kg)", " (cells*10^8/kg)", " (mgC/m³/day)"]
names_units = [arr_names_pp + arr_units for arr_names_pp, arr_units in zip(arr_names, arr_units)]

# Remove outliers using Chauvenet's criterion
df_chl = df[df["Chl"] > -100]#drop NaNs
def chauvenets_criterion(df, col_name):
    data = df[col_name]
    mean = np.mean(data)
    std = np.std(data)
    deviations = np.abs(data - mean)/std
    n = len(data)
    probabilities = 1 - norm.cdf(deviations)
    criterion = 1.0/(2*n)
    non_outliers = probabilities >= criterion
    return df[non_outliers]
for i in range(2, len(arr_names)): #starts at 2 to avoid day of year and depth
     df = chauvenets_criterion(df, arr_names[i])
print("Number of rows after removing outliers:", len(df))
for i in range(2, len(arr_names)): #starts at 2 to avoid day of year and depth
     df_chl = chauvenets_criterion(df_chl, arr_names[i])

fig, axs = plt.subplots(3, 4, figsize=(12, 7))
axs = axs.ravel()
for i in range(len(arr_names)):
    x_min = datetime.datetime(1988, 1, 1)
    x_max = datetime.datetime(2016, 12, 31)
    if (arr_names[i] == 'Chl'):
        scatter = axs[i].scatter(df_chl['yymmdd'], df_chl[arr_names[i]], c=df_chl['Depth'], cmap = 'viridis', s=3, linewidths=0.1)
        axs[i].set_xlabel("Date"), axs[i].set_ylabel(names_units[i])
        axs[i].set_xlim(x_min, x_max)
        axs[i].tick_params(axis = 'x', rotation=45)
        axs[i].set_title(f"{arr_names[i]}")
    else: 
        scatter = axs[i].scatter(df['yymmdd'], df[arr_names[i]], c=df['Depth'], cmap = 'viridis', s=3, linewidths=0.1)
        axs[i].set_xlabel("Date"), axs[i].set_ylabel(names_units[i])
        axs[i].set_xlim(x_min, x_max)
        axs[i].tick_params(axis = 'x', rotation=45)
        axs[i].set_title(f"{arr_names[i]}")
plt.tight_layout()
plt.show()
