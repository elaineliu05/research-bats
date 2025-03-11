import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy.stats import norm

file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/matched_data_from_BATS.xlsx'
df = pd.read_excel(file_path)
df = df.apply(pd.to_numeric, errors='coerce').astype('float64')
df['yymmdd'] = pd.to_datetime(df['yymmdd_in'], format='%Y%m%d') 
df['day_of_year'] = df['yymmdd'].dt.dayofyear
df['PP'] = df['pp'] #* 12 #converting from mmolC to mgC
df['TON'] = df['TN']
df['TOP'] = df['TDP']
df['BAC'] = df['Bact']
print("Number of rows in original dataset:", len(df))

df = df[["yymmdd", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]]
df.to_csv('matched_data_from_BATS_trimmed.csv', index=False) 

arr_names = ["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]
arr_units = ["", " (m)", " (mg/m3)", " (C)", " (umol/kg)", " (umol/kg)", " (umol/kg)", " (ug/kg)", " (ug/kg)", " (umol/kg)"," (umol/kg)"," (umol/kg)"," (umol/kg)", " (nmol/kg)", " (cells*10^8/kg)", " (mgC/m³/day)"]
names_units = [arr_names_pp + arr_units for arr_names_pp, arr_units in zip(arr_names, arr_units)]

print(f"Number of rows: {len(df)}")
bac_count =(df['BAC'].count())
print("Number of non-missing BAC values:", bac_count)
top_count =(df['TOP'].count())
print("Number of non-missing TOP values:", top_count)

# Remove outliers using Chauvenet's criterion
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

fig, axs = plt.subplots(4, 4, figsize=(12, 7))
axs = axs.ravel()
print(df.head())
for i in range(len(arr_names)):
    x_min = datetime.datetime(1988, 1, 1)
    x_max = datetime.datetime(2023, 12, 31)
    if (arr_names[i] == 'Chl'):
        chl_df = df[df[arr_names[i]] > -100]
        chl_df = chauvenets_criterion(df, arr_names[i])
        scatter = axs[i].scatter(chl_df['yymmdd'], chl_df[arr_names[i]], c=chl_df['Depth'], cmap = 'viridis', s=3, linewidths=0.1)
    else:
        filtered_df = chauvenets_criterion(df, arr_names[i])
        scatter = axs[i].scatter(filtered_df['yymmdd'], filtered_df[arr_names[i]], c=filtered_df['Depth'], cmap = 'viridis', s=3, linewidths=0.1)
    print(f"num rows after removing outliers for {arr_names[i]}: {len(filtered_df)}")
    axs[i].set_xlabel("Date"), axs[i].set_ylabel(names_units[i])
    axs[i].set_xlim(x_min, x_max)
    axs[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
    axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[i].set_title(f"{arr_names[i]}")
plt.tight_layout()
plt.show()
