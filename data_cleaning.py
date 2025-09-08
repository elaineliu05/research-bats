import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy.stats import norm

file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/matched_data_from_BATS.xlsx'
orig_df = pd.read_excel(file_path)
#reformatting columns
orig_df = orig_df.apply(pd.to_numeric, errors='coerce').astype('float64')
orig_df['yymmdd'] = pd.to_datetime(orig_df['yymmdd_in'], format='%Y%m%d') #convert to date time
orig_df['day_of_year'] = orig_df['yymmdd'].dt.dayofyear # day out of 365
orig_df['PP'] = orig_df['pp'] 
orig_df['TON'] = orig_df['TN']
orig_df['TOP'] = orig_df['TDP']
orig_df['BAC'] = orig_df['Bact']
print("Number of rows in original dataset:", len(orig_df))

df = orig_df[["yymmdd", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]].copy()
#Adding additional columns
df['year'] = df['yymmdd'].dt.year           # extract year
df['month'] = df['yymmdd'].dt.month         # extract month
df['day'] = df['yymmdd'].dt.day             # extract day
# df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
# df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
df.to_csv('matched_data_from_BATS_trimmed.csv', index=False) 

#naming variables and units
arr_names = ["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]
arr_units =  ["", " (m)", " (mg/m3)", " (C)", " (PSS-78)", " (umol/kg)"," (umol/kg)", " (umol/kg)", " (ug/kg)", " (ug/kg)", " (umol/kg)", " (umol/kg)", " (umol/kg)", " (nmol/kg)", " (cells*10^8/kg)", " (mgC/m³/day)"]
names_units = [arr_names_pp + arr_units for arr_names_pp, arr_units in zip(arr_names, arr_units)]

# remove outliers using Chauvenet's criterion
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

# plotting all variables over time
# fig, axs = plt.subplots(4, 4, figsize=(12, 7))
# axs = axs.ravel()
# for i in range(len(arr_names)):
#     x_min = datetime.datetime(1988, 1, 1)
#     x_max = datetime.datetime(2023, 12, 31)
#     filtered_df = chauvenets_criterion(df, arr_names[i])
#     scatter = axs[i].scatter(filtered_df['yymmdd'], filtered_df[arr_names[i]], c=filtered_df['Depth'], cmap = 'viridis', s=3, linewidths=0.1)
#     # print(f"num rows after removing outliers for {arr_names[i]}: {len(filtered_df)}")
#     axs[i].set_xlabel("Date"), axs[i].set_ylabel(names_units[i])
#     axs[i].set_xlim(x_min, x_max)
#     axs[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
#     axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#     axs[i].set_title(f"{arr_names[i]}")
# plt.tight_layout()
# plt.show()

# selecting variable sets
#set a: all variables available
df_a = df
df_a = df_a.dropna()
#set b: variables not measured prior to 1994 removed (best results)
df_b = df[["yymmdd", "year", "month", "day", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "BAC", "PP"]]
df_b = df_b.dropna() 
#set c: variables with consistent data
df_c = df[["yymmdd", "year", "month", "day", "day_of_year", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "BAC", "PP"]]
df_c = df_c.dropna()
#set d: most common measurements
df_d = df[["yymmdd", "year", "month", "day", "day_of_year", "Depth", "Temp", "O2", "NO3", "PO4", "PP"]]
df_d = df_d.dropna()

# applying chauvenet to all dfs
# start at column 6 (Depth) to avoid date columns
for i in range(6, df_a.shape[1]): df_a = chauvenets_criterion(df_a, df_a.columns[i])
print("Num rows in set A after removing outliers:", len(df_a))
for i in range(6, df_b.shape[1]): df_b = chauvenets_criterion(df_b, df_b.columns[i])
print("Num rows in set B after removing outliers:", len(df_b))
for i in range(6, df_c.shape[1]): df_c = chauvenets_criterion(df_c, df_c.columns[i])
print("Num rows in set C after removing outliers:", len(df_c))
for i in range(6, df_d.shape[1]): df_d = chauvenets_criterion(df_d, df_d.columns[i])
print("Num rows in set D after removing outliers:", len(df_d))

# exporting dataframes
df_a.to_csv('df_a.csv', index=False) 
df_b.to_csv('df_b.csv', index=False) 
df_c.to_csv('df_c.csv', index=False) 
df_d.to_csv('df_d.csv', index=False) 