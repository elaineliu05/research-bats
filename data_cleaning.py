import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy.stats import norm

file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/data/matched_data_from_BATS.xlsx'
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
#convert all negative pp values to 0
df = orig_df.copy()
df.loc[df['PP'] < 0, 'PP'] = 0


df = df[["yymmdd", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]].copy()
#Adding additional columns
df['year'] = df['yymmdd'].dt.year           # extract year
df['month'] = df['yymmdd'].dt.month         # extract month
df['day'] = df['yymmdd'].dt.day             # extract day
df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
# calculating decimal year (takes care of leap year)
year = df['yymmdd'].dt.year
start_of_year = pd.to_datetime(year.astype(str) + '-01-01')
start_next_year = pd.to_datetime((year + 1).astype(str) + '-01-01')
year_length = (start_next_year - start_of_year).dt.days
days_into_year = (df['yymmdd'] - start_of_year).dt.days
df['dec_year'] = year + days_into_year / year_length
# adding set depths
depth_levels = np.array([1, 20, 40, 60, 80, 100, 120, 140])
def snap_depth(value):
    return depth_levels[np.abs(depth_levels - value).argmin()]
# apply to your dataframe
df['set_depths'] = df['Depth'].apply(snap_depth)
# export cleaned dataframe
df.to_csv('matched_data_from_BATS_trimmed.csv', index=False) 

#naming variables and units
arr_names = ["sin_doy", "cos_doy", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]
arr_units =  ["", "", " (m)", " (mg/m3)", " (C)", " (PSS-78)", " (umol/kg)"," (umol/kg)", " (umol/kg)", " (ug/kg)", " (ug/kg)", " (umol/kg)", " (umol/kg)", " (umol/kg)", " (nmol/kg)", " (cells*10^8/kg)", " (mgC/m³/day)"]
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
# fig, axs = plt.subplots(4, 5, figsize=(12, 7))
# axs = axs.ravel()
# for i in range(len(arr_names)):
#     x_min = datetime.datetime(1988, 1, 1)
#     x_max = datetime.datetime(2023, 12, 31)
#     filtered_df = chauvenets_criterion(df, arr_names[i])
#     scatter = axs[i].scatter(filtered_df['yymmdd'], filtered_df[arr_names[i]], c=filtered_df['Depth'], cmap = 'viridis_r', s=3, linewidths=0.1)
#     print(f"num rows after removing outliers for {arr_names[i]}: {len(filtered_df)}")
#     axs[i].set_xlabel("Date"), axs[i].set_ylabel(names_units[i])
#     axs[i].set_xlim(x_min, x_max)
#     axs[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
#     axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
#     axs[i].set_title(f"{arr_names[i]}")
# plt.tight_layout()
# plt.show()


#set a: all variables available
df_a = df[["yymmdd", "dec_year", "year", "month", "day", "day_of_year", "sin_doy", "cos_doy", "Depth", "set_depths", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]]
df_a = df_a.dropna()
print("Num rows in set A:", len(df_a))
print("Num unique dates in set A:", df_a['yymmdd'].nunique())

#set b: maximizes chl and poc measurements
df_b = df[["yymmdd", "dec_year", "year", "month", "day", "day_of_year", "sin_doy", "cos_doy", "Depth", "set_depths", "Chl", "Temp", "O2", "NO3", "PO4", "POC", "PON", "PP"]]
df_b = df_b.dropna() 
print("Num rows in set B:", len(df_b))
print("Num unique dates in set B:", df_b['yymmdd'].nunique())

#set c: maximizes poc measurements
df_c = df[["yymmdd", "dec_year", "year", "month", "day", "day_of_year", "sin_doy", "cos_doy", "Depth", "set_depths", "Temp", "O2", "NO3", "PO4", "POC", "PON", "PP"]]
df_c = df_c.dropna()
print("Num rows in set C:", len(df_c))
print("Num unique dates in set C:", df_c['yymmdd'].nunique())

#set d: maximizes chl measurements
df_d = df[["yymmdd", "dec_year", "year", "month", "day", "day_of_year", "sin_doy", "cos_doy", "Depth", "set_depths", "Chl", "Temp", "O2", "NO3", "PO4", "PP"]]
df_d = df_d.dropna()
print("Num rows in set D:", len(df_d))
print("Num unique dates in set D:", df_d['yymmdd'].nunique())

# applying chauvenet to all dfs
# start at column 7 (Depth) to avoid date columns
for i in range(7, df_a.shape[1]): df_a = chauvenets_criterion(df_a, df_a.columns[i])
print("Num rows in set A after removing outliers:", len(df_a))
for i in range(7, df_b.shape[1]): df_b = chauvenets_criterion(df_b, df_b.columns[i])
print("Num rows in set B after removing outliers:", len(df_b))
for i in range(7, df_c.shape[1]): df_c = chauvenets_criterion(df_c, df_c.columns[i])
print("Num rows in set C after removing outliers:", len(df_c))
for i in range(7, df_d.shape[1]): df_d = chauvenets_criterion(df_d, df_d.columns[i])
print("Num rows in set D after removing outliers:", len(df_d))
for i in range(15, df.shape[1]): df = chauvenets_criterion(df, df.columns[i])
print("Num rows in dataframe after removing outliers:", len(df_d))
# exporting dataframes
df_a.to_csv('df_sets/df_a.csv', index=False) 
df_b.to_csv('df_sets/df_b.csv', index=False) 
df_c.to_csv('df_sets/df_c.csv', index=False) 
df_d.to_csv('df_sets/df_d.csv', index=False) 

# PAPER FIG 2
# surface vs deep - plotting pp 
fig, axs = plt.subplots(2, 1, figsize=(10, 7))
sc = axs[0].scatter(df['day_of_year'], df['PP'], c=df['Depth'], cmap = 'viridis', s=3, linewidths=0.1)
axs[0].set_xlabel("Day of Year"), axs[0].set_ylabel("PP (mgC/m³/day)")
cbar = plt.colorbar(sc, ax=axs[0])
cbar.set_label("Depth (m)")
cbar.ax.invert_yaxis()
# plot monthly average of pp over doy
df_surface = df[df['set_depths'] == 1]
df_deep = df[df['set_depths'] == 100]
df_surface['month'] = df_surface['yymmdd'].dt.month
surface_monthly_avg = df_surface.groupby('month')['PP'].mean()
df_deep['month'] = df_deep['yymmdd'].dt.month
deep_monthly_avg = df_deep.groupby('month')['PP'].mean()
surface_monthly_std = df_surface.groupby('month')['PP'].std()
deep_monthly_std = df_deep.groupby('month')['PP'].std()
axs[1].plot(surface_monthly_avg.index, surface_monthly_avg.values, marker='o', color = 'navy')
axs[1].errorbar(surface_monthly_avg.index, surface_monthly_avg.values, yerr=surface_monthly_std.values, fmt='o', color='navy', capsize=5)
axs[1].plot(deep_monthly_avg.index, deep_monthly_avg.values, marker='o', color = "#07bb97")
axs[1].errorbar(deep_monthly_avg.index, deep_monthly_avg.values, yerr=deep_monthly_std.values, fmt='o', color='#07bb97', capsize=5)
axs[1].set_xlabel("Month"), axs[1].set_ylabel("Average PP (mgC/m³/day)")
axs[1].set_xticks(range(1, 13))
# legend for second plot
axs[0].legend(["Surface (1m)", "Deep (100m)"], loc="upper right")
axs[1].legend(["Surface (1m)", "Deep (100m)"], loc="upper right")
# Panel labels
axs[0].text(-0.08, 1.05, '(a)', transform=axs[0].transAxes, fontsize=14)
axs[1].text(-0.08, 1.05, '(b)', transform=axs[1].transAxes, fontsize=14)
plt.tight_layout()
plt.show()

# PAPER FIG Supporting Info S1 --------------------------------
# #plotting pp (original data) over time
# split_date = datetime.datetime(2018, 4, 26)  # the train/test split date
# fig, ax = plt.subplots(figsize=(10, 5))
# scatter = ax.scatter(df['yymmdd'], df['PP'], c=df['Depth'], cmap = 'viridis', s=3, linewidths=0.1)
# ax.axvline(split_date, color='red', linestyle='--', linewidth=1, label='Train/Test Split')
# ax.set_xlabel("Date"), ax.set_ylabel("PP (mgC/m³/day)")
# ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax.set_title("Primary Productivity (PP) over Time")
# cbar = plt.colorbar(scatter, ax=ax)
# cbar.set_label("Depth (m)")
# cbar.ax.invert_yaxis()
# plt.tight_layout()
# plt.show()

# PAPER FIG Supporting Info S2 ------------------------------------------------------------------
# multicollinearity plot
# from scipy import stats
# from matplotlib.colors import LinearSegmentedColormap
# # Multicollinearity
# df_cut = df_a[["Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]]
# df_matrix = df_cut.corr(method = 'pearson').round(2)
# colors = ["navy", "aliceblue", "navy"]
# custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
# sns.heatmap(df_matrix, annot=True, cmap=custom_cmap, linewidths=0.1, cbar_kws={'label': 'Correlation Coefficient'}, center = 0, vmin=-1, vmax=1)
# plt.show()
# sns.pairplot(df_c[["Depth", "Chl", "Temp", "O2", "NO3", "PO4", "POC", "PON", "BAC"]], plot_kws={"s": 5})
# plt.show()

# PAPER FIG Supporting Info S3 ------------------------------------------------------------------

# # linear regression subplots of each variable against PP
# from scipy import stats
# # arr_names = ["sin_doy", "cos_doy", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]
# # arr_units =  ["", "", " (m)", " (mg/m3)", " (C)", " (PSS-78)", " (umol/kg)"," (umol/kg)", " (umol/kg)", " (ug/kg)", " (ug/kg)", " (umol/kg)", " (umol/kg)", " (umol/kg)", " (nmol/kg)", " (cells*10^8/kg)", " (mgC/m³/day)"]
# arr_names = ["sin_doy", "cos_doy", "Depth", "Chl", "Temp", "O2", "NO3", "PO4", "PP"]
# arr_units =  ["", "", " (m)", " (mg/m3)", " (C)", " (umol/kg)"," (umol/kg)", " (umol/kg)", " (mgC/m³/day)"]
# names_units = [arr_names_pp + arr_units for arr_names_pp, arr_units in zip(arr_names, arr_units)]
# # linear regression subplots of each variable against PP
# def round_sig(x, sig=2):
#     return round(x, sig - int(f"{x:.1e}".split("e")[1]))
# fig, axs = plt.subplots(4, 4, figsize=(12, 7))
# axs = axs.ravel()
# arr_slopes = []
# for i in range(len(arr_names) - 1):
#     x= df_a[arr_names[i]]
#     axs[i].scatter(x, df_a['PP'], s=5, linewidths=1)
#     axs[i].set_xlabel(names_units[i]), axs[i].set_ylabel('PP (mgC/m³/day)') 
#     axs[i].set_xlim(x.min(), x.max()), axs[i].set_ylim(-0.1)
#     m, b, r_value, p_value, std_err = stats.linregress(x, df_a["PP"])
#     arr_slopes.append(m)
#     alpha = 0.05  # 95% confidence interval
#     t = stats.t.ppf(1 - alpha / 2, len(x) - 2)
#     slope_ci_low = m - t * std_err
#     slope_ci_high = m + t * std_err
#     r_sqrd = r_value**2
#     x_extended = np.linspace(x.min(), x.max(), 500)
#     axs[i].axline(xy1=(0, b), slope=m, linestyle="--", linewidth="1", color="r", label=f'$y = {round_sig(m, sig=2)}x {round_sig(b, sig=2):+}$\n$r^2 = {round_sig(r_sqrd, sig=2)}$')
#     axs[i].fill_between(x_extended, slope_ci_low*x_extended + b, slope_ci_high*x_extended + b, color='red', alpha=0.3)
#     axs[i].legend()
# plt.tight_layout()
# plt.show()