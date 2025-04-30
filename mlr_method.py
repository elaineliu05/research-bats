import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import norm


# Load dataset
file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/matched_data_from_BATS_trimmed.csv'
df = pd.read_csv(file_path)
df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce').astype('float64') #apply to everything except yymmdd
df['yymmdd'] = pd.to_datetime(df['yymmdd']) # convert to date time
df['year'] = df['yymmdd'].dt.year           # extract year
df['month'] = df['yymmdd'].dt.month         # extract month
df['day'] = df['yymmdd'].dt.day             # extract day

#set a: all variables available
print("Number of rows in original dataset:", len(df))
df_a = df
df_a = df_a.dropna()
print("Number of rows after dropping NaNs:", len(df_a))
#set b: variables not measured prior to 1994 removed (best results)
df_b = df[["yymmdd", "year", "month", "day", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "BAC", "PP"]]
df_b = df_b.dropna() 
#set c: variables with consistent data
df_c = df[["yymmdd", "year", "month", "day", "day_of_year", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "BAC", "PP"]]
df_c = df_c.dropna()
#set d: most common measurements
df_d = df[["yymmdd", "year", "month", "day", "day_of_year", "Depth", "Temp", "O2", "NO3", "PO4", "PP"]]
df_d = df_d.dropna()

#drop duplicates
df_arr = [df_a, df_b, df_c, df_d]
for i in range(len(df_arr)):
    df_arr[i] = df_arr[i].drop_duplicates(subset=['yymmdd', 'Depth'], keep='first') #drop duplicates

#labels for graphs
arr_names_pp_dfa = ["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]
arr_units_dfa = ["", " (m)", " (mg/m3)", " (C)", " (PSS-78)", " (umol/kg)"," (umol/kg)", " (umol/kg)", " (ug/kg)", " (ug/kg)", " (umol/kg)", " (umol/kg)", " (umol/kg)", " (nmol/kg)", " (cells*10^8/kg)", " (mgC/m³/day)"]
names_units_dfa = [arr_names_pp + arr_units for arr_names_pp, arr_units in zip(arr_names_pp_dfa, arr_units_dfa)]

#Chlorophyll
df_a = df_a[df_a["Chl"] > -100] #drop NaNs
df_b = df_b[df_b["Chl"] > -100] #drop NaNs
# plt.scatter(df["yymmdd"], df["Chl"], s=5)
# plt.xlabel('time')
# plt.ylabel('Chl mg/m3')
# plt.tight_layout()
# plt.show()

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

for i in range(2, df_a.shape[1]): df_a = chauvenets_criterion(df_a, df_a.columns[i])
print("Num rows in set A after removing outliers:", len(df_a))
for i in range(2, df_b.shape[1]): df_b = chauvenets_criterion(df_b, df_b.columns[i])
print("Num rows in set B after removing outliers:", len(df_b))
for i in range(2, df_c.shape[1]): df_c = chauvenets_criterion(df_c, df_c.columns[i])
print("Num rows in set C after removing outliers:", len(df_c))
for i in range(2, df_d.shape[1]): df_d = chauvenets_criterion(df_d, df_d.columns[i])
print("Num rows in set D after removing outliers:", len(df_d))

def round_sig(x, sig=2):
    return round(x, sig - int(f"{x:.1e}".split("e")[1]))
#Linear regression subplots of each variable against PP
fig, axs = plt.subplots(4, 4, figsize=(12, 7))
axs = axs.ravel()
arr_slopes = []
for i in range(len(arr_names_pp_dfa) - 1):
    x= df_a[arr_names_pp_dfa[i]]
    axs[i].scatter(x, df_a['PP'], s=5, linewidths=1)
    axs[i].set_xlabel(names_units_dfa[i]), axs[i].set_ylabel('PP (mgC/m³/day)') 
    axs[i].set_xlim(x.min(), x.max()), axs[i].set_ylim(-0.1)
    m, b, r_value, p_value, std_err = stats.linregress(x, df_a["PP"])
    arr_slopes.append(m)
    alpha = 0.05  # 95% confidence interval
    t = stats.t.ppf(1 - alpha / 2, len(x) - 2)
    slope_ci_low = m - t * std_err
    slope_ci_high = m + t * std_err
    r_sqrd = r_value**2
    x_extended = np.linspace(x.min(), x.max(), 500)
    axs[i].axline(xy1=(0, b), slope=m, linestyle="--", linewidth="1", color="r", label=f'$y = {round_sig(m, sig=2)}x {round_sig(b, sig=2):+}$\n$r^2 = {round_sig(r_sqrd, sig=2)}$')
    axs[i].fill_between(x_extended, slope_ci_low*x_extended + b, slope_ci_high*x_extended + b, color='red', alpha=0.3)
    axs[i].legend()
plt.tight_layout()
plt.show()

#Multiple Linear Regression
X = df_c[["day_of_year", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "BAC"]]  # Predictors (Independent variables)
Y = df_c['PP']                                                                              # Response (Dependent variable)
#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# make model
model = LinearRegression()
model.fit(X_train, Y_train)
# predict
Y_pred = model.predict(X_test) 
# model evaluation
print("Coefficients:", np.around(model.coef_, decimals = 3))
print("Intercept:", model.intercept_)
print("Root mean squared error (RMSE): %.2f" % math.sqrt(mean_squared_error(Y_test, Y_pred)))
print("R² score: %.2f" % r2_score(Y_test, Y_pred), end="\n")

#Plot predictions 
fig, axs = plt.subplots(2, 1)
# Scatter plot for actual PP values
axs[0].scatter(df.loc[Y_test.index, 'day_of_year'], Y_test, color='lightskyblue', label='Actual PP', s=10)
axs[0].scatter(df.loc[Y_test.index, 'day_of_year'], Y_pred, color='salmon', label='Predicted PP', s=10)
neg_acc = np.sum(Y_test < 0) 
neg_pred = np.sum(Y_pred < 0)
print("Percent of negative values:", np.around((neg_acc / len(Y_test) * 100), decimals = 3))
print("Percent of negative predictions:", neg_pred / len(Y_pred) * 100)
axs[0].set_xlabel('Day of Year')
axs[0].set_ylabel('Primary Productivity (mgC/m³/day)')
axs[0].legend(loc = 'upper right')
# Scatter plot for residuals (test set)
resid = Y_test - Y_pred
axs[1].scatter(df.loc[Y_test.index, 'day_of_year'], resid, color='darkslateblue', label='Error (Actual - Predicted)', s=10)
axs[1].set_xlabel('Day of Year')
axs[1].set_ylabel('Error (mgC/m³/day)')
axs[1].legend(loc = 'upper right')
plt.tight_layout()
plt.show()

#Monte Carlo simulation
predictions = pd.DataFrame()
rmses = []
rmse_SD = []
r2S = []
r2_SD = []
def monte_carlo(X, Y):
    all_resid = []
    month_resid = []
    averages_arr = []
    rmse_arr = []
    R2_arr = []
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        # metrics
        rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred))) #arr of each rmse in one monte carlo
        R2_arr.append(r2_score(Y_test, Y_pred))  
        # residuals
        resid_arr = Y_test - Y_pred 
        all_resid.append(resid_arr)
        # monthly sum stuff
        resid_arr.index = df.loc[Y_test.index, 'month'].values
        month_resid.append(resid_arr)
        averages = resid_arr.groupby(resid_arr.index).mean()
        averages_arr.append(averages)
    
    month_resid = pd.concat(month_resid) #flatten into dataframe
    monthly_average = month_resid.groupby(month_resid.index).mean()
    monthly_avg_df = pd.DataFrame(averages_arr)
    print('monthly average df', monthly_avg_df)
    monthly_std = monthly_avg_df.std()
    print('monthly std', monthly_std)

    all_resid = np.concatenate(all_resid) #flatten array 
    predictions["Simulations"] = np.arange(1, 11) 
    predictions["RMSE"] = np.around(rmse_arr, decimals = 3) #all rmses
    predictions["R^2"] = np.around(R2_arr, decimals = 2)    #all r^2s
    print("Average RMSE", predictions['RMSE'].mean())
    rmses.append(predictions['RMSE'].mean())
    rmse_SD.append(predictions['RMSE'].std())
    print("Average R²", predictions['R^2'].mean())
    r2S.append(round(predictions['R^2'].mean(), 2))
    r2_SD.append(predictions['R^2'].std())
    return all_resid, monthly_average, monthly_std

#comparison bar-plot of r^2 and rmse
categories = ["Set A", "Set B", "Set C", "Set D"]
colors = ["#F4A261", "#f6da43", "#46cdb4", "#285f94"]    
# X_a = df_a[["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]]
# Y_a = df_a['PP']
# X_b = df_b[["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "BAC"]]
# Y_b = df_b['PP']
X_c = df_c[["day_of_year", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "BAC"]]
Y_c = df_c['PP']
# X_d = df_d[["day_of_year", "Depth", "Temp", "O2", "NO3", "PO4"]]
# Y_d = df_d['PP']
# print('set A')
# monte_carlo(X_a, Y_a)
# print('\nset B')
# monte_carlo(X_b, Y_b)
print('\nset C')
all_resid, monthly_average, monthly_std = monte_carlo(X_c, Y_c)
# print('\nset D')
# monte_carlo(X_d, Y_d)

# # R^2 and RMSE bar chart
# fig, axs = plt.subplots(1, 2, figsize = (7, 5), sharey=False)
# axs[0].bar(categories, R2S, color=colors)
# axs[0].errorbar(categories, R2S, yerr=R2_SD, fmt="o", color="k", capsize=3)
# axs[0].set_ylabel('Average R^2 Score')
# axs[0].set_ylim(0, 0.75) 
# axs[1].bar(categories, RMSES, color=colors)
# axs[1].errorbar(categories, RMSES, yerr=RMSE_SD, fmt="o", color="k", capsize=3)
# axs[1].set_ylabel('Average RMSE (mgC/m3/day)')
# axs[1].set_ylim(0, 2.5)  
# # print('rmse array: ', RMSES)
# # print('rmse standard deviations: ', RMSE_SD) 
# plt.tight_layout()
# plt.show()

# residual histogram
plt.figure(figsize=(8, 6))
sns.histplot(all_resid, color='royalblue', bins=30, kde = True, alpha=0.8)
plt.xlabel('Residual  (mgC/m³/day)')
plt.ylabel('Frequency')
plt.show()
# monthly sum of residuals lineplot
plt.figure(figsize=(8, 5))
plt.errorbar(monthly_average.index, monthly_average.values, yerr=monthly_std.values, fmt='o-', color='royalblue', ecolor='cornflowerblue', capsize=3)
plt.xlabel('Month')
plt.ylabel('Average Residuals (mgC/m³/day)')
plt.xticks(range(1, 13))
plt.show()

#Multicollinearity
# df_cut = df_a[["Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]]
# df_matrix = df_cut.corr(method = 'pearson').round(2)
# colors = ["navy", "aliceblue", "navy"]
# custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
# sns.heatmap(df_matrix, annot=True, cmap=custom_cmap, linewidths=0.1, cbar_kws={'label': 'Correlation Coefficient'}, center = 0, vmin=-1, vmax=1)
# plt.show()
# sns.pairplot(df_c[["Depth", "Chl", "Temp", "O2", "NO3", "PO4", "POC", "PON", "BAC"]], plot_kws={"s": 5})
# plt.show()

# #contour plot
# main_depths = [1, 20, 40, 60, 80, 100, 120, 140] #adjust depth values
# def adjust_depth(value):
#     for main_depth in main_depths:
#         if abs(value - main_depth) <= 10:
#             return main_depth 
#     return value 
# df['Adjusted_Depth'] = df['Depth'].apply(adjust_depth)
# df_nodup = df.drop_duplicates(subset=['yymmdd', 'Adjusted_Depth'], keep='first') #drop duplicates
# df_pivot = df_nodup.pivot(index='Adjusted_Depth', columns='yymmdd', values='PP').fillna(0) #pivot
# X, Y = np.meshgrid(df_pivot.columns, df_pivot.index)
# Z = df_pivot.values  
# plt.figure(figsize=(10, 6))
# contour = plt.contourf(X, Y, Z, cmap='viridis', levels=15)
# plt.colorbar(contour, label="Primary Productivity (mgC/m³/day)")
# plt.gca().invert_yaxis()  
# plt.xlabel("Date")
# plt.ylabel("Depth (m)")
# plt.title("Time Series Contour Plot of Primary Productivity")
# plt.show()