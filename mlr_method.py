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
from tabulate import tabulate
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib.colors import LinearSegmentedColormap

# Load dataset
file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/matched_data_from_BATS.xlsx'
df = pd.read_excel(file_path)
df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce').astype('float64') #apply to everything except yymmdd

df['yymmdd'] = pd.to_datetime(df['yymmdd']) 
# df['day_of_year'] = df['yymmdd'].dt.dayofyear
# df['PP'] = df['PP'] * 12 #converting from mmolC to mgC
df_a = df
df_a = df_a.dropna() #drop NaNs
df_b = df[["yymmdd", "day_of_year", "Depth", "Chl", "Temp", "Sal", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]]
df_c = df[["yymmdd", "day_of_year", "Depth", "Chl", "Temp", "O2", "NO3", "PO4", "POC", "PON", "TOP", "BAC", "PP"]]
df_c = df_c.dropna() #drop NaNs
df_d = df[["yymmdd", "day_of_year", "Depth", "Temp", "O2", "NO3", "PO4", "PP"]]
df_d = df_d.dropna() #drop NaNs

print("Number of rows in original dataset:", len(df_b))
df_b = df_b.dropna() #drop NaNs
print("Number of rows after dropping NaNs:", len(df_b))

df_arr = [df_a, df_b, df_c, df_d]

for i in range(len(df_arr)):
    df_arr[i] = df_arr[i].drop_duplicates(subset=['yymmdd', 'Depth'], keep='first') #drop duplicates
    
arr_names_pp_dfb = ["day_of_year", "Depth", "Chl", "Temp", "Sal", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC", "PP"]
arr_units_dfb = ["", " (m)", " (mg/m3)", " (C)", " (PSS-78)", " (umol/kg)", " (umol/kg)", " (ug/kg)", " (ug/kg)", " (umol/kg)", " (umol/kg)", " (umol/kg)", " (umol/kg)", " (cells*10^8/kg)", " (mgC/m³/day)"]
names_units_dfb = [arr_names_pp + arr_units for arr_names_pp, arr_units in zip(arr_names_pp_dfb, arr_units_dfb)]

arr_names_pp_dfc = ["day_of_year", "Depth", "Chl", "Temp", "O2", "NO3", "PO4", "POC", "PON", "TOP", "BAC", "PP"]
arr_units_dfc = ["", " (m)", " (mg/m3)", " (C)", " (umol/kg)", " (umol/kg)", " (ug/kg)", " (ug/kg)", " (umol/kg)", " (umol/kg)", " (cells*10^8/kg)", " (mgC/m³/day)"]

#Chlorophyll
df_a = df_a[df_a["Chl"] > -100] #drop NaNs
df_b = df_b[df_b["Chl"] > -100] #drop NaNs
df_c = df_c[df_c["Chl"] > -100] #drop NaNs
# df_d = df_d[df_d["Chl"] > -100] #drop NaNs
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

for i in range(2, df_a.shape[1]):
    df_a = chauvenets_criterion(df_a, df_a.columns[i])
print("Number of rows in set A after removing outliers:", len(df_a))
for i in range(2, df_b.shape[1]):
    df_b = chauvenets_criterion(df_b, df_b.columns[i])
print("Number of rows in set B after removing outliers:", len(df_b))
for i in range(2, df_c.shape[1]):
    df_c = chauvenets_criterion(df_c, df_c.columns[i])
print("Number of rows in set C after removing outliers:", len(df_c))
for i in range(2, df_d.shape[1]):
    df_d = chauvenets_criterion(df_d, df_d.columns[i])
print("Number of rows in set D after removing outliers:", len(df_d))

def round_sig(x, sig=2):
    return round(x, sig - int(f"{x:.1e}".split("e")[1]))
#Create subplot with each variable against PP with linear regression line
fig, axs = plt.subplots(4, 4, figsize=(12, 7))
axs = axs.ravel()
arr_slopes = []
for i in range(len(arr_names_pp_dfb) - 1):
    x= df_b[arr_names_pp_dfb[i]]
    axs[i].scatter(x, df_b['PP'], s=5, linewidths=1)
    axs[i].set_xlabel(names_units_dfb[i]), axs[i].set_ylabel('PP (mgC/m³/day)') 
    axs[i].set_xlim(x.min(), x.max()), axs[i].set_ylim(-0.1)
    m, b, r_value, p_value, std_err = stats.linregress(x, df_b["PP"])
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
X = df_a[["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]]  # Predictors (Independent variables)
Y = df_a['PP']                                                                                     # Response (Dependent variable)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) # split data
# make model
model = LinearRegression()
model.fit(X_train, Y_train) # train model
Y_pred = model.predict(X_test) # predict test data
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
print("Percentage of negative values:", neg_acc / len(Y_test) * 100)
print("Percentage of negative predictions:", neg_pred / len(Y_pred) * 100)
axs[0].set_xlabel('Day of Year')
axs[0].set_ylabel('Primary Productivity (mgC/m³/day)')
axs[0].legend()
# Scatter plot for error
error = Y_test - Y_pred
axs[1].scatter(df.loc[Y_test.index, 'day_of_year'], error, color='darkslateblue', label='Error (Actual - Predicted)', s=10)
axs[1].set_xlabel('Day of Year')
axs[1].set_ylabel('Error')
axs[1].legend()
plt.tight_layout()
plt.show()

#Multicollinearity
df_cut = df_a[["Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]]
df_matrix = df_cut.corr(method = 'pearson').round(2)
colors = ["navy", "aliceblue", "navy"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
sns.heatmap(df_matrix, annot=True, cmap=custom_cmap, linewidths=0.1, cbar_kws={'label': 'Correlation Coefficient'}, center = 0, vmin=-1, vmax=1)
plt.show()

# sns.pairplot(df_c[["Depth", "Chl", "Temp", "O2", "NO3", "PO4", "POC", "PON", "BAC"]], plot_kws={"s": 5})
# plt.show()

#Monte Carlo simulation
predictions = pd.DataFrame()
RMSES = []
RMSE_SD = []
R2S = []
R2_SD = []
def monte_carlo(X, Y):
    RMSE_arr = []
    R2_arr = []
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        RMSE_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred))) #arr of each rmse in one monte carlo
        R2_arr.append(r2_score(Y_test, Y_pred))                        #arr of each r^2 in one monte carlo
    predictions["Simulations"] = np.arange(1, 11) 
    predictions["RMSE"] = np.around(RMSE_arr, decimals = 3)            #all rmses
    predictions["R^2"] = np.around(R2_arr, decimals = 2)               #all r^2s
    monte_head = ["Simulation", "Root Mean Squared Error", "R² Score"]
    #print(tabulate(predictions, headers=monte_head))
    print("Average RMSE", predictions['RMSE'].mean())
    RMSES.append(predictions['RMSE'].mean())
    RMSE_SD.append(predictions['RMSE'].std())
    print("Average R²", predictions['R^2'].mean())
    R2S.append(round(predictions['R^2'].mean(), 2))
    R2_SD.append(predictions['R^2'].std())

#comparison bar-plot of r^2 and rmse
categories = ["Set A", "Set B", "Set C", "Set D"]
colors = ["#F4A261", "#f6da43", "#46cdb4", "#285f94"]    
X_a = df_a[["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]]
Y_a = df_a['PP']
X_b = df_b[["day_of_year", "Depth", "Chl", "Temp", "Sal", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]]
Y_b = df_b['PP']
X_c = df_c[["day_of_year", "Depth", "Chl", "Temp", "O2", "NO3", "PO4", "POC", "PON", "TOP", "BAC"]]
Y_c = df_c['PP']
X_d = df_d[["day_of_year", "Depth", "Temp",  "O2", "NO3", "PO4"]]
Y_d = df_d['PP']
print('set A')
monte_carlo(X_a, Y_a)
print('\nset B')
monte_carlo(X_b, Y_b)
print('\nset C')
monte_carlo(X_c, Y_c)
print('\nset D')
monte_carlo(X_d, Y_d)

fig, axs = plt.subplots(1, 2, figsize = (7, 5), sharey=False)
axs[0].bar(categories, R2S, color=colors)
axs[0].errorbar(categories, R2S, yerr=R2_SD, fmt="o", color="k")
axs[0].set_ylabel('Average R^2 Score')
axs[0].set_ylim(0, 1) 
axs[1].bar(categories, RMSES, color=colors)
axs[1].errorbar(categories, RMSES, yerr=RMSE_SD, fmt="o", color="k")
axs[1].set_ylabel('Average RMSE')
axs[1].set_ylim(0, 3.2)   
plt.tight_layout()
plt.show()

# #property-property plot
# plt.scatter(Y_test, Y_pred)
# # 1:1 line (y = x)
# min_val = min(min(Y_test), min(Y_pred))
# max_val = max(max(Y_test), max(Y_pred))
# plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="1:1 Line")
# plt.xlabel('Actual PP (mgC/m³/day)')
# plt.ylabel('Predicted PP (mgC/m³/day)')
# plt.title("Property-Property Plot of Predicted v Actual PP")
# plt.tight_layout()
# plt.show()

#Converting pp to mgC/m2/day
df_d['Depth_intervals'] = df_d.groupby('yymmdd')['Depth'].diff().fillna(1) #calculate depth intervals
df_d['PP_m2'] = (df_d['PP'] + df_d['PP'].shift(-1)) / 2 * df_d['Depth_intervals'] #trapezoidal integration method
df_d['PP_m2'] = df_d['PP_m2'].fillna(0)  # Handle NaN in the last interval
PP_m2_day = df_d.groupby('yymmdd')['PP_m2'].sum().reset_index() #adding up values for each day

# start = '1987-01-01'
# end = '1997-6-15'
# df_slice = PP_m2_day[(PP_m2_day['yymmdd'] >= start) & (PP_m2_day['yymmdd'] <= end)]

plt.plot(PP_m2_day['yymmdd'], PP_m2_day['PP_m2'])
plt.ylim(0, 1200)
plt.xlabel('Date')
plt.ylabel('Primary Productivity (mgC/m2/day)')
plt.show()

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

# #Comparing MLR and linear reg 
# slopes = pd.DataFrame()
# slopes["Var"] = arr_names_pp[:-1]
# slopes["MLR"] = model.coef_
# slopes["LinReg"] = arr_slopes
# slope_head = ["Var", "MLR", "LinReg"]
# print(tabulate(slopes, headers=slope_head))

# VIF 
# vif_data = pd.DataFrame()
# vif_data["feature"] = df_matrix.columns
# vif_data["VIF"] = [variance_inflation_factor(df_matrix.values, i) # calculating VIF for each feature
#     for i in range(len(df_matrix.columns))]
# print(vif_data)