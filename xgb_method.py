import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.patches as mpatches
from mlr_method import error as mlr_error
from mlr_method import monthly_error_sum as mlr_monthly_error_sum

from rfr_method import error as rfr_error
from rfr_method import monthly_error_sum as rfr_monthly_error_sum

#XGBoost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Load dataset
file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/matched_data_from_BATS_trimmed.csv'
df = pd.read_csv(file_path)
df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce').astype('float64') #apply to everything except yymmdd
df['yymmdd'] = pd.to_datetime(df['yymmdd']) 
df['year'] = df['yymmdd'].dt.year
df['month'] = df['yymmdd'].dt.month
df['day'] = df['yymmdd'].dt.day
print(df.isnull().sum())

print("Number of rows in original dataset:", len(df))
df_a = df
df_a = df_a.dropna() #drop NaNs
# df_a.to_csv('df_a.csv', index=False) 
print("Number of rows after dropping NaNs:", len(df_a))

df_b = df[["year", "month", "day", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "BAC", "PP"]]
df_b = df_b.dropna() #drop NaNs
df_c = df[["year", "month", "day", "day_of_year", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "BAC", "PP"]]
df_c = df_c.dropna() #drop NaNs
df_d = df[["year", "month", "day", "day_of_year", "Depth", "Temp", "O2", "NO3", "PO4", "PP"]]
df_d = df_d.dropna() #drop NaNs

#chlorophyll
df_a = df_a[df_a["Chl"] > -100] #drop weird chlorophylls
print('Number of rows after dropping weird chlorophylls:', len(df_a))
df_b = df_b[df_b["Chl"] > -100] #drop NaNs

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

X = df_c.drop(columns = ['PP'])  # Predictors (Independent variables)
Y = df_c['PP'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # split data
xgb_mod = xgb.XGBRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 10, min_child_weight = 5, subsample = 0.6, colsample_bytree = 0.8, gamma = 0, random_state = 0)
xgb_mod.fit(X_train, Y_train)
Y_pred = xgb_mod.predict(X_test)

mse = round(mean_squared_error(Y_test, Y_pred), 4)
print("XGB Root mean squared error (RMSE): %.2f" % math.sqrt(mean_squared_error(Y_test, Y_pred)))
r2 = round(r2_score(Y_test, Y_pred), 4)
print(f'XGB R-squared: {r2}')
feature_importances = xgb_mod.feature_importances_
feature_importances = [round(importance, 4) for importance in feature_importances]
print(f'XGB Feature importances: {feature_importances}')

#Monte Carlo simulation
predictions = pd.DataFrame()
xgb_rmses = []
xgb_rmse_sd = []
xgb_r2s = []
xgb_r2_sd = []
def xgb_monte_carlo(X, Y):
    Rmse_arr = []
    R2_arr = []
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i) # split data
        xgb_mod = xgb.XGBRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 10, min_child_weight = 5, subsample = 0.6, colsample_bytree = 0.8, gamma = 0, random_state = i)
        xgb_mod.fit(X_train, Y_train)
        Y_pred = xgb_mod.predict(X_test)
        Rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred))) #arr of each rmse in one monte carlo
        R2_arr.append(r2_score(Y_test, Y_pred))                        #arr of each r^2 in one monte carlo
    predictions["Simulations"] = np.arange(1, 11) 
    predictions["RMSE"] = np.around(Rmse_arr, decimals = 3)            #all rmses
    predictions["R^2"] = np.around(R2_arr, decimals = 2)               #all r^2s
    # monte_head = ["Simulation", "Root Mean Squared Error", "R² Score"]
    #print(tabulate(predictions, headers=monte_head))
    print("Average RMSE", predictions['RMSE'].mean())
    xgb_rmses.append(round(predictions['RMSE'].mean(), 2))
    xgb_rmse_sd.append(predictions['RMSE'].std())
    print("Average R²", predictions['R^2'].mean())
    xgb_r2s.append(round(predictions['R^2'].mean(), 3))
    xgb_r2_sd.append(predictions['R^2'].std())
# xgb_monte_carlo(X, Y)

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
axs[0].legend(loc = 'upper right')
# Scatter plot for error
error = Y_test - Y_pred
axs[1].scatter(df.loc[Y_test.index, 'day_of_year'], error, color='darkslateblue', label='Error (Actual - Predicted)', s=10)
axs[1].set_xlabel('Day of Year')
axs[1].set_ylabel('Error (mgC/m³/day)')
axs[1].legend(loc = 'upper right')
plt.tight_layout()
plt.show()

#monthly sum of errors
testing_months = df.loc[Y_test.index, 'month']
residuals_df = pd.DataFrame({
    'month': testing_months,
    'error': error
})
monthly_error_sum = residuals_df.groupby('month')['error'].sum()
plt.figure(figsize=(8, 5))
plt.plot(monthly_error_sum.index, monthly_error_sum.values, marker='o', linestyle='-', color='gold')
plt.xlabel('Month')
plt.ylabel('Sum of Residuals (mgC/m³/day)')
plt.xticks(range(1, 13))
plt.show()

#residual histogram
plt.figure(figsize=(8, 6))
plt.hist(error, color='gold', bins=30, edgecolor='black', alpha=0.5)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

#comparison bar-plot of r^2 and rmse
sets = ["Set A", "Set B", "Set C", "Set D"]
colors = ["#F4A261", "#f6da43", "#46cdb4", "#285f94"]    
X_a = df_a.drop(columns = ['yymmdd', 'PP'])
Y_a = df_a['PP']
X_b = df_b.drop(columns = ['PP'])
Y_b = df_b['PP']
X_c = df_c.drop(columns = ['PP'])
Y_c = df_c['PP']
X_d = df_d.drop(columns = ['PP'])
Y_d = df_d['PP']
# print('set A')
# xgb_monte_carlo(X_a, Y_a)
# print('\nset B')
# xgb_monte_carlo(X_b, Y_b)
# print('\nset C')
# xgb_monte_carlo(X_c, Y_c)
# print('\nset D')
# xgb_monte_carlo(X_d, Y_d)

#residual histogram
plt.figure(figsize=(10, 6))
sns.histplot(mlr_error, color='cornflowerblue', label='MLR Residuals', bins=30, kde = True, alpha=0.8)
sns.histplot(rfr_error, color='yellowgreen', label='RFR Residuals', bins=30, kde = True, alpha=0.8)
sns.histplot(error, color='gold', label='XGB Residuals', bins=30, kde = True, alpha=0.5)
plt.title('Residual Histograms of MLR, RFR, and XGB')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.legend()
plt.show()

#monthly sum of errors line plot
plt.figure()
plt.plot(mlr_monthly_error_sum.index, mlr_monthly_error_sum.values, marker='o', linestyle='-', color='cornflowerblue', label='MLR')
plt.plot(rfr_monthly_error_sum.index, rfr_monthly_error_sum.values, marker='o', linestyle='-', color='yellowgreen', label='RFR')
plt.plot(monthly_error_sum.index, monthly_error_sum.values, marker='o', linestyle='-', color='gold', label='XGB')
plt.xlabel('Month')
plt.ylabel('Sum of Residuals (mgC/m³/day)')
plt.xticks(range(1, 13))
plt.legend()
plt.show()

fig, axs = plt.subplots(1, 2, figsize = (7, 5), sharey=False)
axs[0].bar(sets, xgb_r2s, color=colors)
axs[0].errorbar(sets, xgb_r2s, yerr=xgb_r2_sd, fmt="o", color="k", capsize=3)
axs[0].set_ylabel('Average R^2 Score')
axs[0].set_ylim(0, 0.8) 
axs[1].bar(sets, xgb_rmses, color=colors)
axs[1].errorbar(sets, xgb_rmses, yerr=xgb_rmse_sd, fmt="o", color="k", capsize=3)
axs[1].set_ylabel('Average RMSE (mgC/m3/day)')
axs[1].set_ylim(0, 2)   
print('rmse array: ', xgb_rmses)
print('rmse standard deviations: ', xgb_rmse_sd)
plt.tight_layout()
plt.show()

mlr_r2s = [0.56, 0.58, 0.53, 0.52]
mlr_r2_sd = [0.039, 0.033, 0.028, 0.022]
mlr_rmses = [1.82, 1.74, 2.081, 2.2462]
mlr_rmse_sd = [0.113, 0.099, 0.094, 0.126]
rfr_r2s = [0.692, 0.686, 0.666, 0.696]
rfr_r2_sd = [0.033, 0.045, 0.037, 0.024]
rfr_rmses = [1.51, 1.5, 1.76, 1.78]
rfr_rmse_sd =  [0.067, 0.109, 0.0875, 0.116]
xgb_r2s = [0.709, 0.692, 0.685, 0.717]
xgb_r2_sd = [0.039, 0.057, 0.035, 0.029]
xgb_rmses = [1.47, 1.49, 1.71, 1.72]
xgb_rmse_sd = [0.070, 0.142, 0.100, 0.130]

#r2 comparison
plt.figure(figsize=(10, 6))
r2_data = np.array([mlr_r2s, rfr_r2s, xgb_r2s])
sd_data = np.array([mlr_r2_sd, rfr_r2_sd, xgb_r2_sd])
model_labels = ['MLR', 'RFR', 'XGB']
patterns = ['/', 'x', '.']  
x = np.arange(len(sets))
width = 0.25
for i, (r2_vals, r2_sds, model) in enumerate(zip(r2_data, sd_data, model_labels)):
    plt.bar(x + i * width, r2_vals, width=width, label=model, color=colors, hatch=patterns[i], edgecolor='black', yerr=r2_sds, capsize=5, ecolor='black', error_kw={'elinewidth': 1})
plt.xticks(x + width, sets)  
plt.xlabel("Predictor Sets")
plt.ylabel("R² Score")
plt.title("R² Score Comparison Across Models and Predictor Sets")
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='/', label="MLR"),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='x', label="RFR"),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='.', label="XGB")
]
plt.legend(handles = legend_handles, title="Model", loc="upper center",  handleheight=5.5, handlelength=4, ncol=3)
plt.ylim(0, 1)  # Adjust based on R² range
plt.show()

#rmse comparison
plt.figure(figsize=(10, 6))
rmse_data = np.array([mlr_rmses, rfr_rmses, xgb_rmses])
sd_data = np.array([mlr_rmse_sd, rfr_rmse_sd, xgb_rmse_sd])
model_labels = ['MLR', 'RFR', 'XGB']
patterns = ['/', 'x', '.']  
x = np.arange(len(sets))
width = 0.25
for i, (rmse_vals, rmse_sds, model) in enumerate(zip(rmse_data, sd_data, model_labels)):
    plt.bar(x + i * width, rmse_vals, width=width, label=model, color=colors, hatch=patterns[i], edgecolor='black', yerr=rmse_sds, capsize=5, ecolor='black', error_kw={'elinewidth': 1})
plt.xticks(x + width, sets)  
plt.xlabel("Predictor Sets")
plt.ylabel("RMSE (mgC/m³/day)")
plt.title("RMSE Comparison Across Models and Predictor Sets")
import matplotlib.patches as mpatches
legend_handles = [
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='/', label="MLR"),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='x', label="RFR"),
    mpatches.Patch(facecolor='white', edgecolor='black', hatch='.', label="XGB")
]
plt.legend(handles = legend_handles, title="Model", loc="upper center",  handleheight=5.5, handlelength=4, ncol=3)
plt.ylim(0, 2.7)  # Adjust based on R² range
plt.show()

# param_grid = {
#     'n_estimators': [500, 700], 
#     'learning_rate': [0.01, 0.05, 0.1],
#     'max_depth': [10, 20],
#     'min_child_weight': [5, 10], 
#     'subsample': [0.6, 0.8],
#     'colsample_bytree': [0.7, 0.8, 0.9],
#     'gamma': [0, 0.05, 0.1]
# }
# grid_search = GridSearchCV(estimator=xgb_mod, param_grid=param_grid, scoring='r2', cv=5, verbose=1, n_jobs=-1)
# grid_search.fit(X1_train, Y1_train)
# best_params = grid_search.best_params_
# print("Best parameters found: ", best_params)
# best_xgb = xgb.XGBRegressor(objective='reg:squarederror', **best_params, random_state=0)
# best_xgb.fit(X1_train, Y1_train)
# # Predict and evaluate
# Y1_pred = best_xgb.predict(X1_test)
# r2 = r2_score(Y1_test, Y1_pred)
# print(f'Best XGB R-squared: {r2:.4f}')

# #plotting feature importances
# fig = plt.figure(figsize=(14, 5))
# plt.bar(X1.columns, xgb_mod.feature_importances_)
# plt.show()