import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

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

#Random Forest Regression 
X = df_c.drop(columns = ['PP'])  # Predictors (Independent variables)
Y = df_c['PP'] 
#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # split data
# make model
rfr = RandomForestRegressor(n_estimators=300, random_state=0, oob_score=True)
rfr.fit(X_train, Y_train)
#make predictions
Y_pred = rfr.predict(X_test)

print('first tree depth: ', rfr.estimators_[0].get_depth(), 'first tree leaves: ', rfr.estimators_[0].get_n_leaves())
print('last tree depth: ', rfr.estimators_[-1].get_depth(), 'last tree leaves: ', rfr.estimators_[-1].get_n_leaves())

mse = round(mean_squared_error(Y_test, Y_pred), 4)
print("Root mean squared error (RMSE): %.2f" % math.sqrt(mean_squared_error(Y_test, Y_pred)))
r2 = round(r2_score(Y_test, Y_pred), 4)
print(f'R-squared: {r2}')
feature_importances = rfr.feature_importances_
feature_importances = [round(importance, 4) for importance in feature_importances]
print(f'Feature importances: {feature_importances}')

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

#Monte Carlo simulation
predictions = pd.DataFrame()
rmses = []
rmse_SD = []
r2S = []
r2_SD = []
def rfr_monte_carlo(X, Y):
    all_resid = []
    month_resid = []
    averages_arr = []
    rmse_arr = []
    R2_arr = []
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i) # split data
        rfr = RandomForestRegressor(n_estimators=500, random_state=i, oob_score=True)
        rfr.fit(X_train, Y_train)
        Y_pred = rfr.predict(X_test)
        #metrics
        rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred))) #arr of each rmse in one monte carlo
        R2_arr.append(r2_score(Y_test, Y_pred))  
        #residuals
        resid_arr = Y_test - Y_pred #residual array
        all_resid.append(resid_arr)
        #monthly sum stuff
        resid_arr.index = df.loc[Y_test.index, 'month'].values
        month_resid.append(resid_arr)
        averages = resid_arr.groupby(resid_arr.index).mean()
        averages_arr.append(averages) #append monthly averages to list

    month_resid = pd.concat(month_resid) #flatten into dataframe
    monthly_average = month_resid.groupby(month_resid.index).mean()
    monthly_avg_df = pd.DataFrame(averages_arr)
    print('monthly average df', monthly_avg_df)
    monthly_std = monthly_avg_df.std()
    print('monthly std', monthly_std)

    all_resid = np.concatenate(all_resid) #flatten array 
    predictions["Simulations"] = np.arange(1, 11) 
    predictions["RMSE"] = np.around(rmse_arr, decimals = 2)            #all rmses
    predictions["R^2"] = np.around(R2_arr, decimals = 2)               #all r^2s
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
# X_a = df_a.drop(columns = ['yymmdd', 'PP'])
# Y_a = df_a['PP']
# X_b = df_b.drop(columns = ['PP'])
# Y_b = df_b['PP']
X_c = df_c.drop(columns = ['PP'])
Y_c = df_c['PP']
# X_d = df_d.drop(columns = ['PP'])
# Y_d = df_d['PP']
# print('set A')
# rfr_monte_carlo(X_a, Y_a)
# print('\nset B')
# rfr_monte_carlo(X_b, Y_b)
print('\nset C')
all_resid, monthly_average, monthly_std = rfr_monte_carlo(X_c, Y_c)
# print('\nset D')
# rfr_monte_carlo(X_d, Y_d)

# # R^2 and RMSE bar chart
# fig, axs = plt.subplots(1, 2, figsize = (7, 5), sharey=False)
# axs[0].bar(categories, R2S, color=colors)
# axs[0].errorbar(categories, R2S, yerr=R2_SD, fmt="o", color="k", capsize=3)
# axs[0].set_ylabel('Average R^2 Score')
# axs[0].set_ylim(0, 0.8) 
# axs[1].bar(categories, RMSES, color=colors)
# axs[1].errorbar(categories, RMSES, yerr=RMSE_SD, fmt="o", color="k", capsize=3)
# axs[1].set_ylabel('Average RMSE (mgC/m3/day)')
# axs[1].set_ylim(0, 2.1)  
# # print('rmse array: ', RMSES)
# # print('rmse standard deviations: ', RMSE_SD)
# plt.tight_layout()
# plt.show()

# residual histogram
plt.figure(figsize=(8, 6))
sns.histplot(all_resid, color='yellowgreen', bins=30, kde = True, alpha=0.8)
plt.xlabel('Residual (mgC/m³/day)')
plt.ylabel('Frequency')
plt.show()
#monthly sum of residuals lineplot
plt.figure(figsize=(8, 5))
plt.errorbar(monthly_average.index, monthly_average.values, yerr=monthly_std.values, fmt='o-', color='yellowgreen', ecolor='#c9e675', capsize=3)
plt.xlabel('Month')
plt.ylabel('Average Residuals (mgC/m³/day)')
plt.xticks(range(1, 13))
plt.show()

# # gridsearch for hyperparameter tuning
# param_grid = {
#     'n_estimators': [300, 500], 
#     'max_depth': [15, 20],
#     'max_features': ["sqrt", None],
#     'min_samples_split': [2, 6, 10],
#     'min_samples_leaf': [1, 3, 5],
#     'min_impurity_decrease': [0.0, 0.01]
# }
# grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, scoring='r2', cv=5, verbose=1, n_jobs=-1)
# grid_search.fit(X_train, Y_train)
# best_rfr = grid_search.best_estimator_
# Y_pred_best = best_rfr.predict(X_test)
# r2 = r2_score(Y_test, Y_pred_best)
# # Predict and evaluate
# print(f"Best Parameters: {grid_search.best_params_}")
# print(f"Best R² Score on Test Set: {r2:.4f}")


# X = df_d.drop(columns = ['PP'])  # Predictors (Independent variables)
# Y = df_d['PP'] 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # split data
# rfr = RandomForestRegressor(n_estimators=300, random_state=0, max_depth=10, min_samples_split=140, min_samples_leaf=70, oob_score=True)
# rfr.fit(X_train, Y_train)
# Y_pred = rfr.predict(X_test)
# r2 = round(r2_score(Y_test, Y_pred), 4)
# print(f'Simple tree R-squared: {r2}')
# dot_data = export_graphviz(rfr.estimators_[0], 
#                            out_file=None, 
#                            feature_names=X.columns,
#                            filled=True,
#                            rounded=True,
#                            special_characters=True,
#                            impurity=False,  # Hides MSE
#                            proportion=False, # Hides proportion of samples
#                            precision=2,
#                            )

# graph = graphviz.Source(dot_data)
# graph.render("tree_visualization", format="png", cleanup=True)  # Saves as PNG
# graph.view()  # Opens the visualization
