import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
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
df_a.to_csv('df_a.csv', index=False) 
print("Number of rows after dropping NaNs:", len(df_a))

# #imputing!
# X_dfa = df_a.drop(columns = ['yymmdd', 'PP'])  # Predictors (Independent variables)
# Y_dfa = df_a['PP']
# imputer = IterativeImputer()
# imputed = imputer.fit_transform(X_dfa)
# imputed_df_a = pd.DataFrame(imputed, columns = X_dfa.columns)
# imputed_df_a['PP'] = Y_dfa
# imputed_df_a.to_csv('imputed_df_a.csv', index=False) 
# #count num rows w an imputed value
# imputed_mask = imputed_df_a.ne(X_dfa)  # Compare element-wise
# imputed_rows = imputed_mask.any(axis=1)  # Check if any column has a change in a row
# num_imputed_rows = imputed_rows.sum()
# print(f"Number of rows with at least one imputed value: {num_imputed_rows}")

df_b = df[["year", "month", "day", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "BAC", "PP"]]
df_b = df_b.dropna() #drop NaNs
df_c = df[["year", "month", "day", "day_of_year", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "BAC", "PP"]]
df_c = df_c.dropna() #drop NaNs
df_d = df[["year", "month", "day", "day_of_year", "Depth", "Temp", "O2", "NO3", "PO4", "PP"]]
df_d = df_d.dropna() #drop NaNs

#chlorophyll
df_a = df_a[df_a["Chl"] > -100] #drop NaNs
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

X = df_a.drop(columns = ['yymmdd', 'PP'])  # Predictors (Independent variables)
Y = df_a['PP'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # split data
rfr = RandomForestRegressor(n_estimators=300, random_state=0, oob_score=True)
rfr.fit(X_train, Y_train)
Y_pred = rfr.predict(X_test)

print('first tree depth: ', rfr.estimators_[0].get_depth(), 'first tree leaves: ', rfr.estimators_[0].get_n_leaves())
print('last tree depth: ', rfr.estimators_[-1].get_depth(), 'last tree leaves: ', rfr.estimators_[-1].get_n_leaves())

oob_score = round(rfr.oob_score_, 4)
print(f'Out-of-Bag Score: {oob_score}') #average prediction error for each test sample
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
axs[1].set_ylabel('Error')
axs[1].legend(loc = 'upper right')
plt.tight_layout()
plt.show()

# #plotting feature importances
# fig = plt.figure(figsize=(14, 5))
# plt.bar(X.columns, rfr.feature_importances_)
# plt.show()

# X = df_d.drop(columns = ['PP'])  # Predictors (Independent variables)
# Y = df_d['PP'] 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # split data
# rfr = RandomForestRegressor(n_estimators=300, random_state=0, max_depth=10, min_samples_split=100, min_samples_leaf=50, oob_score=True)
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

#XGBoost
import xgboost as xgb
xgb = xgb.XGBRegressor(n_estimators = 300, learning_rate = 0.1, max_depth = 10, min_child_weight = 1, subsample = 0.8, colsample_bytree = 0.8, gamma = 0, random_state = 42)
