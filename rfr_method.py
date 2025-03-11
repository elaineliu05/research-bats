import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
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
df = df.dropna()
print("Number of rows after dropping NaNs:", len(df))
df = df[df["Chl"] > -100] #drop NaNs
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

for i in range(2, df.shape[1]):
    df = chauvenets_criterion(df, df.columns[i])
print("Number of rows after removing outliers:", len(df))

X = df[["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]]  # Predictors (Independent variables)
Y = df['PP'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # split data
rfr = RandomForestRegressor(n_estimators=300, random_state=0, oob_score=True)
rfr.fit(X_train, Y_train)
Y_pred = rfr.predict(X_test)

oob_score = round(rfr.oob_score_, 4)
print(f'Out-of-Bag Score: {oob_score}') #average prediction error for each test sample
mse = round(mean_squared_error(Y_test, Y_pred), 4)
print("Root mean squared error (RMSE): %.2f" % math.sqrt(mean_squared_error(Y_test, Y_pred)))
r2 = round(r2_score(Y_test, Y_pred), 4)
print(f'R-squared: {r2}')
feature_importances = rfr.feature_importances_
feature_importances = [round(importance, 4) for importance in feature_importances]
print(f'Feature importances: {feature_importances}')


fn=["day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]
cn=["PP"]

# single_tree = rfr.estimators_[0]
# dot_data = export_graphviz(single_tree, 
#                            out_file=None, 
#                            feature_names=X.columns, 
#                            filled=True, 
#                            rounded=True, 
#                            special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.render("random_forest_tree")  # Saves the tree as a file
# graph.view()  # Opens the tree visualization
fig, ax = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rfr.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = True, 
               ax=ax);
for item in ax.get_children():  
    if isinstance(item, plt.Rectangle):  # Check if it's a rectangle (decision node)
        item.set_linewidth(0.5)  # Adjust the border thickness

fig.savefig('rf_onetree.png')
plt.show()
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

#plotting feature importances
fig = plt.figure(figsize=(14, 5))
plt.bar(X.columns, rfr.feature_importances_)
plt.show()

# Grid search
# param_grid = {
#     'n_estimators': [100, 300, 500],
#     'max_depth':[10, 20, 30], 
#     'min_samples_split':[2, 5, 10],
#     'min_samples_leaf':[1, 2, 4],
# }
# from sklearn.model_selection import GridSearchCV
# grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, Y_train)
# best_params = grid_search.best_params_
# print(f'Best parameters: {best_params}')
# best_estimator = grid_search.best_estimator_
# Y_pred = best_estimator.predict(X_test)
# oob_score = round(best_estimator.oob_score_, 4)
# print(f'Out-of-Bag Score: {oob_score}')
# mse = round(mean_squared_error(Y_test, Y_pred), 4)
# print("Root mean squared error (RMSE): %.2f" % math.sqrt(mean_squared_error(Y_test, Y_pred)))
# r2 = round(r2_score(Y_test, Y_pred), 4)
# print(f'R-squared: {r2}')
