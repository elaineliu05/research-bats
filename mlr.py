import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

#importing df sets
df_a = pd.read_csv('df_sets/df_a.csv')
df_b = pd.read_csv('df_sets/df_b.csv')
df_c = pd.read_csv('df_sets/df_c.csv')
df_d = pd.read_csv('df_sets/df_d.csv')

# choose df set
my_df = df_c

#Multiple Linear Regression
X = my_df.drop(columns=['PP', 'yymmdd', 'day_of_year'])
Y = my_df['PP']                                                               
#splitting data, test set is last 15%
test_size = int(0.15 * len(my_df))
X_trainval, X_test = X[:-test_size], X[-test_size:]
Y_trainval, Y_test = Y[:-test_size], Y[-test_size:]

n_runs = 10
rmse_arr, r2_arr, mae_arr = [], [], []
month_resid = []

for i in range(n_runs):
    # fit model (deterministic)
    mlr = LinearRegression()
    mlr.fit(X_trainval, Y_trainval)

    Y_pred = mlr.predict(X_test)

    rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred)))
    r2_arr.append(r2_score(Y_test, Y_pred))
    mae_arr.append(mean_absolute_error(Y_test, Y_pred))

    # residuals by month
    resid = pd.Series(Y_test - Y_pred, index=Y_test.index)
    resid.index = my_df.loc[resid.index, 'month'].values
    month_resid.append(resid)

month_resid_df = pd.concat(month_resid)
monthly_avg = month_resid_df.groupby(month_resid_df.index).mean()
monthly_std = month_resid_df.groupby(month_resid_df.index).std()

print("\nMLR FINAL TEST PERFORMANCE (10 RUNS)")
print(f"RMSE mean: {np.mean(rmse_arr):.3f} | RMSE sd: {np.std(rmse_arr):.3f}")
print(f"R² mean:   {np.mean(r2_arr):.3f} | R² sd:   {np.std(r2_arr):.3f}")
print(f"MAE mean:  {np.mean(mae_arr):.3f} | MAE sd:  {np.std(mae_arr):.3f}")
print("Monthly_mean:", monthly_avg)
print("Monthly_std:", monthly_std)

# # make model
# model = LinearRegression()
# model.fit(X_trainval, Y_trainval)
# # predict
# Y_pred = model.predict(X_test) 
# # model evaluation
# print("MLR Coefficients:", np.around(model.coef_, decimals = 3))
# print("MLR Intercept:", model.intercept_)
# print("MLR Root mean squared error (RMSE): %.2f" % math.sqrt(mean_squared_error(Y_test, Y_pred)))
# print("MLR training R² score: %.2f" % r2_score(Y_test, Y_pred), end="\n")

# # repeated runs to get average
# predictions = pd.DataFrame()
# rmses = []
# rmse_SD = []
# r2S = []
# r2_SD = []
# def mlr_monte_carlo(X, Y):
#     all_resid = []
#     month_resid = []
#     averages_arr = []
#     rmse_arr = []
#     R2_arr = []
#     for i in range(10):
#         X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i)
#         model = LinearRegression()
#         model.fit(X_train, Y_train)
#         Y_pred = model.predict(X_test)
#         # metrics
#         rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred))) #arr of each rmse in one monte carlo
#         R2_arr.append(r2_score(Y_test, Y_pred))  
#         # residuals
#         resid_arr = Y_test - Y_pred 
#         all_resid.append(resid_arr)
#         # monthly sum stuff
#         resid_arr.index = my_df.loc[Y_test.index, 'month'].values
#         month_resid.append(resid_arr)
#         averages = resid_arr.groupby(resid_arr.index).mean()
#         averages_arr.append(averages)
    
#     month_resid = pd.concat(month_resid) #flatten into dataframe
#     monthly_average = month_resid.groupby(month_resid.index).mean()
#     monthly_avg_df = pd.DataFrame(averages_arr)
#     monthly_std = monthly_avg_df.std()

#     all_resid = np.concatenate(all_resid) #flatten array 
#     predictions["Simulations"] = np.arange(1, 11) 
#     predictions["RMSE"] = np.around(rmse_arr, decimals = 3) #all rmses
#     predictions["R^2"] = np.around(R2_arr, decimals = 2)    #all r^2s
#     print("MLR Average RMSE", predictions['RMSE'].mean())
#     rmses.append(predictions['RMSE'].mean())
#     rmse_SD.append(predictions['RMSE'].std())
#     print("MLR Average R²", predictions['R^2'].mean())
#     r2S.append(round(predictions['R^2'].mean(), 2))
#     r2_SD.append(predictions['R^2'].std())
#     return all_resid, monthly_average, monthly_std
