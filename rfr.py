from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split, GridSearchCV, BaseCrossValidator

# importing df sets
df_a = pd.read_csv('df_sets/df_a.csv')
df_b = pd.read_csv('df_sets/df_b.csv')
df_c = pd.read_csv('df_sets/df_c.csv')
df_d = pd.read_csv('df_sets/df_d.csv')
# reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# rolling time cross validation
class RollingTimeSeriesCV(BaseCrossValidator):
    def __init__(self, n_splits=5, val_size_ratio=0.176):
        self.n_splits = n_splits
        self.val_size_ratio = val_size_ratio

    def split(self, X, y=None, groups=None):
        n = len(X)
        val_size = int(self.val_size_ratio * n)

        split_points = np.linspace(
            int(n * 0.4),
            n - val_size,
            self.n_splits,
            dtype=int
        )

        for train_end in split_points:
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, train_end + val_size)
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

my_df = df_c
X = my_df.drop(columns=['PP', 'yymmdd', 'day_of_year'])
y = my_df['PP']

# splitting data, test set is last 15%
test_size = int(0.15 * len(my_df))
X_trainval, X_test = X[:-test_size], X[-test_size:]
y_trainval, y_test = y[:-test_size], y[-test_size:]

# make model
rf = RandomForestRegressor(random_state=SEED)
# gridsearch for hyperparameter tuning
param_grid = {
    'n_estimators': [300],
    'max_depth': [6, 8, 10],
    'max_features': ["sqrt", "log2"],
    'min_samples_split': [10, 20, 40],
    'min_samples_leaf': [5, 10, 20],
    'min_impurity_decrease': [0.0, 0.01]
}

# gridsearch with rolling cv
cv = RollingTimeSeriesCV(n_splits=5)
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='r2',
    cv=cv,
    verbose=1
)
# fit model
grid.fit(X_trainval, y_trainval)
params = grid.best_params_
print("Best RF params:", params)
print("Best Monte Carlo CV R²:", grid.best_score_)

# final evaludation on test set
n_runs = 10
rmse_arr, r2_arr, mae_arr = [], [], []
all_resid, month_resid = [], []

for i in range(n_runs):
    rf_final = RandomForestRegressor(
            **params,
            random_state=i
        )
    rf_final.fit(X_trainval, y_trainval)
    y_pred = rf_final.predict(X_test)
    # main metrics
    rmse_arr.append(math.sqrt(mean_squared_error(y_test, y_pred)))
    r2_arr.append(r2_score(y_test, y_pred))
    mae_arr.append(mean_absolute_error(y_test, y_pred))

    # residuals grouped by month
    resid = pd.Series(y_test - y_pred, index=y_test.index)
    resid.index = my_df.loc[resid.index, 'month'].values
    month_resid.append(resid)
    all_resid.append(resid.values)
# monthly residuals
month_resid_df = pd.concat(month_resid)
monthly_avg = month_resid_df.groupby(month_resid_df.index).mean()
monthly_std = month_resid_df.groupby(month_resid_df.index).std()
    
# summary stats
print("\n RFR FINAL TEST PERFORMANCE (10 RUNS)")
print(f"RMSE mean: {np.mean(rmse_arr):.3f} | RMSE sd: {np.std(rmse_arr):.3f}")
print(f"R² mean:   {np.mean(r2_arr):.3f} | R² sd:   {np.std(r2_arr):.3f}")
print(f"MAE mean:  {np.mean(mae_arr):.3f} | MAE sd:  {np.std(mae_arr):.3f}")
print('Monthly_mean: ', monthly_avg)
print('Monthly_std: ', monthly_std)


# def rf_monte_carlo_final(df, params, n_sim=5):
#     rmse_arr, r2_arr, mae_arr = [], [], []
#     all_resid, month_resid = [], []

#     X = df.drop(columns=['PP', 'day_of_year', 'yymmdd'])
#     Y = df['PP']
#     test_size = int(0.15 * len(df))

#     for i in range(n_sim):
#         X_trainval, X_test = X[:-test_size], X[-test_size:]
#         Y_trainval, Y_test = Y[:-test_size], Y[-test_size:]

#         X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.176, shuffle=False)

#         # Random Forest model
#         rf_mod = RandomForestRegressor(
#             **params,
#             random_state=i
#         )

#         rf_mod.fit(X_train, Y_train)
#         Y_pred = rf_mod.predict(X_test)

#         rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred)))
#         r2_arr.append(r2_score(Y_test, Y_pred))
#         mae_arr.append(mean_absolute_error(Y_test, Y_pred))

#         # residuals grouped by month
#         resid = pd.Series(Y_test - Y_pred, index=Y_test.index)
#         resid.index = df.loc[resid.index, 'month'].values
#         month_resid.append(resid)
#         all_resid.append(resid.values)

#     month_resid_df = pd.concat(month_resid)
#     monthly_avg = month_resid_df.groupby(month_resid_df.index).mean()
#     monthly_std = month_resid_df.groupby(month_resid_df.index).std()

#     summary = {
#         'RMSE_mean': np.round(np.mean(rmse_arr), 3),
#         'RMSE_sd': np.round(np.std(rmse_arr), 3),
#         'R2_mean': np.round(np.mean(r2_arr), 3),
#         'R2_sd': np.round(np.std(r2_arr), 3),
#         'MAE_mean': np.round(np.mean(mae_arr), 3),
#         'MAE_sd': np.round(np.std(mae_arr), 3),
#         'Monthly_mean': monthly_avg,
#         'Monthly_std': monthly_std
#     }

#     return summary

# rf_summary = rf_monte_carlo_final(my_df, grid.best_params_)
# print(rf_summary)
