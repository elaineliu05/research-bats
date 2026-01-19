import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import xgboost as xgb
import random
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.patches as mpatches

# importing df sets
df_a = pd.read_csv('df_sets/df_a.csv')
df_b = pd.read_csv('df_sets/df_b.csv')
df_c = pd.read_csv('df_sets/df_c.csv')
df_d = pd.read_csv('df_sets/df_d.csv')

class RollingTimeCV(BaseCrossValidator):
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
#---------------------------------------------   
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# Choose dataset
my_df = df_c

# Predictors & target
X = my_df.drop(columns=['PP', "yymmdd", 'day_of_year'])
Y = my_df['PP']
# Hold-out test split
test_size = int(0.15 * len(my_df))
X_trainval, X_test = X[:-test_size], X[-test_size:]
Y_trainval, Y_test = Y[:-test_size], Y[-test_size:]
# Base model for grid search
xgb_base = xgb.XGBRegressor(
    random_state=SEED,
    objective='reg:squarederror',
)
param_grid = {
    'learning_rate': [0.001, 0.003],
    'n_estimators': [800, 1200, 1600],
    'max_depth': [2, 3, 4],
    'min_child_weight': [7, 10],
    'subsample': [0.6],
    'colsample_bytree': [0.8, 0.9],
    'reg_alpha': [0],
    'reg_lambda': [1],
    'gamma': [0.05, 0.1]
}
# Monte Carlo CV: 5 seeds = 5 simulations
rolling_cv = RollingTimeCV(n_splits=5)
grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='r2',
    cv=rolling_cv,
    verbose=1
)
# fit model
grid.fit(X_trainval, Y_trainval)
params = grid.best_params_
print("Best XGB parameters:", params)
print("Best Rolling time CV R2:", grid.best_score_)

#final eval
n_runs = 10
rmse_arr, r2_arr, mae_arr = [], [], []
all_resid, month_resid = [], []

for i in range(n_runs):
    # validation split ONLY for early stopping
    # X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.176, shuffle=False)
    xgb_final = xgb.XGBRegressor(
        **params,
        objective='reg:squarederror',
        random_state=i
    )
    xgb_final.fit(X_trainval, Y_trainval, verbose=False)
    Y_pred = xgb_final.predict(X_test)

    rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred)))
    r2_arr.append(r2_score(Y_test, Y_pred))
    mae_arr.append(mean_absolute_error(Y_test, Y_pred))

    # Monthly residuals
    resid = pd.Series(Y_test - Y_pred, index=Y_test.index)
    resid.index = my_df.loc[resid.index, 'month'].values
    month_resid.append(resid)
    all_resid.append(resid.values)

month_resid_df = pd.concat(month_resid)
monthly_avg = month_resid_df.groupby(month_resid_df.index).mean()
monthly_std = month_resid_df.groupby(month_resid_df.index).std()

print("\nXGBOOST FINAL TEST PERFORMANCE (10 RUNS)")
print(f"RMSE mean: {np.mean(rmse_arr):.3f} | RMSE sd: {np.std(rmse_arr):.3f}")
print(f"R² mean:   {np.mean(r2_arr):.3f} | R² sd:   {np.std(r2_arr):.3f}")
print(f"MAE mean:  {np.mean(mae_arr):.3f} | MAE sd:  {np.std(mae_arr):.3f}")
print("Monthly_mean:", monthly_avg)
print("Monthly_std:", monthly_std)
# def xgb_rolling_final(df, params, n_sim=5):
#     rmse_arr, r2_arr, mae_arr = [], [], []
#     all_resid, month_resid = [], []

#     X = df.drop(columns=['PP', 'day_of_year', 'yymmdd'])
#     Y = df['PP']
#     test_size = int(0.15 * len(df))

#     X_trainval, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
#     Y_trainval, Y_test = Y.iloc[:-test_size], Y.iloc[-test_size:]

#     # Apply best hyperparameters
#     for i in range(n_sim):
#         # small validation set ONLY for early stopping
#         X_train, X_val, Y_train, Y_val = train_test_split(
#             X_trainval,
#             Y_trainval,
#             test_size=0.176,
#             random_state=i,
#             shuffle=False   # IMPORTANT: preserve time
#         )
#         xgb_mod = xgb.XGBRegressor(
#             **params,
#             objective='reg:squarederror',
#             random_state=i,
#             early_stopping_rounds=30
#         )
#         xgb_mod.fit(
#             X_train, Y_train,
#             eval_set=[(X_val, Y_val)],
#             verbose=False
#         )
#         Y_pred = xgb_mod.predict(X_test)

#         rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred)))
#         r2_arr.append(r2_score(Y_test, Y_pred))
#         mae_arr.append(mean_absolute_error(Y_test, Y_pred))

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
