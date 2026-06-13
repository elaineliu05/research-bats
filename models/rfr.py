from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
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

        split_points = np.linspace(int(n * 0.4), n - val_size, self.n_splits, dtype=int)

        for train_end in split_points:
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, train_end + val_size)
            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

datasets = {
    # "A": df_a,
    # "B": df_b,
    # "C": df_c,
    "D": df_d
}
for name, my_df in datasets.items():
    print(f"\nRFR on dataset {name}")

    X = my_df.drop(columns=['PP', 'yymmdd', 'day_of_year', 'year', 'month', 'day', 'set_depths'])
    y = my_df['PP']

    # splitting data, test set is last 15%
    test_size = int(0.15 * len(my_df))
    X_trainval, X_test = X[:-test_size], X[-test_size:]
    y_trainval, y_test = y[:-test_size], y[-test_size:]

    # make model
    rf = RandomForestRegressor(random_state=SEED)
    
    # gridsearch for hyperparameter tuning
    param_grid = {
        'n_estimators': [200],
        'max_depth': [4],
        'max_features': [0.5],
        'min_samples_split': [2],
        'min_samples_leaf': [3],
        'min_impurity_decrease': [0.0]
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
    
    # final evaludation on test set - varying seed
    n_runs = 10
    rmse_arr, r2_arr, mae_arr = [], [], []
    all_resid, month_resid = [], []
    train_r2_arr = []
    test_preds = []

    for i in range(n_runs):
        rf_final = RandomForestRegressor(**params, random_state=i)
        # fit and predict
        rf_final.fit(X_trainval, y_trainval)
        
        y_train_pred = rf_final.predict(X_trainval)
        y_pred = rf_final.predict(X_test)
        test_preds.append(y_pred)
        
        # main metrics
        rmse_arr.append(math.sqrt(mean_squared_error(y_test, y_pred)))
        r2_arr.append(r2_score(y_test, y_pred))
        train_r2_arr.append(r2_score(y_trainval, y_train_pred))
        mae_arr.append(mean_absolute_error(y_test, y_pred))

        # calculate resids
        resid = pd.Series(y_test - y_pred, index=y_test.index)
        resid.index = my_df.loc[resid.index, 'month'].values
        month_resid.append(resid)
        all_resid.append(resid.values)
    # group resids by month
    month_resid_df = pd.concat(month_resid)
    monthly_avg = month_resid_df.groupby(month_resid_df.index).mean()
    monthly_std = month_resid_df.groupby(month_resid_df.index).std()

    # dataframe w predictions
    avg_test_pred = np.mean(test_preds, axis=0)

    # DEPTH SEPARATED METRICS ---------------------------------------------------------
    depth_eval = pd.DataFrame({
        "Depth": my_df.loc[y_test.index, "set_depths"],
        "True_PP": y_test.values,
        "Pred_PP": avg_test_pred
    })
    for depth in [1, 100]:
        depth_df = depth_eval[depth_eval["Depth"] == depth]
        if len(depth_df) > 0:
            rmse_depth = np.sqrt(mean_squared_error(depth_df["True_PP"], depth_df["Pred_PP"]))
            r2_depth = r2_score(depth_df["True_PP"], depth_df["Pred_PP"])
            mae_depth = mean_absolute_error(depth_df["True_PP"],depth_df["Pred_PP"])
            print(f"\nDepth = {depth}")
            print(f"n = {len(depth_df)}")
            print(f"RMSE = {rmse_depth:.3f}")
            print(f"R²   = {r2_depth:.3f}")
            print(f"MAE  = {mae_depth:.3f}")
    #-----------------------------------------------------------------

    rfr_df_pred = my_df[['yymmdd', 'year', 'month', 'day', 'set_depths', 'PP']]
    rfr_df_pred.loc[y_test.index, "RFR_Pred_PP"] = avg_test_pred
    rfr_df_pred.to_csv('preds/rfr_dfd_pred.csv', index=False) 
        
    # summary stats
    print("test performance (10 runs)")
    print(f"RMSE mean: {np.mean(rmse_arr):.3f} | RMSE sd: {np.std(rmse_arr):.3f}")
    print(f"Train R² mean:   {np.mean(train_r2_arr):.3f} | R² sd:   {np.std(train_r2_arr):.3f}")
    print(f"R² mean:   {np.mean(r2_arr):.3f} | R² sd:   {np.std(r2_arr):.3f}")
    print(f"MAE mean:  {np.mean(mae_arr):.3f} | MAE sd:  {np.std(mae_arr):.3f}")
    print('Monthly_mean: ', monthly_avg)
    print('Monthly_std: ', monthly_std)
