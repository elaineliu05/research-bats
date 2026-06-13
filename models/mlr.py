import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from sklearn.preprocessing import StandardScaler

#importing df sets
df_a = pd.read_csv('df_sets/df_a.csv')
df_b = pd.read_csv('df_sets/df_b.csv')
df_c = pd.read_csv('df_sets/df_c.csv')
df_d = pd.read_csv('df_sets/df_d.csv')

datasets = {
    # "A": df_a,
    # "B": df_b,
    # "C": df_c,
    "D": df_d
}

for name, my_df in datasets.items():
    print(f"\nMLR on dataset {name}")

    X = my_df.drop(columns=['PP', 'yymmdd', 'day_of_year', 'year', 'month', 'day', 'set_depths']) # dec_year, sin_doy, cos_doy kept as time variables
    Y = my_df['PP']
    # splitting data (test set = last 15%)
    test_size = int(0.15 * len(my_df))

    X_trainval, X_test = X[:-test_size], X[-test_size:]
    Y_trainval, Y_test = Y[:-test_size], Y[-test_size:]

    # standardize for calculating coefficients
    scaler = StandardScaler()
    X_trainval = scaler.fit_transform(X_trainval)
    
    # 10 simulated runs (but deterministic)
    n_runs = 10
    rmse_arr, r2_arr, train_r2_arr, mae_arr = [], [], [], []
    month_resid = []
    test_preds = []

    for i in range(n_runs):
        # fit model
        mlr = LinearRegression()
        mlr.fit(X_trainval, Y_trainval)

        if i == 0:
            coef_df = pd.DataFrame({
                "Feature": X.columns,
                "Coefficient": mlr.coef_,
            })
            # magnitude for comparison
            coef_df["Abs_Importance"] = coef_df["Coefficient"].abs()
            # relative importance (%)
            coef_df["Relative (%)"] = (coef_df["Abs_Importance"] / coef_df["Abs_Importance"].sum()) * 100
            # sort
            coef_df = coef_df.sort_values(by="Relative (%)", ascending=False)
            # print results
            print("\n=== MLR Feature Importance (Dataset D) ===")
            print(coef_df.to_string(index=False))
            print(f"\nIntercept: {mlr.intercept_:.4f}")
            plt.figure(figsize=(10,6))
            plt.barh(coef_df["Feature"], coef_df["Relative (%)"], color = 'salmon')
            plt.gca().invert_yaxis()
            plt.xlabel("Relative % of Absolute Coefficient (Standardized)")
            plt.tight_layout()
            plt.savefig("mlr_coefficients.png", dpi=300)
            plt.close()
        
        # make predictions
        Y_pred_train = mlr.predict(X_trainval)
        Y_pred = mlr.predict(X_test)
        test_preds.append(Y_pred)
        
        # calculate metrics
        rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred)))
        r2_arr.append(r2_score(Y_test, Y_pred))
        train_r2_arr.append(r2_score(Y_trainval, Y_pred_train))
        mae_arr.append(mean_absolute_error(Y_test, Y_pred))

        # monthly residuals
        resid = pd.Series(Y_test - Y_pred, index=Y_test.index)
        resid.index = my_df.loc[resid.index, 'month'].values
        month_resid.append(resid)    
    # residual average over all simulations
    month_resid_df = pd.concat(month_resid)
    monthly_avg = month_resid_df.groupby(month_resid_df.index).mean()
    monthly_std = month_resid_df.groupby(month_resid_df.index).std()
    
    # dataframe w predictions
    print(test_preds)
    avg_test_pred = np.mean(test_preds, axis=0)

    # depth separated metrics --------------------------------------------------------
    depth_eval = pd.DataFrame({
        "Depth": my_df.loc[Y_test.index, "set_depths"],
        "True_PP": Y_test.values,
        "Pred_PP": avg_test_pred
    })
    depth_eval.to_csv('mlr_depth_eval.csv', index=False)
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
    #-----------------------------------------------------------------------------------------

    mlr_df_pred = my_df[['yymmdd', 'year', 'month', 'day', 'set_depths', 'PP']]
    mlr_df_pred.loc[Y_test.index, "MLR_Pred_PP"] = avg_test_pred
    mlr_df_pred.to_csv('preds/mlr_dfd_pred.csv', index=False) 

    print("test performance (10 runs)")
    print(f"RMSE mean: {np.mean(rmse_arr):.3f} | RMSE sd: {np.std(rmse_arr):.3f}")
    print(f"Train R² mean:   {np.mean(train_r2_arr):.3f} | Train R² sd:   {np.std(train_r2_arr):.3f}")
    print(f"R² mean:   {np.mean(r2_arr):.3f} | R² sd:   {np.std(r2_arr):.3f}")
    print(f"MAE mean:  {np.mean(mae_arr):.3f} | MAE sd:  {np.std(mae_arr):.3f}")
    print("Monthly_mean:", monthly_avg)
    print("Monthly_std:", monthly_std)