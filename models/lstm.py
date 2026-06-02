import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras import Input
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import BaseCrossValidator
from tensorflow.keras.layers import GRU

# importing df sets
df_a = pd.read_csv('df_sets/df_a.csv')
df_b = pd.read_csv('df_sets/df_b.csv')
df_c = pd.read_csv('df_sets/df_c.csv')
df_d = pd.read_csv('df_sets/df_d.csv')

#rolling time cross validation
class RollingTimeCV(BaseCrossValidator):
    def __init__(self, n_splits=5, val_ratio=0.176): # 5 train/val splits. 0.85*0.176 is 0.15, so val is 15% of data. 
        self.n_splits = n_splits
        self.val_ratio = val_ratio

    # set indexes for train/val
    def split(self, X, y=None, groups=None):
        n = len(X)
        val_size = int(self.val_ratio * n)
        split_points = np.linspace(int(n * 0.4), n - val_size, self.n_splits, dtype=int) # start at 40% of data
        for train_end in split_points:
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, train_end + val_size)
            yield train_idx, val_idx
            
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# function to create lstm
def build_lstm(units=128, dense_units=32, dropout_rate=0.2, optimizer="adam"): 
    model = Sequential([
        Input(shape=(SEQ_LEN, X_trainval.shape[2])),
        LSTM(units, return_sequences=True), 
        LSTM(units // 2),
        Dense(dense_units, activation="relu"),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=optimizer, loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model
#---------------------------------------------------------------------------------------------

datasets = {
    # "A": df_a,
    # "B": df_b,
    # "C": df_c,
    "D": df_d
}
for name, my_df in datasets.items():
    print(f"\nLSTM on dataset {name}") 
    my_df = my_df.drop(columns=['day_of_year']) # other features kept as time variables (dec_year, sin_doy, cos_doy)
    
    # forecast split, last 15% for test
    test_size = int(0.15 * len(my_df))
    trainval_df = my_df[:-test_size]
    test_df = my_df[-test_size:]

    # drop year/month/day. data uses decimal_year, sin_dayofyear, cos_dayofyear instead
    trainval_df = trainval_df.drop(columns=['yymmdd', 'year', 'month', 'day', 'set_depths'])
    test_df = test_df.drop(columns=['yymmdd', 'year', 'month', 'day', 'set_depths'])
    
    # scaling - separate trainval and test to prevent data leakage
    scaler = MinMaxScaler()
    trainval_scaled = scaler.fit_transform(trainval_df)
    test_scaled = scaler.transform(test_df)
    
    #creating sequences
    SEQ_LEN = 90
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len, :-1]) # exclude last column, which is the target variable
            y.append(data[i+seq_len, -1])
        return np.array(X), np.array(y)

    # recreate data as sequences
    X_trainval, y_trainval = create_sequences(trainval_scaled, SEQ_LEN)
    X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

    #build inital model
    keras_model = KerasRegressor(model=build_lstm, verbose=0)
    param_grid = {
        "model__units": [128],
        "model__dense_units": [128],
        "model__dropout_rate": [0.3],
        "model__optimizer": ["adam"],
        "batch_size": [8],
        "epochs": [100]
    }

    # cross validation
    cv = RollingTimeCV(n_splits=5)
    best_score = -np.inf
    best_params = None
    # gridsearch 
    grid = GridSearchCV(
        estimator=keras_model,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        verbose=1,
        n_jobs=1  
    )

    # fit model
    grid.fit(X_trainval, y_trainval)

    cv_results = grid.cv_results_
    best_idx = grid.best_index_
    
    print("\nPer-split R² scores (best model):")
    for i in range(cv.get_n_splits()):
        split_score = cv_results[f"split{i}_test_score"][best_idx]
        print(f"  Split {i+1}: R² = {split_score:.3f}")

    # save best
    best_params = grid.best_params_
    best_score = grid.best_score_
    print("Best LSTM params:", best_params)
    print("Mean rolling CV R²:", round(best_score, 3))
    final_params = {
        "units": best_params["model__units"],
        "dense_units": best_params["model__dense_units"],
        "dropout_rate": best_params["model__dropout_rate"],
        "optimizer": best_params["model__optimizer"]
    }
    
    # final eval, 10 simulated runs to account for randomization
    n_runs = 10
    rmse_arr, r2_arr, mae_arr = [], [], []
    all_resid, month_resid = [], []
    train_r2_arr, val_r2_arr = [], []
    test_preds = []
    
    for i in range(n_runs):
        tf.keras.utils.set_random_seed(i) 
        #build model
        model = build_lstm(**final_params)
        # set up early stopping 
        es = EarlyStopping(patience=6, restore_best_weights=True)
        # fit model and save for plotting
        history = model.fit(
            X_trainval, y_trainval,
            validation_split=0.176,
            epochs=100, 
            batch_size=8,
            callbacks=[es],
            verbose=0
        )
        # plotting train/val loss (only first run)
        if i == 0:  
            train_rmse = history.history["rmse"]
            val_rmse = history.history["val_rmse"]
            epochs_range = range(1, len(train_rmse) + 1)
        
            plt.figure()
            plt.plot(epochs_range, train_rmse, label="Training RMSE")
            plt.plot(epochs_range, val_rmse, label="Validation RMSE")
            plt.xlabel("Epochs")
            plt.ylabel("RMSE")
            plt.title("LSTM Training vs Validation RMSE")
            plt.legend()
            plt.savefig("trainval_loss_plot.png", dpi=300)
            plt.close()

        # calculate trainval r2s (recreate train/val split used by keras)
        val_size = int(0.176 * len(X_trainval))
        X_train = X_trainval[:-val_size]
        y_train = y_trainval[:-val_size]
        X_val = X_trainval[-val_size:]
        y_val = y_trainval[-val_size:]
        # --- predictions ---
        y_train_pred = model.predict(X_train, verbose=0)
        y_val_pred = model.predict(X_val, verbose=0)
        # --- inverse scaling ---
        def inverse_transform_preds(y_scaled):
            full = np.zeros((len(y_scaled), scaler.n_features_in_))
            full[:, -1] = y_scaled[:, 0]
            return scaler.inverse_transform(full)[:, -1]
        y_train_pred_inv = inverse_transform_preds(y_train_pred)
        y_val_pred_inv = inverse_transform_preds(y_val_pred)
        # true values (inverse)
        full = np.zeros((len(y_train), scaler.n_features_in_))
        full[:, -1] = y_train
        y_train_true = scaler.inverse_transform(full)[:, -1]
        full = np.zeros((len(y_val), scaler.n_features_in_))
        full[:, -1] = y_val
        y_val_true = scaler.inverse_transform(full)[:, -1]
        # --- compute R² ---
        train_r2_arr.append(r2_score(y_train_true, y_train_pred_inv))
        val_r2_arr.append(r2_score(y_val_true, y_val_pred_inv))
        # make predictions 
        y_pred_scaled = model.predict(X_test, verbose=0)
        # inverse scaling
        full = np.zeros((len(y_pred_scaled), scaler.n_features_in_))
        full[:, -1] = y_pred_scaled[:, 0]
        y_pred = scaler.inverse_transform(full)[:, -1]
        full[:, -1] = y_test
        y_true = scaler.inverse_transform(full)[:, -1]
        test_preds.append(y_pred)
        
        # calculate metrics
        rmse_arr.append(math.sqrt(mean_squared_error(y_true, y_pred)))
        r2_arr.append(r2_score(y_true, y_pred))
        mae_arr.append(mean_absolute_error(y_true, y_pred))
        
        # calculate residuals
        resid_index = test_df.index[SEQ_LEN:]  # index corresponding to y_test / y_true after sequence creation
        resid = pd.Series( y_true - y_pred, index=resid_index)
        # map to months
        resid.index = my_df.loc[resid.index, "month"].values
        month_resid.append(resid)
        all_resid.append(resid.values)
    # group by month
    month_resid_df = pd.concat(month_resid)
    monthly_avg = month_resid_df.groupby(month_resid_df.index).mean()
    monthly_std = month_resid_df.groupby(month_resid_df.index).std()

    # dataframe w predictions
    pred_index = test_df.index[SEQ_LEN:]
    avg_test_pred = np.mean(test_preds, axis=0)
    
    lstm_df_pred = my_df[['yymmdd', 'year', 'month', 'day', 'set_depths', 'PP']].copy()
    lstm_df_pred.loc[pred_index, "LSTM_Pred_PP"] = avg_test_pred
    lstm_df_pred.to_csv('preds/lstm_dfd_pred.csv', index=False) 
    
    print("test performance (10 runs)")
    print(f"Train R² mean: {np.mean(train_r2_arr):.3f} | Train R² sd: {np.std(train_r2_arr):.3f}")
    print(f"Val   R² mean: {np.mean(val_r2_arr):.3f} | Val   R² sd: {np.std(val_r2_arr):.3f}")
    print(f"RMSE mean: {np.mean(rmse_arr):.3f} | RMSE sd: {np.std(rmse_arr):.3f}")
    print(f"R² mean:   {np.mean(r2_arr):.3f} | R² sd:   {np.std(r2_arr):.3f}")
    print(f"MAE mean:  {np.mean(mae_arr):.3f} | MAE sd:  {np.std(mae_arr):.3f}")
    print("Monthly residual mean:", monthly_avg)
    print("Monthly residual std:", monthly_std)
