import numpy as np
import pandas as pd
import seaborn as sns
import math
import keras
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error, make_scorer, mean_squared_error, mean_absolute_error
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import train_test_split


# importing df sets
df_a = pd.read_csv('C:/Users/elain/OneDrive/Documents/Research - BATS/df_sets/df_a.csv')
df_b = pd.read_csv('C:/Users/elain/OneDrive/Documents/Research - BATS/df_sets/df_b.csv')
df_c = pd.read_csv('C:/Users/elain/OneDrive/Documents/Research - BATS/df_sets/df_c.csv')
df_d = pd.read_csv('C:/Users/elain/OneDrive/Documents/Research - BATS/df_sets/df_d.csv')

# choose df set
my_df = df_a.copy()

my_df = my_df.drop('yymmdd', axis=1) #cant take in datetime
my_df.insert(0, "sin_doy", np.sin(2 * np.pi * my_df["day_of_year"] / 365))
my_df.insert(1, "cos_doy", np.cos(2 * np.pi * my_df["day_of_year"] / 365))

#look at seasonality 
from statsmodels.tsa.seasonal import seasonal_decompose
df_pp = my_df.dropna(subset=['PP'])
results = seasonal_decompose(df_pp['PP'], model='additive', period=365)
fig = results.plot()
for ax in fig.axes:
    for line in ax.get_lines():
        line.set_linewidth(0.8)  
        line.set_markersize(1.5) 
plt.show()

#plot histogram of PP
# plt.figure(figsize=(10, 5))
# plt.hist(my_df['PP'], bins=50, color='skyblue', edgecolor='black')
# plt.show()

#normalize data
seq_length = 60  # how many time steps (30-360)

#sequences for LSTM (need to better understand this)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # all features except PP
        y.append(data[i+seq_length, -1])     # target PP at t+1
    return np.array(X), np.array(y)

#get splits 
train_size = int(0.7 * len(my_df)) # first 70% is training
# val_size = int(0.15 * len(my_df))  # 70-85% is validation
test_size = len(my_df) - train_size  # last 15% is testing

# Split data BEFORE scaling
train_df = my_df[:train_size]
# val_df = my_df[train_size:train_size + val_size]
test_df = my_df[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
# val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

X_train, y_train = create_sequences(train_scaled, seq_length)
# X_val, y_val = create_sequences(val_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)
doy_test = test_df["day_of_year"].values[seq_length:]
month_test = test_df["month"].values[seq_length:]
print("month_test:", month_test)
# print("X_train shape:", X_train.shape, "\nX_val shape:", X_val.shape, "\nX_test shape:", X_test.shape)
print("y_test", y_test)

# model
def create_model(units = 64, dp_rate = 0.1, opt = 'adam', dense_units = 32):
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units, return_sequences = False)) #first layer
    model.add(Dense(dense_units, activation='relu'))  # Hidden layer
    model.add(Dropout(dp_rate))  # Dropout layer
    model.add(Dense(1))  # Output layer

    model.summary()
    model.compile(optimizer=opt, loss='mae', metrics = [keras.metrics.RootMeanSquaredError()])
    return model

#scale pp
def inverse_transform_predictions(X_seq, y_scaled, scaler):
    n_samples = len(y_scaled)
    n_features = scaler.n_features_in_  # total features scaler was fit on
    full = np.zeros((n_samples, n_features))
    full[:, :X_seq.shape[2]] = X_seq[:, -1, :]
    full[:, -1] = y_scaled.flatten()
    y_rescaled = scaler.inverse_transform(full)[:, -1]
    return y_rescaled

#train w monte carlo
predictions = pd.DataFrame()
lstm_rmses = []
lstm_rmse_sd = []
lstm_r2s = []
lstm_r2_sd = []
lstm_maes = []
lstm_mae_SD = []
def lstm_monte_carlo(X, Y):
    all_resid = []
    month_resid = []
    averages_arr = []
    rmse_arr = []
    R2_arr = []
    mae_arr = []
    for run in range(5):
        print(f"\n===== Run {run+1} =====")
        model = create_model()
        es = EarlyStopping(patience=6, restore_best_weights=True)
        history = model.fit(X_train, y_train, 
                            epochs=50, batch_size=8, verbose=1, callbacks=[es])
        # predict
        y_pred = model.predict(X_test)
        # rescale
        y_pred_rescaled = inverse_transform_predictions(X_test, y_pred, scaler)
        y_test_rescaled = inverse_transform_predictions(X_test, y_test, scaler)
        #metrics
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)
        rmse = root_mean_squared_error(y_test_rescaled, y_pred_rescaled)
        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        lstm_r2s.append(r2)
        lstm_rmses.append(rmse)
        R2_arr.append(r2)
        rmse_arr.append(rmse)
        mae_arr.append(mae)
        #residuals
        resid_arr = y_test_rescaled - y_pred_rescaled 
        all_resid.append(resid_arr)

        resid_series = pd.Series(resid_arr, index=month_test)
        month_resid.append(resid_series)
        averages = resid_series.groupby(resid_series.index).mean()
        averages_arr.append(averages) #append monthly averages to list
        print(f"Run {run+1} R²: {r2:.4f}")
        print(f"Run {run+1} RMSE: {rmse:.4f}")
        
    month_resid = pd.concat(month_resid) #flatten into dataframe
    monthly_average = month_resid.groupby(month_resid.index).mean()
    monthly_avg_df = pd.DataFrame(averages_arr)
    print('monthly average df', monthly_average)
    monthly_std = monthly_avg_df.std()
    print('monthly std', monthly_std)

    all_resid = np.concatenate(all_resid) #flatten array 
    predictions["Simulations"] = np.arange(1, 6) 
    predictions["RMSE"] = np.around(rmse_arr, decimals = 3)            #all rmses
    predictions["R^2"] = np.around(R2_arr, decimals = 2)               #all r^2s
    predictions["MAE"] = np.around(mae_arr, decimals = 3)
    print("Average RMSE", predictions['RMSE'].mean())
    lstm_rmses.append(round(predictions['RMSE'].mean(), 2))
    lstm_rmse_sd.append(predictions['RMSE'].std())
    print("Average R²", predictions['R^2'].mean())
    lstm_r2s.append(round(predictions['R^2'].mean(), 3))
    lstm_r2_sd.append(predictions['R^2'].std())
    print("MLR Average MAE", predictions['MAE'].mean())
    lstm_maes.append(predictions['MAE'].mean())
    lstm_mae_SD.append(predictions['MAE'].std())
    return all_resid, monthly_average, monthly_std

    # #loss
    # plt.plot(history.history['loss'], label='train loss')
    # plt.plot(history.history['val_loss'], label='val loss')
    # plt.legend()
    # plt.title("Training and Validation Loss")
    # plt.show()

    # #Plot predictions 
    # fig, axs = plt.subplots(2, 1)
    # # Scatter plot for actual PP values
    # axs[0].scatter(doy_test, y_test_rescaled, color='lightskyblue', label='Actual PP', s=10)
    # axs[0].scatter(doy_test, y_pred_rescaled, color='salmon', label='Predicted PP', s=10)
    # neg_acc = np.sum(y_test_rescaled < 0) 
    # neg_pred = np.sum(y_pred_rescaled < 0)
    # print("Percentage of negative values:", neg_acc / len(y_test_rescaled) * 100)
    # print("Percentage of negative predictions:", neg_pred / len(y_pred_rescaled) * 100)
    # axs[0].set_xlabel('Day of Year')
    # axs[0].set_ylabel('Primary Productivity (mgC/m³/day)')
    # axs[0].legend(loc = 'upper right')
    # # Scatter plot for error
    # error = y_test_rescaled - y_pred_rescaled
    # axs[1].scatter(doy_test, error, color='darkslateblue', label='Error (Actual - Predicted)', s=10)
    # axs[1].set_xlabel('Day of Year')
    # axs[1].set_ylabel('Error (mgC/m³/day)')
    # axs[1].legend(loc = 'upper right')
    # plt.tight_layout()
    # plt.show()

X_c = df_c.drop(columns = ['PP', 'yymmdd'])
Y_c = df_c['PP']
lstm_resid, lstm_monthly_avg, lstm_monthly_std = lstm_monte_carlo(X_c, Y_c)

print("lstm monthly average:", lstm_monthly_avg)
print("lstm monthly std:", lstm_monthly_std)