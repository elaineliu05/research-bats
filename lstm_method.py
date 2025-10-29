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
val_size = int(0.15 * len(my_df))  # 70-85% is validation
test_size = len(my_df) - train_size - val_size  # last 15% is testing

# Split data BEFORE scaling
train_df = my_df[:train_size]
val_df = my_df[train_size:train_size + val_size]
test_df = my_df[train_size + val_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
test_scaled = scaler.transform(test_df)

X_train, y_train = create_sequences(train_scaled, seq_length)
X_val, y_val = create_sequences(val_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)
doy_test = test_df["day_of_year"].values[seq_length:]
print("X_train shape:", X_train.shape, "\nX_val shape:", X_val.shape, "\nX_test shape:", X_test.shape)


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
r2_scores = []
rmse_scores = []
for run in range(5):
    print(f"\n===== Run {run+1} =====")
    model = create_model()
    es = EarlyStopping(patience=6, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        epochs=50, batch_size=8, validation_data=(X_val, y_val), verbose=1, callbacks=[es])
    # predict
    y_pred = model.predict(X_test)
    # rescale
    y_pred_rescaled = inverse_transform_predictions(X_test, y_pred, scaler)
    y_test_rescaled = inverse_transform_predictions(X_test, y_test, scaler)
    
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    rmse = root_mean_squared_error(y_test_rescaled, y_pred_rescaled)
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    print(f"Run {run+1} R²: {r2:.4f}")
    print(f"Run {run+1} RMSE: {rmse:.4f}")
    
avg_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
avg_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
print(f"Average R² over 5 runs: {avg_r2:.4f} ± {std_r2:.4f}")
print(f"Average RMSE over 5 runs: {avg_rmse:.4f} ± {std_rmse:.4f}")

#loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

#Plot predictions 
fig, axs = plt.subplots(2, 1)
# Scatter plot for actual PP values
axs[0].scatter(doy_test, y_test_rescaled, color='lightskyblue', label='Actual PP', s=10)
axs[0].scatter(doy_test, y_pred_rescaled, color='salmon', label='Predicted PP', s=10)
neg_acc = np.sum(y_test_rescaled < 0) 
neg_pred = np.sum(y_pred_rescaled < 0)
print("Percentage of negative values:", neg_acc / len(y_test_rescaled) * 100)
print("Percentage of negative predictions:", neg_pred / len(y_pred_rescaled) * 100)
axs[0].set_xlabel('Day of Year')
axs[0].set_ylabel('Primary Productivity (mgC/m³/day)')
axs[0].legend(loc = 'upper right')
# Scatter plot for error
error = y_test_rescaled - y_pred_rescaled
axs[1].scatter(doy_test, error, color='darkslateblue', label='Error (Actual - Predicted)', s=10)
axs[1].set_xlabel('Day of Year')
axs[1].set_ylabel('Error (mgC/m³/day)')
axs[1].legend(loc = 'upper right')
plt.tight_layout()
plt.show()


#Monte Carlo simulation
predictions = pd.DataFrame()
xgb_rmses = []
xgb_rmse_sd = []
xgb_r2s = []
xgb_r2_sd = []
maes = []
mae_SD = []
def xgb_monte_carlo(X, Y):
    all_resid = []
    month_resid = []
    averages_arr = []
    rmse_arr = []
    R2_arr = []
    mae_arr = []
    for i in range(10):
        # train_size = int(0.8 * len(my_df)) 
        # X_train = X[:train_size]
        # Y_train = Y[:train_size]
        # X_test = X[train_size:]
        # Y_test = Y[train_size:]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=i) # split data
        xgb_mod = xgb.XGBRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 10, min_child_weight = 5, subsample = 0.6, colsample_bytree = 0.8, gamma = 0, random_state = i)
        xgb_mod.fit(X_train, Y_train)
        Y_pred = xgb_mod.predict(X_test)
        #metrics
        rmse_arr.append(math.sqrt(mean_squared_error(Y_test, Y_pred))) #arr of each rmse in one monte carlo
        R2_arr.append(r2_score(Y_test, Y_pred))  
        mae_arr.append(mean_absolute_error(Y_test, Y_pred))
        #residuals
        resid_arr = Y_test - Y_pred 
        all_resid.append(resid_arr)
        #monthly sum stuff
        resid_arr.index = my_df.loc[Y_test.index, 'month'].values
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
    predictions["RMSE"] = np.around(rmse_arr, decimals = 3)            #all rmses
    predictions["R^2"] = np.around(R2_arr, decimals = 2)               #all r^2s
    predictions["MAE"] = np.around(mae_arr, decimals = 3)
    print("Average RMSE", predictions['RMSE'].mean())
    xgb_rmses.append(round(predictions['RMSE'].mean(), 2))
    xgb_rmse_sd.append(predictions['RMSE'].std())
    print("Average R²", predictions['R^2'].mean())
    xgb_r2s.append(round(predictions['R^2'].mean(), 3))
    xgb_r2_sd.append(predictions['R^2'].std())
    print("MLR Average MAE", predictions['MAE'].mean())
    maes.append(predictions['MAE'].mean())
    mae_SD.append(predictions['MAE'].std())
    return all_resid, monthly_average, monthly_std


# regressor = KerasRegressor(model= create_model, verbose=0)

# param_grid = {
#     "model__units": [32, 64, 128],
#     "model__dense_units": [32, 64, 128],
#     "model__dropout_rate": [0.3, 0.5],
#     "model__optimizer": ["adam", "rmsprop"],
#     "model__learning_rate": [0.001, 0.0005],
#     "batch_size": [16, 32, 64],
#     "epochs": [30, 50]
# }

# grid = GridSearchCV(
#     estimator=regressor,
#     param_grid=param_grid,
#     cv=3,  # 3-fold cross-validation
#     scoring=make_scorer(r2_score),
#     verbose=2,
#     n_jobs=-1
# )

# es = EarlyStopping(patience=6, restore_best_weights=True)

# grid_result = grid.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     callbacks=[es]
# )

# print(f"Best params: {grid_result.best_params_}")
# print(f"Best R² score: {grid_result.best_score_:.4f}")

# #predict on test set with best model
# y_pred_scaled = grid_result.best_estimator_.predict(X_test)

# def inverse_transform_predictions(X_seq, y_scaled, scaler):
#     full = np.zeros((len(y_scaled), X_seq.shape[2]))
#     full[:, :-1] = X_seq[:, -1, :-1]  # last timestep features
#     full[:, -1] = y_scaled
#     y_rescaled = scaler.inverse_transform(full)[:, -1]
#     return y_rescaled

# y_pred_rescaled = inverse_transform_predictions(X_test, y_pred_scaled, scaler)
# y_test_rescaled = inverse_transform_predictions(X_test, y_test, scaler)

# r2 = r2_score(y_test_rescaled, y_pred_rescaled)
# rmse = root_mean_squared_error(y_test_rescaled, y_pred_rescaled)
# print(f"Test R²: {r2:.4f}, RMSE: {rmse:.4f}")