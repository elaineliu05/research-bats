import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# importing df sets
df_a = pd.read_csv('C:/Users/elain/OneDrive/Documents/Research - BATS/df_sets/df_a.csv')
df_b = pd.read_csv('C:/Users/elain/OneDrive/Documents/Research - BATS/df_sets/df_b.csv')
df_c = pd.read_csv('C:/Users/elain/OneDrive/Documents/Research - BATS/df_sets/df_c.csv')
df_d = pd.read_csv('C:/Users/elain/OneDrive/Documents/Research - BATS/df_sets/df_d.csv')

# choose df set
my_df = df_c 
my_df["sin_doy"] = np.sin(2 * np.pi * my_df["day_of_year"] / 365)
my_df["cos_doy"] = np.cos(2 * np.pi * my_df["day_of_year"] / 365)

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
scaler = MinMaxScaler()
my_df_scaled = scaler.fit_transform(my_df)
doy_values = my_df["day_of_year"].values[seq_length:] 

#sequences for LSTM (need to better understand this)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # all features except PP
        y.append(data[i+seq_length, -1])     # target PP at t+1
    return np.array(X), np.array(y)


X, y = create_sequences(my_df_scaled, seq_length)
#split
train_size = int(0.7 * len(X)) # first 70% is training
val_size = int(0.15 * len(X))  # 70-85% is validation
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]
doy_test = doy_values[train_size + val_size:]
print("X_train shape:", X_train.shape, "\nX_val shape:", X_val.shape, "\nX_test shape:", X_test.shape)
#model
#ways to avoid overfitting. l2 regularization and dropout
#how long to train? make a graph of validation set and test set, with iteration on x axis and accuracy on y
# make plot of training loss on the left, val and train accuracy on the right
model = Sequential()
#change activation function?
model.add(LSTM(128, activation='relu', return_sequences = False, input_shape=(X.shape[1], X.shape[2]))) #first layer
# model.add(LSTM(64, activation='tanh', return_sequences = False)) #second layer
# model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(32, activation='relu'))  # Hidden layer
model.add(Dropout(0.4))  # Dropout layer
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')
model.summary()

#train
es = EarlyStopping(patience=6, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=25, batch_size=12,
                    validation_data=(X_val, y_val), verbose=1)

#loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

#predict
y_pred = model.predict(X_test)

#rescale
pp_scaler = MinMaxScaler()
pp_scaler.fit(df_c[['PP']])
y_pred_rescaled = (pp_scaler.inverse_transform(y_pred.reshape(-1, 1))).flatten()
y_test_rescaled = (pp_scaler.inverse_transform(y_test.reshape(-1, 1))).flatten()
r2 = r2_score(y_test_rescaled, y_pred_rescaled)
print(f"R² Score (rescaled): {r2:.4f}")

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

# # if want average of 5 runs (cuz of variability)
# for run in range(num_runs):
#     print(f"Run {run+1}/{num_runs}")
#     X, y = create_sequences(dfc_scaled, seq_length)
#     #split
#     train_size = int(0.7 * len(X))
#     val_size = int(0.15 * len(X))
#     X_train = X[:train_size]
#     y_train = y[:train_size]
#     X_val = X[train_size:train_size + val_size]
#     y_val = y[train_size:train_size + val_size]
#     X_test = X[train_size + val_size:]
#     y_test = y[train_size + val_size:]
#     doy_test = doy_values[train_size + val_size:]
#     print("X_train shape:", X_train.shape, "\nX_val shape:", X_val.shape, "\nX_test shape:", X_test.shape)
#     #model
#     model = Sequential()
#     model.add(LSTM(128, activation='tanh', return_sequences = True, input_shape=(X.shape[1], X.shape[2]))) #first layer
#     model.add(LSTM(64, activation='tanh', return_sequences = False)) #second layer
#     model.add(Dropout(0.2))  # Dropout layer
#     model.add(Dense(32, activation='tanh'))  # Hidden layer
#     model.add(Dropout(0.2))  # Dropout layer
#     model.add(Dense(1))  # Output layer
#     model.compile(optimizer='adam', loss='mse')
#     # model.summary()

#     #train
#     history = model.fit(X_train, y_train, epochs=25, batch_size=12,
#                     validation_data=(X_val, y_val), verbose=0)

#     #predict
#     y_pred = model.predict(X_test, verbose=0)

#     #rescale
#     pp_scaler = MinMaxScaler()
#     pp_scaler.fit(df_c[['PP']])
#     y_pred_rescaled = (pp_scaler.inverse_transform(y_pred.reshape(-1, 1))).flatten()
#     y_test_rescaled = (pp_scaler.inverse_transform(y_test.reshape(-1, 1))).flatten()
    
#     r2 = r2_score(y_test_rescaled, y_pred_rescaled)
#     r2_scores.append(r2)
#     rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
#     rmse_scores.append(rmse)

# print(f"Mean R²: {np.mean(r2_scores):.4f}")
# print(f"Std Dev of R²: {np.std(r2_scores):.4f}")
# print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")