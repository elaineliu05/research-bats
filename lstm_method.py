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


# Load dataset
file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/matched_data_from_BATS_trimmed.csv'
df = pd.read_csv(file_path)
df[df.columns[1:]] = df[df.columns[1:]].apply(pd.to_numeric, errors='coerce').astype('float64') #apply to everything except yymmdd
df['yymmdd'] = pd.to_datetime(df['yymmdd']) 
df['year'] = df['yymmdd'].dt.year
df['month'] = df['yymmdd'].dt.month
df['day'] = df['yymmdd'].dt.day
df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
# print(df.isnull().sum())

# print("Number of rows in original dataset:", len(df))
df_a = df
df_a = df_a.dropna() #drop NaNs
# df_a.to_csv('df_a.csv', index=False) 
# print("Number of rows after dropping NaNs:", len(df_a))

df_b = df[["year", "month", "day", "sin_doy", "cos_doy", "day_of_year", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "BAC", "PP"]]
df_b = df_b.dropna() #drop NaNs
df_c = df[["year", "month", "day", "sin_doy", "cos_doy", "day_of_year", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "BAC", "PP"]]
df_c = df_c.dropna() #drop NaNs
df_d = df[["year", "month", "day", "sin_doy", "cos_doy", "day_of_year", "Depth", "Temp", "O2", "NO3", "PO4", "PP"]]
df_d = df_d.dropna() #drop NaNs

#chlorophyll
df_a = df_a[df_a["Chl"] > -100] #drop weird chlorophylls
# print('Number of rows after dropping weird chlorophylls:', len(df_a))
df_b = df_b[df_b["Chl"] > -100] #drop NaNs

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

for i in range(4, df_a.shape[1]): df_a = chauvenets_criterion(df_a, df_a.columns[i])
# print("Num rows in set A after removing outliers:", len(df_a))
for i in range(4, df_b.shape[1]): df_b = chauvenets_criterion(df_b, df_b.columns[i])
# print("Num rows in set B after removing outliers:", len(df_b))
for i in range(4, df_c.shape[1]): df_c = chauvenets_criterion(df_c, df_c.columns[i])
# print("Num rows in set C after removing outliers:", len(df_c))
for i in range(4, df_d.shape[1]): df_d = chauvenets_criterion(df_d, df_d.columns[i])
# print("Num rows in set D after removing outliers:", len(df_d))

# #look at seasonality 
# from statsmodels.tsa.seasonal import seasonal_decompose
# df_pp = df = df.dropna(subset=['PP'])
# results = seasonal_decompose(df_pp['PP'], model='additive', period=365)
# fig = results.plot()
# for ax in fig.axes:
#     for line in ax.get_lines():
#         line.set_linewidth(0.8)  
#         line.set_markersize(1.5) 
# plt.show()

#plot histogram of PP
# plt.figure(figsize=(10, 5))
# plt.hist(df_c['PP'], bins=50, color='skyblue', edgecolor='black')
# plt.show()

#normalize data
scaler = MinMaxScaler()
dfc_scaled = scaler.fit_transform(df_c)
doy_values = df_c["day_of_year"].values[60:] 

#sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # all features except PP
        y.append(data[i+seq_length, -1])     # target PP at t+1
    return np.array(X), np.array(y)

seq_length = 60  # how many time steps the model will look at

X, y = create_sequences(dfc_scaled, seq_length)
#split
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))
X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]
doy_test = doy_values[train_size + val_size:]
print("X_train shape:", X_train.shape, "\nX_val shape:", X_val.shape, "\nX_test shape:", X_test.shape)
#model
model = Sequential()
model.add(LSTM(128, activation='tanh', return_sequences = False, input_shape=(X.shape[1], X.shape[2]))) #first layer
# model.add(LSTM(64, activation='tanh', return_sequences = False)) #second layer
# model.add(Dropout(0.2))  # Dropout layer
model.add(Dense(32, activation='tanh'))  # Hidden layer
model.add(Dropout(0.4))  # Dropout layer
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')
model.summary()

#train
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