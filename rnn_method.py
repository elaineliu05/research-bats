import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout # type: ignore
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
df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
print(df.isnull().sum())

print("Number of rows in original dataset:", len(df))
df_a = df
df_a = df_a.dropna() #drop NaNs
# df_a.to_csv('df_a.csv', index=False) 
print("Number of rows after dropping NaNs:", len(df_a))

df_b = df[["year", "month", "day", "sin_doy", "cos_doy", "Depth", "Chl", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "BAC", "PP"]]
df_b = df_b.dropna() #drop NaNs
df_c = df[["year", "month", "day", "sin_doy", "cos_doy", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "BAC", "PP"]]
df_c = df_c.dropna() #drop NaNs
df_d = df[["year", "month", "day", "sin_doy", "cos_doy", "Depth", "Temp", "O2", "NO3", "PO4", "PP"]]
df_d = df_d.dropna() #drop NaNs

#chlorophyll
df_a = df_a[df_a["Chl"] > -100] #drop weird chlorophylls
print('Number of rows after dropping weird chlorophylls:', len(df_a))
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
print("Num rows in set A after removing outliers:", len(df_a))
for i in range(4, df_b.shape[1]): df_b = chauvenets_criterion(df_b, df_b.columns[i])
print("Num rows in set B after removing outliers:", len(df_b))
for i in range(4, df_c.shape[1]): df_c = chauvenets_criterion(df_c, df_c.columns[i])
print("Num rows in set C after removing outliers:", len(df_c))
for i in range(4, df_d.shape[1]): df_d = chauvenets_criterion(df_d, df_d.columns[i])
print("Num rows in set D after removing outliers:", len(df_d))

#normalize data
scaler = MinMaxScaler()
dfc_scaled = scaler.fit_transform(df_c)

#sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # all features except PP
        y.append(data[i+seq_length, -1])     # target PP at t+1
    return np.array(X), np.array(y)

seq_length = 100  # how many time steps the model will look at
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
#model
model = Sequential()
model.add(SimpleRNN(units=128, activation='tanh', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.4))
model.add(SimpleRNN(units=64, activation='tanh'))
model.add(Dropout(0.4))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# train
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=20, batch_size=32)

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
y_pred_rescaled = pp_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_rescaled = pp_scaler.inverse_transform(y_test.reshape(-1, 1))

r2 = r2_score(y_test_rescaled, y_pred_rescaled)
rmse = root_mean_squared_error(y_test_rescaled, y_pred_rescaled)
print(f"R² Score (rescaled): {r2:.4f}")

#acc v pred
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual PP')
plt.plot(y_pred_rescaled, label='Predicted PP', linestyle='--')
plt.xlabel("Sample")
plt.ylabel("Primary Productivity")
plt.title("LSTM Predictions vs Actual")
plt.legend()
plt.show()