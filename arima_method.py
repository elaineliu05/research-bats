import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

#Load dataset and drop NaNs
file_path = 'C:/Users/elain/OneDrive/Documents/Research - BATS/matched_data_closest.xlsx'
df = pd.read_excel(file_path)
df = df.apply(pd.to_numeric, errors='coerce').astype('float64')
df = df.dropna()
df['yymmdd'] = pd.to_datetime(df['yymmdd'], format='%Y%m%d')
df['day_of_year'] = df['yymmdd'].dt.dayofyear
df.to_csv('df.csv')
df = df.drop_duplicates(subset=['yymmdd', 'Depth'], keep='first') #drops 4 duplicates
nutrient_names = ["Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]

#Plotting date differences
df['date_diff'] = df['yymmdd'].diff().dt.days
plt.scatter(df['yymmdd'][1:], df['date_diff'][1:], linestyle='-', color='b', s = 10)
plt.xlabel('Date')
plt.ylabel('Days Between Dates')
plt.title('Time Intervals Between Consecutive Dates')
plt.show()

#adjust depth values
main_depths = [1, 20, 40, 60, 80, 100, 120, 140]
def adjust_depth(value):
    for main_depth in main_depths:
        if abs(value - main_depth) <= 10:
            return main_depth  # Round to the main depth
    return value  # Leave as is if not within range
df['Adjusted_Depth'] = df['Depth'].apply(adjust_depth)


#Make uniform time steps
df.set_index('yymmdd', inplace=True)
resampled_list = []
for depth in [1, 20, 40, 60, 80, 100, 120, 140]:
    depth_df = df[df["Adjusted_Depth"] == depth]
    resampled = depth_df.resample('MS').first()
    resampled["Adjusted_Depth"] = depth
    resampled["Depth"] = depth
    resampled_list.append(resampled)
resampled_df = pd.concat(resampled_list)
resampled_df.reset_index(inplace = True)
resampled_df = resampled_df.sort_values(by = ["yymmdd", "Adjusted_Depth"])
df = resampled_df

#fill in missing data, closest finds dates before and after, then gets average value at that depth
df['PP'] = df.groupby('Depth')['PP'].transform(lambda x: x.interpolate(method='linear')) #fill in PPs
for nutrient in nutrient_names: #fill in nutrients
    df[nutrient] = df.groupby('Depth')[nutrient].transform(lambda x: x.interpolate(method='linear'))
df.to_csv('resampled_df.csv')
df.reset_index(inplace=True)

#Testing stationarity
result = adfuller(df['PP'])
print(f"ADF Statistic: {result[0]}")
print(f"P-value: {result[1]}")
if result[1] < 0.05:
    print("The series is stationary.")
else:
    print("The series is non-stationary. Differencing may be needed.")

#Plotting ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_acf(df['PP'].dropna(), ax=axes[0])
plot_pacf(df['PP'].dropna(), ax=axes[1])
plt.show()

#ARIMA Model
y = df['PP']  # Target variable
pop_exog = df[['POP']]  # Exogenous variable (Particulate Organic Phosphate)

# Split data into training and test sets
y_train, y_test, X_train, X_test = train_test_split(y, pop_exog, test_size=0.2, shuffle=False)

# Fit the ARIMAX model with just one exogenous variable (POP)
# Choose an initial order (p, d, q) for the model based on analysis of your data
model = sm.tsa.SARIMAX(y_train, exog=X_train, order=(16, 1, 13))
model_fit = model.fit(disp=False)

# Make predictions on the test set
predictions = model_fit.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1, exog=X_test)

# Evaluate the predictions
r2 = r2_score(y_test, predictions)
print(f"R Squared: {r2}")
