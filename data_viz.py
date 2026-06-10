from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from sklearn.model_selection import train_test_split, GridSearchCV, BaseCrossValidator
import seaborn as sns
from scipy.stats import norm
import matplotlib.patches as mpatches
from sklearn.tree import export_graphviz
import graphviz
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV

# importing df sets
df_a = pd.read_csv('df_sets/df_a.csv')
df_b = pd.read_csv('df_sets/df_b.csv')
df_c = pd.read_csv('df_sets/df_c.csv')
df_d = pd.read_csv('df_sets/df_d.csv')
# reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

sets = ["Set A", "Set B", "Set C", "Set D"]
colors = ["#F4A261", "#f6da43", "#46cdb4", "#285f94"]    

mlr_monthly_avg = pd.DataFrame({"":[0.247112, 1.363281, 1.355337, 0.117548, -0.293959, 0.018531, -0.062896, 1.144556, 0.859989, -0.059041, -0.202651, 0.044390]}, index=range(1, 13))
mlr_monthly_std = pd.DataFrame({"":[1.648082, 3.664253, 3.011842, 1.891613,1.512056, 1.251613, 1.068069, 1.292274, 1.711652, 1.842697, 1.295177, 2.076447]}, index=range(1, 13))
rfr_monthly_avg = pd.DataFrame({"":[0.282284, 1.292054, 1.061939, 0.332385, -0.203462, 0.081788, -0.169878, 0.719180, 0.443400, -0.605203, -0.270259, 0.087718]}, index=range(1, 13))
rfr_monthly_std = pd.DataFrame({"":[1.793809, 3.230510, 2.655339, 1.913534, 1.496532, 1.110822, 1.118953, 0.995320, 1.804560, 1.489286,1.095432, 2.415170]}, index=range(1, 13))
xgb_monthly_avg = pd.DataFrame({"":[0.364318, 1.421778, 1.201495, 0.446439, -0.159208, -0.011512, -0.200899, 0.797890, 0.594930, -0.393786, -0.045490, 0.267335]}, index=range(1, 13))
xgb_monthly_std = pd.DataFrame({"":[1.739079, 3.317956, 2.555674, 1.901718, 1.472168, 1.130896, 1.054532, 1.080790, 1.812549, 1.464128, 1.031624, 2.191296]}, index=range(1, 13))
lstm_monthly_avg = pd.DataFrame({"":[-0.126099, 0.387439, 2.044997, 0.602777, 0.035633, -0.411303, -0.078022, 0.831582, 0.439063, -0.497720, -0.152063, 0.378191]}, index=range(1, 13))
lstm_monthly_std = pd.DataFrame({"":[1.271468, 3.776398, 3.669217, 1.732650, 1.481222, 1.547194,  1.232499, 1.320648, 2.099762, 1.030679, 0.842553, 2.370105]}, index=range(1, 13))


# PAPER FIG 3 ---------------------
X = df_d.drop(columns = ['PP',  "yymmdd", 'day_of_year'])  # Predictors (Independent variables)
Y = df_d['PP'] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # split data
rfr = RandomForestRegressor(n_estimators=300, random_state=0, max_depth=3, min_samples_split=140, min_samples_leaf=70, oob_score=True)
rfr.fit(X_train, Y_train)
Y_pred = rfr.predict(X_test)
r2 = round(r2_score(Y_test, Y_pred), 4)
print(f'Simple tree R-squared: {r2}')
dot_data = export_graphviz(rfr.estimators_[0], out_file=None, feature_names=X.columns, filled=True, rounded=True, special_characters=True, impurity=False,  proportion=False, precision=2)

graph = graphviz.Source(dot_data)
graph.render("tree_visualization", format="png", cleanup=True)  # Saves as PNG

# FEATURE IMPORTANCES ----------------------------------------------------------------------------------------------
# rfr
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
my_df = df_d
X = my_df.drop(columns=['PP', 'yymmdd', 'day_of_year', 'year', 'month', 'day', 'set_depths'])
y = my_df['PP']
test_size = int(0.15 * len(my_df))
X_trainval, X_test = X[:-test_size], X[-test_size:]
y_trainval, y_test = y[:-test_size], y[-test_size:]
rf = RandomForestRegressor(random_state=SEED)
param_grid = {'n_estimators': [200],'max_depth': [4],'max_features': [0.5],'min_samples_split': [2],'min_samples_leaf': [3],'min_impurity_decrease': [0.0]
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
# feature importance on best model
rf_final = RandomForestRegressor(**params, random_state=SEED)
rf_final.fit(X_trainval, y_trainval)
importances = rf_final.feature_importances_
feature_names = X_trainval.columns
feat_df_rfr = pd.DataFrame({"feature": feature_names,"importance": importances})
# Convert to percentage
feat_df_rfr["importance"] = feat_df_rfr["importance"] * 100
feat_df_rfr = feat_df_rfr.sort_values("importance", ascending=True)
#---------------------------------------------   
import xgboost as xgb
import random
from xgboost import XGBRegressor
import matplotlib.patches as mpatches

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
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
my_df = df_d
X = my_df.drop(columns=['PP', "yymmdd", 'day_of_year', 'year', 'month', 'day', 'set_depths'])
Y = my_df['PP']
test_size = int(0.15 * len(my_df))
X_trainval, X_test = X[:-test_size], X[-test_size:]
Y_trainval, Y_test = Y[:-test_size], Y[-test_size:]
xgb_base = xgb.XGBRegressor(random_state=SEED, objective='reg:squarederror')
param_grid = {'learning_rate': [0.003], 'n_estimators': [1000],'max_depth': [3],'min_child_weight': [40],'subsample': [0.6],'colsample_bytree': [0.9],'reg_alpha': [0],'reg_lambda': [1],'gamma': [0.01]}
rolling_cv = RollingTimeCV(n_splits=5)
grid = GridSearchCV(estimator=xgb_base, param_grid=param_grid,scoring='r2',cv=rolling_cv,verbose=1)
grid.fit(X_trainval, Y_trainval)
params = grid.best_params_
print("Best XGB parameters:", params)
print("Best Rolling time CV R2:", grid.best_score_)

xgb_final = xgb.XGBRegressor(
    **params,
    objective='reg:squarederror',
    random_state=SEED
)
xgb_final.fit(X_trainval, Y_trainval, verbose=False)
# feature importances on best modle
booster = xgb_final.get_booster()
importance_dict = booster.get_score(importance_type="gain")
feat_df_xgb = pd.DataFrame({"feature": list(importance_dict.keys()), "importance": list(importance_dict.values())})
feat_df_xgb = feat_df_xgb.merge(pd.DataFrame({"feature": X_trainval.columns}), on="feature", how="right").fillna(0)
# Convert gain to percentage
total_gain = feat_df_xgb["importance"].sum()
feat_df_xgb["importance"] = (feat_df_xgb["importance"] / total_gain) * 100
feat_df_xgb = feat_df_xgb.sort_values("importance", ascending=True)
# -------------------------------------------------------------------------------------------
# calculate mlr importances 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

my_df = df_d
X = my_df.drop(columns=['PP', 'yymmdd', 'day_of_year', 'year', 'month', 'day', 'set_depths']) # dec_year, sin_doy, cos_doy kept as time variables
Y = my_df['PP']
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
mlr = LinearRegression()
mlr.fit(X_trainval, Y_trainval)

coef_df = pd.DataFrame({"Feature": X.columns,"Coefficient": mlr.coef_})
# magnitude for comparison
coef_df["Abs_Importance"] = coef_df["Coefficient"].abs()
# relative importance (%)
coef_df["Relative (%)"] = (coef_df["Abs_Importance"] / coef_df["Abs_Importance"].sum()) * 100
coef_df = coef_df.sort_values(by="Relative (%)", ascending=True)
    

# PAPER FIG 5 - feature importance comparison ------------------------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
# mlr
axes[0].barh(coef_df["Feature"], coef_df["Relative (%)"], color = 'salmon')
axes[0].set_xlabel("MLR Importance (Relative Coefficient %)")
axes[0].set_ylabel("Feature")
axes[0].grid(axis="x", linestyle="--", alpha=0.4)
# --- Random Forest ---
axes[1].barh(feat_df_rfr["feature"], feat_df_rfr["importance"], color="gold")
axes[1].set_xlabel("RFR Importance (Gini %)")
axes[1].set_ylabel("Feature")
axes[1].grid(axis="x", linestyle="--", alpha=0.4)
# --- XGBoost ---
axes[2].barh(feat_df_xgb["feature"], feat_df_xgb["importance"], color='yellowgreen')
axes[2].set_xlabel("XGB Importance (Gain %)")
axes[2].set_ylabel("Feature")
axes[2].set_xlim(0, 65)
axes[2].grid(axis="x", linestyle="--", alpha=0.4)
# Panel labels below x-axis labels
axes[0].text(-0.12, 1.02, '(a)', transform=axes[0].transAxes, fontsize=14, va='top')
axes[1].text(-0.12, 1.02, '(b)', transform=axes[1].transAxes, fontsize=14, va='top')
axes[2].text(-0.12, 1.02, '(c)', transform=axes[2].transAxes,fontsize=14, va='top')
plt.tight_layout()
plt.savefig("feat_importance_mlr_rfr_xgb.png", dpi=300)
plt.show()

# PAPER FIG 6 ------------------------------------------------------------------------------------------------------------------------------
train_test_split_date = datetime.datetime(2018, 4, 26)
# CALCULATING PP DATA OVER SURFACE AND SUBSURFACE
# train and test over depth = 1 and depth = 100
df_train = df_d[df_d['yymmdd'] < train_test_split_date]
df_test = df_d[df_d['yymmdd'] >= train_test_split_date]
df_train['month'] = df_train['yymmdd'].dt.month
df_test['month'] = df_test['yymmdd'].dt.month
# train with surface and deep monthly averages and stds
df_train_surface = df_train[df_train['set_depths'] == 1]
df_train_deep = df_train[df_train['set_depths'] == 100]
# test w surface and deep
df_test_surface = df_test[df_test['set_depths'] == 1]
df_test_deep = df_test[df_test['set_depths'] == 100]
# calculate monthly average and std for each train/test and surface/deep
train_surface_monthly_avg = df_train_surface.groupby('month')['PP'].mean()
train_deep_monthly_avg = df_train_deep.groupby('month')['PP'].mean()
test_surface_monthly_avg = df_test_surface.groupby('month')['PP'].mean()
test_deep_monthly_avg = df_test_deep.groupby('month')['PP'].mean()
train_surface_monthly_std = df_train_surface.groupby('month')['PP'].std()
train_deep_monthly_std = df_train_deep.groupby('month')['PP'].std()
test_surface_monthly_std = df_test_surface.groupby('month')['PP'].std()
test_deep_monthly_std = df_test_deep.groupby('month')['PP'].std()

# ALL RESIDS OVER SURFACE AND SUBSURFACE
mlr_pred = pd.read_csv('preds/mlr_dfd_pred.csv')
rfr_pred = pd.read_csv('preds/rfr_dfd_pred.csv')
xgb_pred = pd.read_csv('preds/xgb_dfd_pred.csv')
lstm_pred = pd.read_csv('preds/lstm_dfd_pred.csv')
all_df = mlr_pred.copy()
# merge others one by one
all_df = all_df.merge(
    xgb_pred[['yymmdd', 'set_depths', 'XGB_Pred_PP']],
    on=['yymmdd', 'set_depths'],
    how='left'
)
all_df = all_df.merge(
    lstm_pred[['yymmdd', 'set_depths', 'LSTM_Pred_PP']],
    on=['yymmdd', 'set_depths'],
    how='left'
)
all_df = all_df.merge(
    rfr_pred[['yymmdd', 'set_depths', 'RFR_Pred_PP']],
    on=['yymmdd', 'set_depths'],
    how='left'
)
all_df['MLR_resid']  = all_df['PP'] - all_df['MLR_Pred_PP']
all_df['XGB_resid']  = all_df['PP'] - all_df['XGB_Pred_PP']
all_df['LSTM_resid'] = all_df['PP'] - all_df['LSTM_Pred_PP']
all_df['RFR_resid']  = all_df['PP'] - all_df['RFR_Pred_PP']
depth1_df = all_df[all_df['set_depths'] == 1]
depth100_df = all_df[all_df['set_depths'] == 100]
# group by month (mean residuals)
depth1_monthly = depth1_df.groupby('month')[['MLR_resid','XGB_resid','LSTM_resid','RFR_resid']].mean()
depth100_monthly = depth100_df.groupby('month')[['MLR_resid','XGB_resid','LSTM_resid','RFR_resid']].mean()

# PAPER FIG S6
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(10, 10), sharex=True)
# top left
for col, color in zip(depth1_monthly.columns, colors):
    axes[0, 0].plot(depth1_monthly.index, depth1_monthly[col], marker='o', label=col, color=color)
axes[0, 0].set_title("Surface (Depth = 0)")
axes[0, 0].set_ylabel("Residual (True - Predicted)")
axes[0, 0].axhline(y=0, color='grey', linestyle='--')
axes[0, 0].set_ylim(-3.5, 13)
# top right
for col, color in zip(depth100_monthly.columns, colors):
    axes[0, 1].plot(depth100_monthly.index, depth100_monthly[col], marker='o', label=col, color=color)
axes[0, 1].set_title("Subsurface (Depth = 100)")
axes[0, 1].axhline(y=0, color='grey', linestyle='--')
axes[0, 1].legend(loc="upper right")
axes[0, 1].set_ylim(-3.5, 13)
# bottom left
axes[1, 0].plot(train_surface_monthly_avg.index, train_surface_monthly_avg.values, marker='o', color = 'orange')
axes[1, 0].errorbar(train_surface_monthly_avg.index, train_surface_monthly_avg.values, yerr=train_surface_monthly_std.values, fmt='o', color='orange', capsize=5)
axes[1, 0].plot(test_surface_monthly_avg.index, test_surface_monthly_avg.values, marker='o', color = "blueviolet")
axes[1, 0].errorbar(test_surface_monthly_avg.index, test_surface_monthly_avg.values, yerr=test_surface_monthly_std.values, fmt='o', color='blueviolet', capsize=5)
axes[1, 0].set_xlabel("Month"), axes[1, 0].set_ylabel("Average PP (mgC/m³/day)")
axes[1, 0].set_ylim(-3.5, 13)
# bottom right
axes[1, 1].plot(train_deep_monthly_avg.index, train_deep_monthly_avg.values, marker='o', color = 'orange')
axes[1, 1].errorbar(train_deep_monthly_avg.index, train_deep_monthly_avg.values, yerr=train_deep_monthly_std.values, fmt='o', color='orange', capsize=5)
axes[1, 1].plot(test_deep_monthly_avg.index, test_deep_monthly_avg.values, marker='o', color = "blueviolet")
axes[1, 1].errorbar(test_deep_monthly_avg.index, test_deep_monthly_avg.values, yerr=test_deep_monthly_std.values, fmt='o', color='blueviolet', capsize=5)
axes[1, 1].set_xlabel("Month")
axes[1, 1].set_ylim(-3.5, 13)
axes[1, 1].legend(["Train", "Test"], loc="upper right")
# Panel labels
axes[0, 0].text(-0.08, 1.05, '(a)', transform=axes[0, 0].transAxes, fontsize=14)
axes[0, 1].text(-0.08, 1.05, '(b)', transform=axes[0, 1].transAxes, fontsize=14)
axes[1, 0].text(-0.08, 1.05, '(c)', transform=axes[1, 0].transAxes, fontsize=14)
axes[1, 1].text(-0.08, 1.05, '(d)', transform=axes[1, 1].transAxes, fontsize=14)
plt.tight_layout()
plt.show()