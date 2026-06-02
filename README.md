
# Comparing Machine Learning Models to Predict Ocean Primary Productivity in the Sargasso Sea: Insights from the Bermuda Atlantic Time-series Study

Primary productivity is an essential indicator of marine ecosystem health and ocean biophysical dynamics. The Bermuda Atlantic Time-series Study (BATS) monthly time series, starting in 1988, provides long-term records of seasonal, interannual, and decadal trends in primary productivity and biogeochemical variables. The objective of this work is to evaluate the ability of different machine learning models to predict depth-resolved primary productivity (mgC/m³/day) at BATS using environmental variables such as depth, temperature, nutrients, and chlorophyll. The study compares four different regression models - Multiple Linear Regression, Random Forest Regression, XGBoost, and a Long Short-Term Memory (LSTM) neural network. The results suggest that decision tree-based models, such as Random Forest and XGBoost, perform the best at capturing long-term changes in primary productivity. Specifically, XGBoost achieved the best predictive accuracy with an R² score of 0.57, a root mean square error of 1.86 mgC/m³/day, and a mean absolute error of 1.14 mgC/m³/day. The worst-performing model was the LSTM, with an R² score of 0.47 and root mean square error of 2.03 mgC/m³/day. The results contradicted the prior expectation that LSTM would outperform the other machine learning models because LSTMs are designed for time series applications. These findings may indicate that more complex neural networks such as LSTM require larger datasets than what is typically possible for in situ ocean measurements (for BATS ~ 3 × 10³ primary productivity observations), and are better suited to be used with remote sensing data. 



## To activate environment:

conda env create -f environment.yml

conda activate bats
