# What's this?

In the field of time series, due to different sensor positions or other reasons, there are time differences between different sensors.

# How to use this?
1. Delete files in data and result.
2. Put your CSV files in the data folders. Each CSV file represents a time series, rows represent different times, and columns represent different variables.
3. Modify the calculation parameters in the main.py
4. run the main.py

# How to calculate?
1. Adding different lags for the target variable
2. Slicing data using sliding windows
3. Calculate Spearman correlation coefficient within each slice
4. Capture useful parts
5. Average the Spearman correlation coefficients obtained from each slice

# update
Added application of different correlation coefficients, existing correlation coefficients are calculated mainly by: ['pearsonr', 'spearmanr', 'kendalltau']
