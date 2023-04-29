import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from scipy.stats import t
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Read csv data with specific seperator, skip the first line and fix the data
df = pd.read_csv("C:\\Users\\mike\\Downloads\\ml_methods\\goog_spy.csv", sep=";", engine="python", skiprows=[1])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.head()

# Extract the required columns and fix data issues
df = df[["Date", "YieldS"]]
df['YieldS'] = df['YieldS'].str.rstrip('%').astype('float') / 100.0
df.columns = ['date', 'ys']
df.dropna(inplace=True)
df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")  # Rollback to original data
df.set_index(['date'], inplace=True)
const = abs(df['ys'].min()) + 0.1
df['ys_pos'] = df['ys'] + const
df.index.freq = 'MS'


# End of data fixing


# Define function to find the best Exponential Smoothing model
def best_model():
    # Define parameter grid to search over
    global best_params
    trend_params = ['add', 'mul', None]
    seasonal_params = ['add', 'mul', None]
    seasonal_periods = [12]

    param_grid = list(itertools.product(trend_params, seasonal_params, seasonal_periods))

    # Evaluate models using each combination of parameters
    best_model = None
    best_mse = float('inf')

    for params in param_grid:
        trend, seasonal, seasonal_period = params

        model = ExponentialSmoothing(
            df['ys_pos'],
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_period,
            freq="MS"
        )
        fit = model.fit()

        # Compute mean squared error on validation set
        # mse = ((fit.predict(start='2022-01-01', end='2022-06-01') - df['ys'].loc[
        #                                                             '2022-01-01':'2022-06-01']) ** 2).mean()
        mse = ((fit.predict(start='2015-02-01', end='2022-12-01') - df['ys'].loc[
                                                                    '2015-02-01':'2022-12-01']) ** 2).mean()


        if mse < best_mse:
            best_mse = mse
            best_model = fit
            best_params = params

    # Print the best hyperparameters
    # Evaluate the model on the test set
    y_pred = fit.predict(start='2015-02-01', end='2022-12-01')
    mae = mean_absolute_error(df['ys_pos'], y_pred)
    mse = mean_squared_error(df['ys_pos'], y_pred)
    rmse = np.sqrt(mse)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("Best hyperparameters:", best_mse, best_model, best_params)
    return best_model


# Find the best Exponential Smoothing model using the defined function
fitted = best_model()

# Forecast future values
forecast = fitted.forecast(12)
forecast -= const
predictions = fitted.predict(start='2015-02-01', end='2022-12-01')
predictions = predictions - const

# Calculate the mean squared error for 95% confidence interval for forecasted values
n = len(forecast)
sd = np.std(forecast)
sef = sd / np.sqrt(n)
t_value = t.ppf(0.975, n - 1)
margin_of_error = t_value * sef
lower_bound_f = forecast - margin_of_error
upper_bound_f = forecast + margin_of_error

# Plot the results
plt.plot(df['ys'], label='Actual')
plt.plot(predictions, label='Predicted')
plt.plot(forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('SPY Yield (%)')
plt.fill_between(lower_bound_f.index, lower_bound_f, upper_bound_f, color='grey', alpha=0.3,
                 label='Confidence Interval forecasted')
plt.legend()
plt.show()
