import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
import numpy as np
import seaborn as sns


def check_stationarity(df):
    result = adfuller(df['ys'], autolag='BIC')
    print("Results of Dickey-Fuller Test for SPY:")
    print(f'ADF Statistic: {result[0]}')
    print(f'n_lags: {result[1]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')
    print()
    result2 = adfuller(df['yg'], autolag='BIC')
    print("Results of Dickey-Fuller Test for GOOGL:")
    print(f'ADF Statistic: {result2[0]}')
    print(f'n_lags: {result2[1]}')
    print(f'p-value: {result2[1]}')
    for key, value in result2[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')


def detect_outliers(data):
    sns.boxplot(df['y'])
    plt.show()
    print(np.where(df['y'] > 0.17))
    print(np.where(df['y'] < -0.13))

    print(df['ds'][5], df['y'][5])
    print(df['ds'][86], df['y'][86])


# Read csv data with specific seperator, skip the first line and fix the data
df = pd.read_csv("C:\\Users\\mike\\Downloads\\ml_methods\\goog_spy.csv", sep=";", engine="python", skiprows=[1])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
# pd.set_option('display.max_columns', 10)
df.head()
# df = df[["Date", "YieldS", "YieldG"]] #Rollback to original data
df = df[["YieldS", "YieldG"]]
df['YieldS'] = df['YieldS'].str.rstrip('%').astype('float') / 100.0
df['YieldG'] = df['YieldG'].str.rstrip('%').astype('float') / 100.0
# df.columns = ['date', 'ys', 'yg'] #Rollback to original data
df.columns = ['ys', 'yg']
df.dropna(inplace=True)
# df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y") #Rollback to original data
# End of data fixing


# Remove outliers
df.drop(df[(df['ys'] > 0.17) | (df['ys'] < -0.13) | (df['yg'] > 0.11) | (df['yg'] < -0.1)].index, inplace=True)

# Split data into training and testing sets
x = df['yg'].values.reshape(-1, 1)
y = df['ys'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression().fit(x_train, y_train)
check_stationarity(df)
# print(model.coef_[0])
# print(model.intercept_)

# Plot the linear regression line
x_line = np.linspace(-0.15, 0.15, 50)
coef = float(model.coef_[0]).__round__(4)
intercept = float(model.intercept_).__round__(4)
y_line = model.coef_[0] * x_line + model.intercept_
print(coef, intercept)

# Plot the data
plt.scatter(df['yg'], df['ys'])
plt.annotate(f"y = {coef}x + {intercept}", xy=(0.2, 0.8), xycoords='axes fraction')
plt.plot(x_line, y_line)
plt.xlabel("GOOGL Yield (%)")
plt.ylabel("SPY Yield (%)")
plt.show()

# Evaluate the model on the test set
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

