import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns

model1_metrics = []
model2_metrics = []


def detect_outliers(df):
    sns.boxplot(df['y'])
    plt.show()
    print(np.where(df['y'] > 0.17))
    print(np.where(df['y'] < -0.13))

    print(df['ds'][5], df['y'][5])
    print(df['ds'][86], df['y'][86])

def calculate_metrics(y_true, y_pred, model_type):
    # Calculate mean absolute error
    mae = mean_absolute_error(y_true, y_pred)
    # Calculate mean squared error
    mse = mean_squared_error(y_true, y_pred)
    # Calculate root mean squared error
    rmse = np.sqrt(mse)
    # Calculate mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # Print metrics
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("MAPE:", mape, "%")
    global model1_metrics
    global model2_metrics
    if model_type == "multiplicative":
        model1_metrics += [mae, mse, rmse, mape]
    else:
        model2_metrics += [mae, mse, rmse, mape]


def compare_models(mae1, mse1, rmse1, mape1, mae2, mse2, rmse2, mape2):
    metrics1 = [mae1, mse1, rmse1, mape1]
    metrics2 = [mae2, mse2, rmse2, mape2]

    # Compute the mean of each metric for both models
    mean_metrics1 = sum(metrics1) / len(metrics1)
    mean_metrics2 = sum(metrics2) / len(metrics2)

    # Check which model has the lowest mean of metrics
    if mean_metrics1 < mean_metrics2:
        return "Model 1 is better."
    elif mean_metrics1 > mean_metrics2:
        return "Model 2 is better."
    else:
        return "Both models are equally good."


def evaluate_prophet_model(training_set, testing_set, train_coefficient, model_type):
    # Train the model
    model = Prophet(seasonality_mode=model_type)
    model.fit(training_set)

    # Make predictions on the test set
    future = model.make_future_dataframe(periods=len(testing_set), freq='M')
    forecast = model.predict(future)
    predictions = forecast.iloc[train_coefficient:, -1].values
    upper = forecast.iloc[train_coefficient:, 2].values
    lower = forecast.iloc[train_coefficient:, 3].values

    # Plot the predictions, actual values and confidence intervals
    plt.figure(figsize=(16, 6))
    plt.plot(training_set['ds'], training_set['y'], color='blue', label='Train')
    plt.plot(testing_set['ds'], testing_set['y'], color='green', label='Test')
    plt.plot(testing_set['ds'], predictions, color='red', label='Predictions')
    plt.fill_between(testing_set['ds'], lower, upper, color='grey', alpha=0.3, label='Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('YieldS')
    plt.title(f'{model_type} Model Predictions')
    plt.legend()
    plt.show()

    calculate_metrics(testing_set['y'].values, predictions, model_type)


def plot_predictions(data, forecast):
    fig, ax = plt.subplots(figsize=(14, 7))

    # plot actual data
    ax.plot(data['ds'], data['y'], color='blue', label='Actual Yield')
    ax.plot(forecast.loc[forecast['ds'] >= data['ds'].max(), 'ds'],
            forecast.loc[forecast['ds'] >= data['ds'].max(), 'yhat'],
            color='green', label='Predicted Yield')

    ax.fill_between(forecast.loc[forecast['ds'] >= data['ds'].max(), 'ds'],
                    forecast.loc[forecast['ds'] >= data['ds'].max(), 'yhat_lower'],
                    forecast.loc[forecast['ds'] >= data['ds'].max(), 'yhat_upper'],
                    color='grey', alpha=0.5, label='Confidence Interval')
    # set plot properties
    ax.set_title('Yield Prediction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Yield')
    ax.legend()
    ax.grid(True)
    plt.show()


def predict_future_yield(data, periods=6):
    # Train the model on the input data
    model = Prophet(seasonality_mode='additive')
    model.fit(data)

    # Generate future dates to predict
    future_dates = model.make_future_dataframe(periods=periods, freq='M')

    # Make predictions for the future dates
    forecast = model.predict(future_dates)
    # Return the predicted values ([['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    return forecast


# Read csv data with specific seperator, skip the first line and fix the data
df = pd.read_csv("C:\\Users\\mike\\Downloads\\ml_methods\\goog_spy.csv", sep=";", engine="python", skiprows=[1])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
# pd.set_option('display.max_columns', 10)
df.head()
df = df[["Date", "YieldS"]]
df['YieldS'] = df['YieldS'].str.rstrip('%').astype('float') / 100.0
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'], format="%d/%m/%Y")
# End of data fixing

# Start of training/testing data separation
training_coefficient = int(0.8 * len(df))
print("training coefficient =", training_coefficient)  # Show coefficient (Number of obs used in training)
train = df.iloc[:training_coefficient]  # Training set
print("Training set:\n", train)
test = df.iloc[training_coefficient:]  # Testing set
print("Testing set:\n", test)
# End of separation

print("Model 1:")
evaluate_prophet_model(train, test, training_coefficient, 'multiplicative')
print("\n Model 2:")
evaluate_prophet_model(train, test, training_coefficient, 'additive')
print(compare_models(model1_metrics[0], model1_metrics[1], model1_metrics[2], model1_metrics[3], model2_metrics[0],
                     model2_metrics[1], model2_metrics[2], model2_metrics[3]))

plot_predictions(df, predict_future_yield(df, 6))
