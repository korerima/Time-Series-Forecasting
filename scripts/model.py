import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from preprocess import cleaning
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import tensorflow as tf
from pmdarima import auto_arima
from scipy.stats import norm

class Train:
    def __init__(self, data, column='Close'):
        """
        Initialize the model trainer.

        """
        self.df = data
        self.column = column
        self.scaler=None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train = None
        self.test = None
        self.last_sequence=None
        self.model = {}
        self.prediction = {}
        self.scaler=StandardScaler()
        self.scalerr=MinMaxScaler()

    def train_test_split_time_series(self, train_size=0.9):
        """Train test split and scaling """
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])  # Convert Date column to datetime
            self.df.set_index('Date', inplace=True)  # Set it as index
           
        else:
            raise ValueError("The dataset does not contain a 'Date' column.")

        # Ensure index is in correct format
        if not isinstance(self.df.index, pd.DatetimeIndex):
            self.df.index = pd.to_datetime(self.df.index)
        
       
        split_idx = int(len(self.df) * train_size)
        self.train, self.test = self.df[:split_idx], self.df[split_idx:]
        train_scaled = self.scalerr.fit_transform(self.train[[self.column]]).astype(np.float64)
        test_scaled = self.scalerr.transform(self.test[[self.column]]).astype(np.float64)
        self.train.loc[:, self.column] = train_scaled.flatten()
        self.test.loc[:, self.column] = test_scaled.flatten()
   


    def arima_model(self):
        """ ARIMA model using auto_arima"""

        # Fit ARIMA model with seasonal component
        model = auto_arima(self.train[self.column], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        self.model['ARIMA']=model
        print(model.summary())
    


    def sarim_model(self):
        """ SARIMA model using auto_arima"""
        
        model = auto_arima(self.train[self.column], seasonal=True, m=12, stepwise=True, trace=True)
       
        self.model['SARIMA']=model
        print(model.summary())

    def create_sequences(self,data, window=60):
        """ Creating sequence for LTSM"""
            
        
        data = self.scaler.fit_transform(data.values.reshape(-1,1))
        X, y = [], []
        for i in range(len(data)-window):
            X.append(data[i:i+window])
            y.append(data[i+window])
       
        return np.array(X), np.array(y)

    def ltsm(self):
        """LTSM model """

        data = self.df[self.column]
        #data.set_index('Date', inplace=True) " # Set as index

        # Prepare data

        X, y= self.create_sequences(data)
        self.X_train, self.X_test = X[:-60], X[-60:]
        self.y_train, self.y_test = y[:-60], y[-60:]
        model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')

        # Train
        history=model.fit(self.X_train, self.y_train, epochs=50, batch_size=32,
                        validation_split=0.2, verbose=0)
        self.model['LSTM'] = {'model': model, 'history': history}
        self.last_sequence = self.X_test[-1] 
 

    def evaluate(self,model_name):
        """Evaluating the three models developed above"""
        test=self.test[self.column]
        if model_name == 'SARIMA':
            #train, test = train_test_split_time_series(df)
            results=self.model['SARIMA']
            # Get the predicted mean values

            forecast = results.predict(n_periods=len(test))  
            # Add predictions to test data
            self.prediction['SARIMA']=forecast
            print(f'MAE:{mean_absolute_error(test,forecast)}')
            print(f'MSE:{mean_squared_error(test,forecast)}')
            print(f'MAPE:{np.mean(np.abs((test- forecast) / test)) * 100}')
            
        elif model_name == 'ARIMA':
            #train, test = train_test_split_time_series(df)
            model= self.model['ARIMA']
            forecast = model.predict(n_periods=len(test))  
            # Add predictions to test data
            self.prediction['ARIMA']=forecast

            print(f'MAE:{mean_absolute_error(test,forecast)}')
            print(f'MSE:{mean_squared_error(test,forecast)}')
            print(f'MAPE:{np.mean(np.abs((test- forecast) / test)) * 100}')
            
        elif model_name == 'LTSM':
            model = self.model['LSTM']['model'] 
            lstm_forecast = model.predict(self.X_test)
            y_test=self.y_test
            print(f'MAE:{mean_absolute_error(y_test,lstm_forecast)}')
            print(f'MSE:{mean_squared_error(y_test,lstm_forecast)}')
            print(f'MAPE:{np.mean(np.abs((y_test- lstm_forecast) / y_test)) * 100}')
            lstm_forecast = self.scaler.inverse_transform(lstm_forecast).flatten()
            self.prediction['LSTM']=lstm_forecast
            y_test=self.scaler.inverse_transform(self.y_test).flatten()
            self.y_test=y_test
 

    def plot(self):
        """Plotting the prediction of the models"""
        plt.plot(self.test[self.column], label='Actual', color='blue')
        plt.plot(self.test.index,self.prediction['ARIMA'], label='ARIMA', color='orange')
        plt.plot(self.test.index,self.prediction['SARIMA'], label='SARIMA', color='green')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.legend()
        plt.show()
        plt.plot(self.y_test, label='Actual')
        plt.plot(self.prediction['LSTM'], label='LSTM Forecast', linestyle='--')
        plt.title('TSLA Stock Price Forecast (LSTM)')
        plt.legend()
        plt.show()

   
    
    def forecast(self, months=12, output_file='forecast_results.csv', best_model=None, alpha=0.05):
        """
        Generate forecasts for 6 to 12 months into the future and save to CSV.
        """
        
        # Validate input months
        if months not in [6, 12]:
            raise ValueError("Months should be either 6 or 12.")
        
        # Ensure data is sorted in ascending order by date
        if self.df.index.is_monotonic_decreasing:
            self.df = self.df.sort_index(ascending=True)
           
        
        # Convert months to trading days (assuming 21 trading days per month)
        periods = 21 * months


        try:
            # Prepare forecast date range
            forecast_dates = pd.date_range(start=self.df.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
            forecast_data = pd.DataFrame(index=forecast_dates, columns=[best_model])

            # Check if the model exists in self.model
            model_data = self.model.get(best_model)
            if not model_data:
                raise ValueError(f"Model '{best_model}' is not trained or available.")

            # Forecast based on model type
            if best_model in ['ARIMA', 'SARIMA']:
                # Forecasting with ARIMA or SARIMA
                try:
                    forecast_values,confint = model_data.predict(n_periods=252, return_conf_int=True)
                    #print(forecast_values.shape)
                    # If forecast_values is not a Series, convert it
                    if not isinstance(forecast_values, pd.Series):
                        forecast_values = pd.Series(forecast_values, index=forecast_dates)
                     
                    # Assign forecasted values to the 'forecast' column in forecast_data
                    forecast_data['forecast'] = forecast_values.values
                    forecast_data['conf_lower'] = confint[:, 0]
                    forecast_data['conf_upper'] = confint[:, 1]
                    forecast_data[['forecast', 'conf_lower', 'conf_upper']] = self.scalerr.inverse_transform(forecast_data[['forecast', 'conf_lower', 'conf_upper']])
                    

                except Exception as e:
                   
                    raise ValueError("ARIMA/SARIMA forecasting failed")

            elif best_model == 'LSTM':
                predictions = []
                residuals = []
                current_sequence = self.last_sequence.copy()  # Use the last sequence as the starting point

                for _ in range(252):
                    # Predict the next value
                    next_pred = self.model['LSTM']['model'].predict(current_sequence[np.newaxis, :, :])[0][0]
                    predictions.append(next_pred)
                    actual_last_value = self.X_train[-1]  # Assuming train_data contains past actual values
                    residuals.append(actual_last_value - next_pred)

                    # Update the sequence with the predicted value
                    current_sequence = np.roll(current_sequence, -1)  # Shift the sequence to the left
                    current_sequence[-1] = next_pred  # Replace the last value with the predicted value

                # Convert predictions to DataFrame
                forecast_data['forecast'] = predictions

                # **Compute Confidence Intervals (95%)**
                std_dev = np.std(residuals)
                z_score = norm.ppf(0.975)  # 1.96 for 95% CI

                forecast_data['conf_lower'] = forecast_data['forecast'] - (z_score * std_dev)
                forecast_data['conf_upper'] = forecast_data['forecast'] + (z_score * std_dev)

                # Ensure it's a NumPy array before inverse transforming
                scaled_values = forecast_data[['forecast', 'conf_lower', 'conf_upper']].values  # Extract as NumPy array

                # Reshape to 2D array (n_samples, n_features) for inverse_transform
                scaled_values = scaled_values.reshape(-1, 3)  # Ensure 2D shape

                # Apply inverse transform
                forecast_data = self.scaler.inverse_transform(scaled_values)


                #print("Scaler min:", self.scaler.min_)

            # Save forecast results to CSV
            forecast_data.to_csv(output_file)
          
        except Exception as e:
         
            raise ValueError("Forecasting failed")

