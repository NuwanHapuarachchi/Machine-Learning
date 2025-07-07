# Time Series Regression Utilities
# Tools for time series analysis and forecasting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TimeSeriesRegression:
    """
    Utilities for time series regression and forecasting
    """
    
    @staticmethod
    def create_lagged_features(X, y, n_lags=5):
        """
        Create lagged features from time series data
        
        Parameters:
        -----------
        X : array-like, time series features
        y : array-like, time series target
        n_lags : int, number of lags to create
        
        Returns:
        --------
        X_lagged : array of lagged features
        y_filtered : array of target values corresponding to lagged features
        """
        X_lagged_list = []
        
        for i in range(n_lags):
            X_lagged_list.append(np.roll(X, i+1, axis=0))
        
        X_lagged = np.hstack(X_lagged_list)
        
        # Remove rows with nan values from rolling
        X_lagged = X_lagged[n_lags:, :]
        y_filtered = y[n_lags:]
        
        return X_lagged, y_filtered
    
    @staticmethod
    def plot_forecasting_results(y_true, y_pred, train_size=None, title='Forecast Results'):
        """
        Visualize forecasting results
        """
        plt.figure(figsize=(14, 6))
        
        plt.plot(y_true, label='Actual', linewidth=2, alpha=0.7)
        plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
        
        if train_size:
            plt.axvline(x=train_size, color='red', linestyle='--', linewidth=2, label='Train/Test Split')
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def rolling_window_validation(model, X, y, window_size=0.8, step_size=0.1):
        """
        Perform rolling window validation for time series
        
        Parameters:
        -----------
        model : sklearn estimator
        X : array-like, features
        y : array-like, target
        window_size : float, proportion of data for training window
        step_size : float, proportion to step forward each iteration
        
        Returns:
        --------
        scores : list of validation scores
        """
        n_samples = len(y)
        train_window = int(n_samples * window_size)
        step = int(n_samples * step_size)
        
        scores = []
        predictions = []
        
        for i in range(0, n_samples - train_window, step):
            X_train = X[i:i+train_window]
            y_train = y[i:i+train_window]
            
            X_test_start = i + train_window
            X_test_end = min(X_test_start + step, n_samples)
            
            X_test = X[X_test_start:X_test_end]
            y_test = y[X_test_start:X_test_end]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            scores.append(r2)
            predictions.extend(y_pred)
        
        print(f"Rolling Window Validation Results:")
        print(f"Mean R² Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        return scores, predictions
    
    @staticmethod
    def forecast_confidence_intervals(y_train, y_pred, confidence=0.95):
        """
        Calculate forecast confidence intervals
        """
        residuals = y_train - np.mean(y_train)
        residual_std = np.std(residuals)
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin_of_error = z_score * residual_std
        
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        
        return lower_bound, upper_bound
