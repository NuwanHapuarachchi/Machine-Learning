# Advanced Model Validation Techniques
# This module provides utilities for comprehensive model validation

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt

def plot_learning_curves(estimator, X, y, cv=5, n_jobs=-1):
    """
    Plot learning curves to diagnose bias-variance tradeoff
    
    Parameters:
    -----------
    estimator : sklearn estimator
    X : array-like training features
    y : array-like target values
    cv : int, cross-validation folds
    n_jobs : int, parallel jobs
    
    Returns:
    --------
    train_sizes : array of training set sizes
    train_scores : array of training scores
    test_scores : array of test scores
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation Score', linewidth=2)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    plt.xlabel('Training Set Size')
    plt.ylabel('Score (R²)')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return train_sizes, train_scores, test_scores

def analyze_residuals(y_true, y_pred):
    """
    Comprehensive residual analysis for regression models
    
    Parameters:
    -----------
    y_true : array-like actual values
    y_pred : array-like predicted values
    
    Returns:
    --------
    residuals : array of residuals
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scale-Location
    standardized_residuals = residuals / np.std(residuals)
    axes[1, 1].scatter(y_pred, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('√|Standardized Residuals|')
    axes[1, 1].set_title('Scale-Location Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return residuals
