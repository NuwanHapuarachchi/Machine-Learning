# Model Interpretation and Explainability Module
# Tools for understanding and explaining regression models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class ModelInterpretability:
    """
    Tools for interpreting and explaining machine learning models
    """
    
    @staticmethod
    def permutation_importance(model, X, y, metric='r2'):
        """
        Calculate permutation importance for each feature
        
        Parameters:
        -----------
        model : fitted sklearn model
        X : array-like, features
        y : array-like, target
        metric : str, 'r2' or 'rmse'
        
        Returns:
        --------
        importances : array of feature importances
        """
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Baseline score
        y_pred = model.predict(X)
        if metric == 'r2':
            baseline_score = r2_score(y, y_pred)
        else:
            baseline_score = -np.sqrt(mean_squared_error(y, y_pred))
        
        importances = []
        
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            
            y_pred_permuted = model.predict(X_permuted)
            if metric == 'r2':
                permuted_score = r2_score(y, y_pred_permuted)
            else:
                permuted_score = -np.sqrt(mean_squared_error(y, y_pred_permuted))
            
            importance = baseline_score - permuted_score
            importances.append(importance)
        
        return np.array(importances)
    
    @staticmethod
    def plot_feature_importance(importances, feature_names=None, top_k=10):
        """
        Visualize feature importances
        """
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Get top k features
        indices = np.argsort(np.abs(importances))[-top_k:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importances, color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importances')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def partial_dependence_plot(model, X, feature_idx, feature_name=None):
        """
        Plot partial dependence of a feature
        """
        if feature_name is None:
            feature_name = f'Feature {feature_idx}'
        
        # Create grid of feature values
        feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 50)
        predictions = []
        
        X_mean = X.mean(axis=0)
        
        for value in feature_values:
            X_temp = np.tile(X_mean, (len(X), 1))
            X_temp[:, feature_idx] = value
            pred = model.predict(X_temp).mean()
            predictions.append(pred)
        
        plt.figure(figsize=(10, 6))
        plt.plot(feature_values, predictions, 'o-', linewidth=2, markersize=6)
        plt.xlabel(feature_name)
        plt.ylabel('Partial Dependence')
        plt.title(f'Partial Dependence Plot: {feature_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def coefficient_analysis(model, feature_names=None, top_k=10):
        """
        Analyze model coefficients (for linear models)
        """
        if not hasattr(model, 'coef_'):
            print("Model does not have coefficients (linear-based)")
            return
        
        coef = model.coef_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(coef))]
        
        # Sort by absolute value
        indices = np.argsort(np.abs(coef))[-top_k:]
        
        plt.figure(figsize=(10, 6))
        colors = ['red' if c < 0 else 'green' for c in coef[indices]]
        plt.barh(range(len(indices)), coef[indices], color=colors, alpha=0.7)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_k} Model Coefficients')
        plt.grid(True, alpha=0.3, axis='x')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def prediction_error_analysis(y_true, y_pred, residuals_threshold=2.0):
        """
        Analyze prediction errors and identify outliers
        """
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        
        outliers = np.abs(residuals) > residuals_threshold * residual_std
        n_outliers = np.sum(outliers)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_true[~outliers], y_pred[~outliers], alpha=0.6, label='Regular')
        axes[0].scatter(y_true[outliers], y_pred[outliers], color='red', s=100, marker='x', label='Outliers', linewidths=2)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0].plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Actual vs Predicted')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        axes[1].scatter(y_pred[~outliers], residuals[~outliers], alpha=0.6, label='Regular')
        axes[1].scatter(y_pred[outliers], residuals[outliers], color='red', s=100, marker='x', label='Outliers', linewidths=2)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=2)
        axes[1].axhline(y=residuals_threshold * residual_std, color='red', linestyle=':', alpha=0.5)
        axes[1].axhline(y=-residuals_threshold * residual_std, color='red', linestyle=':', alpha=0.5)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Number of outliers detected: {n_outliers} ({100*n_outliers/len(y_true):.1f}%)")
        
        return outliers
