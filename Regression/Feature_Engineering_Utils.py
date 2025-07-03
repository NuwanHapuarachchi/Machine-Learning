# Feature Engineering and Preprocessing Module
# Comprehensive utilities for data preprocessing and feature engineering

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, f_regression, SelectKBest
import matplotlib.pyplot as plt

class FeatureEngineering:
    """
    Utilities for feature engineering, selection, and transformation
    """
    
    @staticmethod
    def scaling_comparison(X, y):
        """
        Compare different scaling methods and their impact on model performance
        """
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'PowerTransformer': PowerTransformer()
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        X_original = X.copy()
        
        for idx, (name, scaler) in enumerate(scalers.items()):
            X_scaled = scaler.fit_transform(X)
            
            axes[idx].hist(X_scaled.ravel(), bins=50, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{name}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def feature_selection(X, y, n_features=10):
        """
        Select top k features using multiple methods
        """
        methods = {
            'Mutual Information': SelectKBest(mutual_info_regression, k=min(n_features, X.shape[1])),
            'F-statistic': SelectKBest(f_regression, k=min(n_features, X.shape[1]))
        }
        
        results = {}
        for name, selector in methods.items():
            selector.fit(X, y)
            selected_features = selector.get_support()
            scores = selector.scores_
            results[name] = {
                'selected': np.where(selected_features)[0],
                'scores': scores
            }
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for idx, (name, data) in enumerate(results.items()):
            scores = data['scores']
            top_indices = np.argsort(scores)[-n_features:]
            
            axes[idx].barh(range(len(top_indices)), scores[top_indices])
            axes[idx].set_yticks(range(len(top_indices)))
            axes[idx].set_yticklabels([f'Feature {i}' for i in top_indices])
            axes[idx].set_xlabel('Score')
            axes[idx].set_title(f'{name} - Top {n_features} Features')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    @staticmethod
    def pca_analysis(X, n_components=None):
        """
        PCA analysis and explained variance visualization
        """
        if n_components is None:
            n_components = min(10, X.shape[1])
        
        pca = PCA(n_components=n_components)
        X_transformed = pca.fit_transform(X)
        
        explained_variance = pca.explained_variance_ratio_
        cumsum_variance = np.cumsum(explained_variance)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scree plot
        axes[0].bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        axes[0].plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', linewidth=2)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Scree Plot')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        axes[1].plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 'o-', linewidth=2)
        axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
        print(f"Number of components explaining 95% variance: {n_components_95}")
        
        return pca, X_transformed
