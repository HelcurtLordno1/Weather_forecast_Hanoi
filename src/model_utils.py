"""
Model Utilities for Hanoi Temperature Forecasting

This module contains functions for training, evaluating, and deploying ML models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging
from typing import Dict, List, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemperatureForecastingModel:
    """
    Main class for temperature forecasting models.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_columns = None
        self.target_column = None
        
    def prepare_data(self, df: pd.DataFrame, target_column: str, feature_columns: List[str] = None, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            df (pd.DataFrame): Input data
            target_column (str): Name of target column
            feature_columns (List[str]): List of feature columns
            test_size (float): Size of test set
            random_state (int): Random state for reproducibility
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        logger.info(f"Data prepared: Train shape {X_train.shape}, Test shape {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """Train Linear Regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.models['linear_regression'] = model
        logger.info("Linear Regression model trained")
        return model
    
    def train_ridge_regression(self, X_train: np.ndarray, y_train: np.ndarray, alpha: float = 1.0) -> Ridge:
        """Train Ridge Regression model."""
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        self.models['ridge_regression'] = model
        logger.info("Ridge Regression model trained")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                           n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
        """Train Random Forest model."""
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        logger.info("Random Forest model trained")
        return model
    
    def train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray,
                               n_estimators: int = 100, random_state: int = 42) -> GradientBoostingRegressor:
        """Train Gradient Boosting model."""
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        self.models['gradient_boosting'] = model
        logger.info("Gradient Boosting model trained")
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all available models."""
        logger.info("Training all models...")
        
        self.train_linear_regression(X_train, y_train)
        self.train_ridge_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        self.train_gradient_boosting(X_train, y_train)
        
        logger.info(f"All {len(self.models)} models trained successfully")
        return self.models
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict: Evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models."""
        results = {}
        
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            results[model_name] = metrics
            logger.info(f"{model_name} - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict[str, float]], metric: str = 'rmse') -> str:
        """
        Select best model based on specified metric.
        
        Args:
            results: Evaluation results
            metric: Metric to use for selection
            
        Returns:
            str: Name of best model
        """
        if metric in ['mse', 'rmse', 'mae']:
            best_model_name = min(results.keys(), key=lambda k: results[k][metric])
        else:  # r2
            best_model_name = max(results.keys(), key=lambda k: results[k][metric])
        
        self.best_model = self.models[best_model_name]
        logger.info(f"Best model selected: {best_model_name}")
        return best_model_name
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray, 
                             model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for specified model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_type: Type of model to tune
            
        Returns:
            Dict: Best parameters
        """
        if model_type == 'random_forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_type}")
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.models[f'{model_type}_tuned'] = grid_search.best_estimator_
        
        logger.info(f"Hyperparameter tuning completed for {model_type}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_params_
    
    def save_model(self, model_name: str, file_path: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model_name: Name of model to save
            file_path: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.models[model_name], file_path)
        logger.info(f"Model {model_name} saved to {file_path}")
    
    def load_model(self, file_path: str, model_name: str = None) -> Any:
        """
        Load a model from disk.
        
        Args:
            file_path: Path to load the model from
            model_name: Name to assign to loaded model
            
        Returns:
            Loaded model
        """
        model = joblib.load(file_path)
        
        if model_name:
            self.models[model_name] = model
        
        logger.info(f"Model loaded from {file_path}")
        return model
    
    def predict(self, X: np.ndarray, model_name: str = None) -> np.ndarray:
        """
        Make predictions using specified model or best model.
        
        Args:
            X: Features for prediction
            model_name: Name of model to use
            
        Returns:
            Predictions
        """
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            model = self.models[model_name]
        elif self.best_model is not None:
            model = self.best_model
        else:
            raise ValueError("No model specified and no best model selected")
        
        predictions = model.predict(X)
        return predictions


def create_prediction_intervals(predictions: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create prediction intervals (simplified approach).
    
    Args:
        predictions: Model predictions
        confidence: Confidence level
        
    Returns:
        Tuple: (lower_bounds, upper_bounds)
    """
    # Simplified approach - in practice, you'd use model-specific methods
    std_error = np.std(predictions) * 0.1  # Simplified standard error
    z_score = 1.96 if confidence == 0.95 else 2.58  # For 95% or 99% confidence
    
    margin = z_score * std_error
    lower_bounds = predictions - margin
    upper_bounds = predictions + margin
    
    return lower_bounds, upper_bounds