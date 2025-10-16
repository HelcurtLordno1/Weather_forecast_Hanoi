"""
Shared Metrics Utilities for Weather Forecasting

Common evaluation metrics and functions used by both daily and hourly forecasting models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        'mse': mean_squared_error(y_true, y_pred)
    }
    
    # Additional custom metrics
    metrics['max_error'] = np.max(np.abs(y_true - y_pred))
    metrics['mean_residual'] = np.mean(y_true - y_pred)
    metrics['std_residual'] = np.std(y_true - y_pred)
    
    return metrics


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (how often the model predicts the correct direction of change).
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        float: Directional accuracy percentage
    """
    if len(y_true) < 2:
        return 0.0
        
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    correct_direction = (true_direction == pred_direction).sum()
    total_direction = len(true_direction)
    
    return (correct_direction / total_direction) * 100


def evaluate_model_performance(model, X_test: np.ndarray, y_test: np.ndarray, 
                             model_name: str = "Model") -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model with predict method
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        model_name (str): Name of the model
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    try:
        y_pred = model.predict(X_test)
        
        # Basic metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        
        # Directional accuracy
        metrics['directional_accuracy'] = calculate_directional_accuracy(y_test, y_pred)
        
        # Model-specific information
        evaluation = {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': y_pred,
            'residuals': y_test - y_pred,
            'test_size': len(y_test)
        }
        
        logger.info(f"{model_name} - RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, R²: {metrics['r2']:.3f}")
        
        return evaluation
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {str(e)}")
        return {'error': str(e)}


def compare_models(evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple model evaluations.
    
    Args:
        evaluations (Dict): Dictionary of model evaluations
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, eval_data in evaluations.items():
        if 'error' not in eval_data:
            metrics = eval_data['metrics']
            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MAPE': metrics['mape'],
                'Max Error': metrics['max_error'],
                'Directional Accuracy': metrics['directional_accuracy']
            })
    
    df = pd.DataFrame(comparison_data)
    if not df.empty:
        # Sort by RMSE (ascending - lower is better)
        df = df.sort_values('RMSE')
        df = df.round(4)
    
    return df


def print_model_summary(evaluation: Dict[str, Any]) -> None:
    """
    Print a formatted summary of model evaluation.
    
    Args:
        evaluation (Dict): Model evaluation results
    """
    if 'error' in evaluation:
        print(f"❌ Error in model evaluation: {evaluation['error']}")
        return
    
    model_name = evaluation['model_name']
    metrics = evaluation['metrics']
    
    print(f"\n🎯 {model_name} Performance Summary")
    print("=" * 50)
    print(f"📊 RMSE: {metrics['rmse']:.4f}")
    print(f"📊 MAE: {metrics['mae']:.4f}")
    print(f"📊 R²: {metrics['r2']:.4f}")
    print(f"📊 MAPE: {metrics['mape']:.2f}%")
    print(f"📊 Max Error: {metrics['max_error']:.4f}")
    print(f"📊 Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
    print(f"📊 Test Samples: {evaluation['test_size']}")
    print("=" * 50)


def format_metrics_for_display(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format metrics for display in UI applications.
    
    Args:
        metrics (Dict): Raw metrics dictionary
        
    Returns:
        Dict[str, str]: Formatted metrics for display
    """
    return {
        'RMSE': f"{metrics['rmse']:.3f}°C",
        'MAE': f"{metrics['mae']:.3f}°C", 
        'R²': f"{metrics['r2']:.3f}",
        'MAPE': f"{metrics['mape']:.1f}%",
        'Max Error': f"{metrics['max_error']:.2f}°C",
        'Directional Accuracy': f"{metrics['directional_accuracy']:.1f}%"
    }