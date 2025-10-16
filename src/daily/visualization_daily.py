"""
Visualization Utilities for Hanoi Temperature Forecasting

This module contains functions for creating visualizations and charts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('default')
sns.set_palette('husl')


def plot_temperature_trends(df: pd.DataFrame, datetime_col: str, temp_col: str, 
                           title: str = "Temperature Trends") -> plt.Figure:
    """
    Plot temperature trends over time.
    
    Args:
        df (pd.DataFrame): Data with datetime and temperature
        datetime_col (str): Name of datetime column
        temp_col (str): Name of temperature column
        title (str): Plot title
        
    Returns:
        plt.Figure: Generated figure
    """
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(df[datetime_col], df[temp_col], linewidth=0.8, alpha=0.8)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    logger.info("Temperature trends plot created")
    return fig


def plot_seasonal_patterns(df: pd.DataFrame, temp_col: str, datetime_col: str) -> plt.Figure:
    """
    Plot seasonal temperature patterns.
    
    Args:
        df (pd.DataFrame): Data with datetime and temperature
        temp_col (str): Name of temperature column
        datetime_col (str): Name of datetime column
        
    Returns:
        plt.Figure: Generated figure
    """
    df_copy = df.copy()
    df_copy['month'] = pd.to_datetime(df_copy[datetime_col]).dt.month
    df_copy['hour'] = pd.to_datetime(df_copy[datetime_col]).dt.hour
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Monthly averages
    monthly_avg = df_copy.groupby('month')[temp_col].mean()
    axes[0, 0].bar(monthly_avg.index, monthly_avg.values, color='skyblue')
    axes[0, 0].set_title('Average Temperature by Month', fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Hourly averages
    hourly_avg = df_copy.groupby('hour')[temp_col].mean()
    axes[0, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', color='orange')
    axes[0, 1].set_title('Average Temperature by Hour', fontweight='bold')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature distribution
    axes[1, 0].hist(df_copy[temp_col], bins=50, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Temperature Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Box plot by month
    monthly_data = [df_copy[df_copy['month'] == month][temp_col].values for month in range(1, 13)]
    axes[1, 1].boxplot(monthly_data, labels=range(1, 13))
    axes[1, 1].set_title('Temperature Distribution by Month', fontweight='bold')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.info("Seasonal patterns plot created")
    return fig


def plot_correlation_matrix(df: pd.DataFrame, columns: List[str] = None) -> plt.Figure:
    """
    Plot correlation matrix of features.
    
    Args:
        df (pd.DataFrame): Data
        columns (List[str]): Columns to include
        
    Returns:
        plt.Figure: Generated figure
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    correlation_matrix = df[columns].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, ax=ax, fmt='.2f')
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    logger.info("Correlation matrix plot created")
    return fig


def plot_model_comparison(results: Dict[str, Dict[str, float]], metric: str = 'rmse') -> plt.Figure:
    """
    Plot model performance comparison.
    
    Args:
        results (Dict): Model evaluation results
        metric (str): Metric to compare
        
    Returns:
        plt.Figure: Generated figure
    """
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(models, values, color='steelblue', alpha=0.7)
    ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel(f'{metric.upper()}', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    logger.info("Model comparison plot created")
    return fig


def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model") -> plt.Figure:
    """
    Plot predictions vs actual values.
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
        model_name (str): Name of the model
        
    Returns:
        plt.Figure: Generated figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    axes[0].scatter(y_true, y_pred, alpha=0.6, color='steelblue')
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Temperature (°C)', fontsize=12)
    axes[0].set_ylabel('Predicted Temperature (°C)', fontsize=12)
    axes[0].set_title(f'{model_name} - Predictions vs Actual', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual plot
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6, color='orange')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Temperature (°C)', fontsize=12)
    axes[1].set_ylabel('Residuals (°C)', fontsize=12)
    axes[1].set_title(f'{model_name} - Residual Plot', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    logger.info("Predictions vs actual plot created")
    return fig


def plot_feature_importance(model: Any, feature_names: List[str], 
                           model_name: str = "Model") -> plt.Figure:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): Names of features
        model_name (str): Name of the model
        
    Returns:
        plt.Figure: Generated figure
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return None
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Take top 20 features
    top_n = min(20, len(feature_names))
    indices = indices[:top_n]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(range(top_n), importances[indices], color='forestgreen', alpha=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'{model_name} - Feature Importance (Top {top_n})', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, importances[indices])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{importance:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    logger.info("Feature importance plot created")
    return fig


def create_interactive_temperature_plot(df: pd.DataFrame, datetime_col: str, 
                                      temp_col: str, title: str = "Interactive Temperature Plot"):
    """
    Create interactive temperature plot using Plotly.
    
    Args:
        df (pd.DataFrame): Data with datetime and temperature
        datetime_col (str): Name of datetime column
        temp_col (str): Name of temperature column
        title (str): Plot title
        
    Returns:
        plotly.graph_objects.Figure: Interactive plot
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[datetime_col],
        y=df[temp_col],
        mode='lines',
        name='Temperature',
        line=dict(color='steelblue', width=1),
        hovertemplate='<b>Date:</b> %{x}<br><b>Temperature:</b> %{y}°C<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        hovermode='x unified',
        showlegend=True,
        height=600
    )
    
    logger.info("Interactive temperature plot created")
    return fig


def create_prediction_dashboard(df_actual: pd.DataFrame, df_pred: pd.DataFrame,
                               datetime_col: str, temp_col: str, pred_col: str):
    """
    Create comprehensive prediction dashboard.
    
    Args:
        df_actual (pd.DataFrame): Actual data
        df_pred (pd.DataFrame): Prediction data
        datetime_col (str): Datetime column name
        temp_col (str): Actual temperature column
        pred_col (str): Predicted temperature column
        
    Returns:
        plotly.graph_objects.Figure: Dashboard figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature Forecast', 'Prediction Accuracy', 
                       'Error Distribution', 'Daily Average Comparison'),
        specs=[[{"colspan": 2}, None],
               [{}, {}]]
    )
    
    # Main forecast plot
    fig.add_trace(
        go.Scatter(x=df_actual[datetime_col], y=df_actual[temp_col],
                  mode='lines', name='Actual', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_pred[datetime_col], y=df_pred[pred_col],
                  mode='lines', name='Predicted', line=dict(color='red')),
        row=1, col=1
    )
    
    # Accuracy scatter plot
    merged_df = pd.merge(df_actual, df_pred, on=datetime_col, how='inner')
    fig.add_trace(
        go.Scatter(x=merged_df[temp_col], y=merged_df[pred_col],
                  mode='markers', name='Accuracy', marker=dict(color='green')),
        row=2, col=1
    )
    
    # Error distribution
    errors = merged_df[temp_col] - merged_df[pred_col]
    fig.add_trace(
        go.Histogram(x=errors, name='Error Distribution', marker=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Temperature Forecasting Dashboard")
    
    logger.info("Prediction dashboard created")
    return fig