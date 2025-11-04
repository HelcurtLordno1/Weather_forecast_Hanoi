"""
Shared Plotting Utilities for Weather Forecasting

Common visualization functions used by both daily and hourly forecasting analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_time_series_plot(df: pd.DataFrame, date_col: str, value_col: str, 
                           title: str = "Time Series Plot", 
                           y_label: str = "Value") -> go.Figure:
    """
    Create an interactive time series plot using Plotly.
    
    Args:
        df (pd.DataFrame): Data containing time series
        date_col (str): Name of date column
        value_col (str): Name of value column
        title (str): Plot title
        y_label (str): Y-axis label
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[value_col],
        mode='lines',
        name=value_col,
        line=dict(width=1.5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        width=1000,
        height=500
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, title: str = "Feature Correlation Matrix") -> go.Figure:
    """
    Create an interactive correlation heatmap.
    
    Args:
        df (pd.DataFrame): Data for correlation analysis
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly heatmap figure
    """
    # Calculate correlation matrix
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        width=800,
        height=800,
        template='plotly_white'
    )
    
    return fig


def create_distribution_plot(df: pd.DataFrame, column: str, 
                            title: str = "Distribution Plot") -> go.Figure:
    """
    Create distribution plot with histogram and box plot.
    
    Args:
        df (pd.DataFrame): Data containing the column
        column (str): Column name to plot
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Histogram', 'Box Plot'),
        row_heights=[0.7, 0.3],
        vertical_spacing=0.1
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[column], nbinsx=50, name='Distribution'),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(x=df[column], name='Box Plot', orientation='h'),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig


def create_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray, 
                             title: str = "Predictions vs Actual") -> go.Figure:
    """
    Create scatter plot of predictions vs actual values.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly scatter plot
    """
    # Calculate R² for display
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=6, opacity=0.6),
        hovertemplate='Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>'
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"{title} (R² = {r2:.3f})",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        template='plotly_white',
        height=500,
        width=500
    )
    
    return fig


def create_residuals_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = "Residuals Plot") -> go.Figure:
    """
    Create residuals plot to assess model performance.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly residuals plot
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    
    # Residuals scatter plot
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(size=6, opacity=0.6),
        hovertemplate='Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>'
    ))
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", 
                  annotation_text="Zero residual line")
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        template='plotly_white',
        height=500,
        width=700
    )
    
    return fig


def create_feature_importance_plot(importance_dict: Dict[str, float], 
                                  title: str = "Feature Importance",
                                  top_n: int = 20) -> go.Figure:
    """
    Create horizontal bar plot for feature importance.
    
    Args:
        importance_dict (Dict): Dictionary of feature names and importance scores
        title (str): Plot title
        top_n (int): Number of top features to display
        
    Returns:
        go.Figure: Plotly bar plot
    """
    # Sort by importance and take top N
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importance_scores = zip(*sorted_features)
    
    fig = go.Figure(go.Bar(
        x=importance_scores,
        y=features,
        orientation='h',
        marker_color='skyblue',
        hovertemplate='Feature: %{y}<br>Importance: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        template='plotly_white',
        height=max(400, top_n * 25),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_model_comparison_plot(comparison_df: pd.DataFrame, 
                                metric: str = 'RMSE',
                                title: str = "Model Performance Comparison") -> go.Figure:
    """
    Create bar plot comparing model performance.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame with model comparison results
        metric (str): Metric to plot
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly bar plot
    """
    fig = go.Figure(go.Bar(
        x=comparison_df['Model'],
        y=comparison_df[metric],
        marker_color='lightcoral',
        hovertemplate='Model: %{x}<br>' + f'{metric}: %{{y:.4f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"{title} - {metric}",
        xaxis_title="Model",
        yaxis_title=metric,
        template='plotly_white',
        height=500
    )
    
    return fig


def create_seasonal_decomposition_plot(df: pd.DataFrame, date_col: str, value_col: str,
                                     title: str = "Seasonal Decomposition") -> go.Figure:
    """
    Create seasonal decomposition plot (simplified version).
    
    Args:
        df (pd.DataFrame): Time series data
        date_col (str): Date column name
        value_col (str): Value column name
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly subplot figure
    """
    # Simple seasonal decomposition using rolling averages
    df_temp = df.copy()
    df_temp['month'] = pd.to_datetime(df_temp[date_col]).dt.month
    df_temp['year'] = pd.to_datetime(df_temp[date_col]).dt.year
    
    # Calculate trend (12-period moving average)
    df_temp['trend'] = df_temp[value_col].rolling(window=12, center=True).mean()
    
    # Calculate seasonal component (monthly averages)
    monthly_avg = df_temp.groupby('month')[value_col].mean()
    df_temp['seasonal'] = df_temp['month'].map(monthly_avg)
    
    # Calculate residual
    df_temp['residual'] = df_temp[value_col] - df_temp['trend'] - df_temp['seasonal']
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.05
    )
    
    # Original data
    fig.add_trace(go.Scatter(x=df_temp[date_col], y=df_temp[value_col], 
                            mode='lines', name='Original'), row=1, col=1)
    
    # Trend
    fig.add_trace(go.Scatter(x=df_temp[date_col], y=df_temp['trend'], 
                            mode='lines', name='Trend'), row=2, col=1)
    
    # Seasonal
    fig.add_trace(go.Scatter(x=df_temp[date_col], y=df_temp['seasonal'], 
                            mode='lines', name='Seasonal'), row=3, col=1)
    
    # Residual
    fig.add_trace(go.Scatter(x=df_temp[date_col], y=df_temp['residual'], 
                            mode='lines', name='Residual'), row=4, col=1)
    
    fig.update_layout(
        title=title,
        height=800,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def save_plot(fig: go.Figure, filename: str, output_dir: str = "../outputs/") -> None:
    """
    Save Plotly figure to file.
    
    Args:
        fig (go.Figure): Plotly figure to save
        filename (str): Output filename (without extension)
        output_dir (str): Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as HTML (interactive)
    fig.write_html(f"{output_dir}/{filename}.html")
    
    # Save as PNG (static)
    try:
        fig.write_image(f"{output_dir}/{filename}.png", width=1200, height=800, scale=2)
    except Exception as e:
        print(f"Could not save PNG (kaleido required): {e}")