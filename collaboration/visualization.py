import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

def create_temperature_chart(df):
    """Create temperature trend line chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["temperature"],
        name="Temperature",
        line=dict(color="#1E88E5", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["feels_like"],
        name="Feels Like",
        line=dict(color="#FFC107", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title="Temperature Trends",
        xaxis_title="Time",
        yaxis_title="Temperature (°F)",
        hovermode="x unified"
    )
    
    return fig

def create_precipitation_chart(df):
    """Create precipitation bar chart"""
    fig = px.bar(
        df,
        x="timestamp",
        y="precipitation",
        title="Precipitation Over Time",
        labels={"precipitation": "Precipitation (mm)", "timestamp": "Time"}
    )
    
    fig.update_traces(marker_color="#1E88E5")
    return fig

def create_wind_rose(df):
    """Create wind rose diagram"""
    wind_direction_bins = np.arange(0, 361, 45)
    wind_speed_bins = np.arange(0, df["wind_speed"].max() + 5, 5)
    
    fig = go.Figure()
    
    for i in range(len(wind_speed_bins)-1):
        mask = (df["wind_speed"] >= wind_speed_bins[i]) & (df["wind_speed"] < wind_speed_bins[i+1])
        
        fig.add_trace(go.Barpolar(
            r=df[mask].groupby(pd.cut(df[mask]["wind_deg"], wind_direction_bins), observed=True).size(),
            theta=wind_direction_bins[:-1],
            name=f'{wind_speed_bins[i]}-{wind_speed_bins[i+1]} mph',
            width=45
        ))
    
    fig.update_layout(
        title="Wind Rose Diagram",
        showlegend=True
    )
    
    return fig

def create_historical_comparison(df):
    """Create historical comparison chart"""
    daily_stats = df.set_index("timestamp").resample("D").agg({
        "temperature": "mean",
        "precipitation": "sum"
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_stats["timestamp"],
        y=daily_stats["temperature"],
        name="Temperature",
        yaxis="y1"
    ))
    
    fig.add_trace(go.Bar(
        x=daily_stats["timestamp"],
        y=daily_stats["precipitation"],
        name="Precipitation",
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Temperature and Precipitation Comparison",
        yaxis=dict(title="Temperature (°F)"),
        yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right"),
        hovermode="x unified"
    )
    
    return fig

def create_generic_scatter(df, x_col, y_col, title=None):
    """Create a generic scatter plot"""
    if title is None:
        title = f"{x_col} vs {y_col}"
        
    fig = px.scatter(df, x=x_col, y=y_col, title=title)
    return fig

def create_generic_histogram(df, column, title=None):
    """Create a generic histogram"""
    if title is None:
        title = f"Distribution of {column}"
        
    fig = px.histogram(df, x=column, title=title)
    return fig

def create_generic_box_plot(df, column, title=None):
    """Create a generic box plot"""
    if title is None:
        title = f"Box Plot of {column}"
        
    fig = px.box(df, y=column, title=title)
    return fig

def create_correlation_heatmap(df, numeric_cols=None, title="Correlation Matrix"):
    """Create a correlation heatmap"""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title=title
    )
    
    return fig
    
def create_generic_box_plot_categorical(df, numeric_column, categorical_column, title=None):
    """Create a box plot of a numeric variable grouped by a categorical variable"""
    if title is None:
        title = f"{numeric_column} by {categorical_column}"
        
    fig = px.box(
        df, 
        x=categorical_column, 
        y=numeric_column, 
        title=title,
        color=categorical_column
    )
    
    fig.update_layout(showlegend=False)
    return fig
