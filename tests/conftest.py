"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_aqi_data():
    """Create sample AQI data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(
        start="2023-01-01 00:00:00",
        end="2023-01-07 23:00:00",
        freq="H"
    )
    
    cities = ["Delhi", "Mumbai", "Bangalore"]
    
    data = []
    for date in dates:
        for city in cities:
            data.append({
                "datetime": date,
                "city": city,
                "aqi": np.random.normal(150, 50),
                "pm25": np.random.normal(60, 20),
                "pm10": np.random.normal(100, 30),
                "temperature": np.random.normal(25, 10),
                "humidity": np.random.normal(65, 15),
                "pressure": np.random.normal(1013, 10),
                "wind_speed": np.random.normal(5, 2),
            })
    
    df = pd.DataFrame(data)
    # Ensure no negative values
    numeric_cols = ["aqi", "pm25", "pm10", "temperature", "humidity", "pressure", "wind_speed"]
    for col in numeric_cols:
        df[col] = df[col].clip(lower=0)
    
    return df


@pytest.fixture
def sample_data_with_missing():
    """Create sample data with missing values for testing imputation."""
    np.random.seed(42)
    
    data = {
        "datetime": pd.date_range("2023-01-01", periods=100, freq="H"),
        "city": np.random.choice(["Delhi", "Mumbai"], 100),
        "aqi": np.random.normal(150, 50, 100),
        "temperature": np.random.normal(25, 10, 100),
        "humidity": np.random.normal(65, 15, 100),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[missing_indices[:10], "aqi"] = np.nan
    df.loc[missing_indices[10:], "temperature"] = np.nan
    
    return df


@pytest.fixture
def config_data():
    """Sample configuration for testing."""
    return {
        "model_config": {
            "linear_regression": {"fit_intercept": True},
            "random_forest": {"n_estimators": 10, "max_depth": 5, "random_state": 42},
            "xgboost": {"n_estimators": 10, "max_depth": 3, "random_state": 42}
        },
        "training_config": {
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_state": 42
        }
    }