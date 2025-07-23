"""
Central Configuration File
Contains all project constants, file paths, and configuration settings.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Top 8 Indian cities for analysis (as per requirement)
TARGET_CITIES = [
    'Delhi', 'Mumbai', 'Bangalore', 'Hyderabad', 
    'Ahmedabad', 'Chennai', 'Kolkata', 'Pune'
]

# City coordinates (Top 8 cities only)
CITY_COORDINATES = {
    'Delhi': {'lat': 28.6139, 'lon': 77.2090, 'state': 'Delhi'},
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra'},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946, 'state': 'Karnataka'},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu'},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639, 'state': 'West Bengal'},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'state': 'Telangana'},
    'Pune': {'lat': 18.5204, 'lon': 73.8567, 'state': 'Maharashtra'},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'state': 'Gujarat'}
}

# AQI categories
AQI_CATEGORIES = {
    'Good': {'min': 0, 'max': 50, 'color': '#00E400'},
    'Satisfactory': {'min': 51, 'max': 100, 'color': '#FFFF00'},
    'Moderately Polluted': {'min': 101, 'max': 200, 'color': '#FF7E00'},
    'Poor': {'min': 201, 'max': 300, 'color': '#FF0000'},
    'Very Poor': {'min': 301, 'max': 400, 'color': '#8F3F97'},
    'Severe': {'min': 401, 'max': 500, 'color': '#7E0023'}
}

# Model configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'linear_regression': {},  # No params for LinearRegression
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    },
    'xgboost': {
        'n_estimators': 100,
        'random_state': 42,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
}

# Training configuration
TRAINING_CONFIG = {
    'cv_folds': 5,
    'scoring_metric': 'neg_mean_squared_error',
    'n_jobs': -1,
    'random_state': 42,
    'validation_split': 0.2
}

def get_aqi_category(aqi_value: float) -> str:
    """Get AQI category based on value."""
    for category, bounds in AQI_CATEGORIES.items():
        if bounds['min'] <= aqi_value <= bounds['max']:
            return category
    return 'Severe' if aqi_value > 400 else 'Good'

def get_model_path(model_type: str, model_name: str, version: str = None) -> Path:
    """Get model file path for saving/loading."""
    if version:
        return MODELS_DIR / f"{model_type}_{model_name}_{version}.joblib"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return MODELS_DIR / f"{model_type}_{model_name}_{timestamp}.joblib"