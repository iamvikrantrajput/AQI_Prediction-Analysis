"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import AQIDataPreprocessor


class TestAQIDataPreprocessor:
    """Test AQIDataPreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = AQIDataPreprocessor()

    def test_init(self):
        """Test AQIDataPreprocessor initialization."""
        preprocessor = AQIDataPreprocessor()
        assert hasattr(preprocessor, 'raw_data_dir')
        assert hasattr(preprocessor, 'processed_data_dir')

    def test_standardize_kaggle_columns(self):
        """Test Kaggle column standardization."""
        # Create sample Kaggle-format data
        kaggle_data = pd.DataFrame({
            'City': ['Delhi', 'Mumbai'],
            'Date': ['2020-01-01', '2020-01-01'],
            'PM2.5': [50, 60],
            'PM10': [80, 90],
            'AQI': [100, 120],
            'NO2': [20, 25]
        })
        
        result = self.preprocessor._standardize_kaggle_columns(kaggle_data)
        
        # Check column renaming
        assert 'city' in result.columns
        assert 'pm25' in result.columns
        assert 'pm10' in result.columns
        assert 'aqi' in result.columns
        assert 'no2' in result.columns
        assert 'source' in result.columns
        assert result['source'].iloc[0] == 'Kaggle_Dataset'

    def test_standardize_kaggle_columns_with_datetime(self):
        """Test Kaggle column standardization with datetime creation."""
        kaggle_data = pd.DataFrame({
            'City': ['Delhi'],
            'Date': ['2020-01-01'],
            'PM2.5': [50],
            'AQI': [100]
        })
        
        result = self.preprocessor._standardize_kaggle_columns(kaggle_data)
        
        assert 'datetime' in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result['datetime'])

    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.preprocessor.clean_data(empty_df)
        assert result.empty

    def test_clean_data_with_valid_data(self):
        """Test cleaning with valid air quality data."""
        valid_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=5, freq='H'),
            'city': ['Delhi'] * 5,
            'pm25': [50, 60, 55, 45, 70],
            'pm10': [80, 90, 85, 75, 100],
            'aqi': [100, 120, 110, 90, 140],
            'source': ['Kaggle_Dataset'] * 5
        })
        
        result = self.preprocessor.clean_data(valid_data)
        
        assert not result.empty
        assert 'city' in result.columns
        assert 'datetime' in result.columns
        assert len(result) <= len(valid_data)  # May remove outliers

    def test_clean_data_outlier_removal(self):
        """Test outlier removal in cleaning."""
        data_with_outliers = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=10, freq='H'),
            'city': ['Delhi'] * 10,
            'pm25': [50, 60, 55, 45, 70, 5000, 65, 50, 55, 60],  # 5000 is outlier
            'aqi': [100, 120, 110, 90, 140, 999, 130, 100, 110, 120]
        })
        
        result = self.preprocessor.clean_data(data_with_outliers)
        
        # Outliers should be handled (replaced or removed)
        assert result['pm25'].max() < 1000

    def test_create_features_time_features(self):
        """Test time-based feature creation."""
        sample_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=24, freq='H'),
            'city': ['Delhi'] * 24,
            'aqi': range(100, 124),
            'pm25': range(50, 74)
        })
        
        result = self.preprocessor.create_features(sample_data)
        
        # Check time features
        expected_time_features = ['hour', 'day_of_week', 'month', 'season', 
                                'is_weekend', 'is_rush_hour', 'is_clean_hour']
        for feature in expected_time_features:
            assert feature in result.columns

    def test_create_features_location_features(self):
        """Test location-based feature creation."""
        sample_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=5, freq='H'),
            'city': ['Delhi', 'Mumbai', 'Delhi', 'Mumbai', 'Delhi'],
            'aqi': [100, 120, 110, 130, 105]
        })
        
        result = self.preprocessor.create_features(sample_data)
        
        # Check location features
        assert 'latitude' in result.columns
        assert 'longitude' in result.columns
        assert 'state' in result.columns
        
        # Check that coordinates are assigned correctly
        delhi_rows = result[result['city'] == 'Delhi']
        assert not delhi_rows['latitude'].isna().all()

    def test_create_features_lag_features(self):
        """Test lag feature creation."""
        # Create longer time series for lag features
        sample_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=50, freq='H'),
            'city': ['Delhi'] * 50,
            'aqi': range(100, 150)
        })
        
        result = self.preprocessor.create_features(sample_data)
        
        # Check lag features
        lag_features = ['aqi_lag_1h', 'aqi_lag_6h', 'aqi_lag_12h', 'aqi_lag_24h']
        for feature in lag_features:
            assert feature in result.columns

    def test_create_features_rolling_features(self):
        """Test rolling feature creation."""
        sample_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=50, freq='H'),
            'city': ['Delhi'] * 50,
            'aqi': range(100, 150)
        })
        
        result = self.preprocessor.create_features(sample_data)
        
        # Check rolling features
        rolling_features = ['aqi_rolling_6h', 'aqi_rolling_12h', 'aqi_rolling_24h']
        for feature in rolling_features:
            assert feature in result.columns

    def test_handle_missing_values(self):
        """Test missing value handling."""
        data_with_missing = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=10, freq='H'),
            'city': ['Delhi'] * 10,
            'pm25': [50, np.nan, 55, np.nan, 70, 65, np.nan, 50, 55, 60],
            'aqi': [100, np.nan, 110, np.nan, 140, 130, np.nan, 100, 110, 120]
        })
        
        result = self.preprocessor.handle_missing_values(data_with_missing)
        
        # Should have fewer missing values
        assert result['pm25'].isna().sum() <= data_with_missing['pm25'].isna().sum()
        assert result['aqi'].isna().sum() <= data_with_missing['aqi'].isna().sum()

    def test_pm25_to_aqi_conversion(self):
        """Test PM2.5 to AQI conversion."""
        # Test various PM2.5 values
        test_values = [10, 25, 45, 100, 200]
        for pm25_val in test_values:
            aqi_val = self.preprocessor._pm25_to_aqi(pm25_val)
            assert aqi_val > 0
            assert not pd.isna(aqi_val)
        
        # Test NaN handling
        assert pd.isna(self.preprocessor._pm25_to_aqi(np.nan))

    @patch('preprocessing.logger')
    def test_run_preprocessing_pipeline_success(self, mock_logger):
        """Test successful preprocessing pipeline run."""
        # Mock the load_data method to return sample data
        sample_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='H'),
            'city': ['Delhi'] * 50 + ['Mumbai'] * 50,
            'pm25': np.random.normal(60, 20, 100),
            'pm10': np.random.normal(90, 30, 100),
            'aqi': np.random.normal(120, 40, 100)
        })
        
        with patch.object(self.preprocessor, 'load_data', return_value=sample_data):
            with patch.object(self.preprocessor, 'save_processed_data'):
                result = self.preprocessor.run_preprocessing_pipeline()
                
                assert not result.empty
                assert len(result.columns) > len(sample_data.columns)  # Should have more features

    def test_run_preprocessing_pipeline_empty_data(self):
        """Test preprocessing pipeline with empty data."""
        with patch.object(self.preprocessor, 'load_data', return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No data available for processing"):
                self.preprocessor.run_preprocessing_pipeline()

    def test_season_mapping(self):
        """Test season mapping functionality."""
        sample_data = pd.DataFrame({
            'datetime': pd.to_datetime(['2020-01-15', '2020-04-15', '2020-07-15', '2020-10-15']),
            'city': ['Delhi'] * 4,
            'aqi': [100, 110, 120, 130],
            'month': [1, 4, 7, 10]  # Add month column that create_features expects
        })
        
        result = self.preprocessor.create_features(sample_data)
        
        seasons = result['season'].unique()
        expected_seasons = ['Winter', 'Spring', 'Monsoon', 'Autumn']
        for season in expected_seasons:
            assert season in seasons

    def test_rush_hour_detection(self):
        """Test rush hour detection."""
        # Create data with rush hour times
        rush_hours = [7, 8, 9, 17, 18, 19, 20]
        non_rush_hours = [2, 3, 14, 22]
        
        all_hours = rush_hours + non_rush_hours
        sample_data = pd.DataFrame({
            'datetime': [pd.Timestamp(f'2020-01-01 {hour:02d}:00:00') for hour in all_hours],
            'city': ['Delhi'] * len(all_hours),
            'aqi': [100] * len(all_hours)
        })
        
        result = self.preprocessor.create_features(sample_data)
        
        # Check rush hour detection
        rush_hour_rows = result[result['hour'].isin(rush_hours)]
        non_rush_hour_rows = result[result['hour'].isin(non_rush_hours)]
        
        assert rush_hour_rows['is_rush_hour'].all()
        assert not non_rush_hour_rows['is_rush_hour'].any()

    def test_city_filtering(self):
        """Test city filtering functionality."""
        sample_data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=10, freq='H'),
            'city': ['Delhi', 'Unknown_City', 'Mumbai', 'Random_City', 'Chennai'] * 2,
            'aqi': range(100, 110)
        })
        
        result = self.preprocessor.clean_data(sample_data)
        
        # Should only contain cities from TARGET_CITIES
        valid_cities = result['city'].unique()
        expected_cities = ['Delhi', 'Mumbai', 'Chennai']  # These should be in TARGET_CITIES
        
        for city in valid_cities:
            assert city in expected_cities


# Fixtures
@pytest.fixture
def sample_aqi_data():
    """Create sample AQI data for testing."""
    return pd.DataFrame({
        'datetime': pd.date_range('2020-01-01', periods=72, freq='H'),
        'city': ['Delhi'] * 24 + ['Mumbai'] * 24 + ['Chennai'] * 24,
        'aqi': np.random.normal(120, 30, 72),
        'pm25': np.random.normal(60, 15, 72),
        'pm10': np.random.normal(90, 20, 72),
        'no2': np.random.normal(25, 8, 72),
        'so2': np.random.normal(10, 5, 72),
        'co': np.random.normal(1.5, 0.5, 72),
        'o3': np.random.normal(35, 10, 72)
    })


@pytest.fixture
def sample_kaggle_data():
    """Create sample Kaggle-format data for testing."""
    return pd.DataFrame({
        'City': ['Delhi', 'Mumbai', 'Chennai'] * 10,
        'Date': ['2020-01-01'] * 30,
        'PM2.5': np.random.normal(60, 15, 30),
        'PM10': np.random.normal(90, 20, 30),
        'AQI': np.random.normal(120, 30, 30),
        'NO2': np.random.normal(25, 8, 30),
        'SO2': np.random.normal(10, 5, 30),
        'CO': np.random.normal(1.5, 0.5, 30),
        'O3': np.random.normal(35, 10, 30)
    })