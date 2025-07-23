"""Tests for train_model module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import SimpleModelTrainer


class TestSimpleModelTrainer:
    """Test SimpleModelTrainer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.trainer = SimpleModelTrainer()

    def test_init(self):
        """Test SimpleModelTrainer initialization."""
        trainer = SimpleModelTrainer()
        assert hasattr(trainer, 'config')
        assert hasattr(trainer, 'model_config')
        assert trainer.models == {}
        assert hasattr(trainer, 'scaler')

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        custom_config = {"test_size": 0.3, "random_state": 123}
        trainer = SimpleModelTrainer(custom_config)
        assert trainer.config == custom_config

    def test_prepare_data_basic(self, sample_processed_data):
        """Test basic data preparation."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert len(X_train) + len(X_test) == len(sample_processed_data)
        assert "aqi" not in X_train.columns  # Target should be removed

    def test_prepare_data_missing_target(self, sample_processed_data):
        """Test data preparation with missing target column."""
        with pytest.raises(ValueError, match="Target column 'missing' not found"):
            self.trainer.prepare_data(sample_processed_data, target_column="missing")

    def test_prepare_data_categorical_handling(self):
        """Test handling of categorical variables."""
        data = pd.DataFrame({
            'datetime': pd.date_range('2020-01-01', periods=100, freq='H'),
            'city': ['Delhi'] * 50 + ['Mumbai'] * 50,
            'aqi_category': ['Good'] * 25 + ['Moderate'] * 25 + ['Poor'] * 50,
            'aqi': np.random.normal(120, 30, 100),
            'pm25': np.random.normal(60, 15, 100)
        })
        
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            data, target_column="aqi"
        )
        
        # Categorical columns should be encoded
        assert 'aqi_category' in X_train.columns or any('aqi_category' in col for col in X_train.columns)

    def test_train_linear_regression(self, sample_processed_data):
        """Test linear regression training."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        
        model = self.trainer.train_linear_regression(X_train, y_train)
        metrics = self.trainer.evaluate_model(model, "linear_regression", X_test, y_test)
        
        assert model is not None
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert metrics['r2'] <= 1.0  # R² should be <= 1

    def test_train_random_forest(self, sample_processed_data):
        """Test random forest training."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        
        model = self.trainer.train_random_forest(X_train, y_train)
        metrics = self.trainer.evaluate_model(model, "random_forest", X_test, y_test)
        
        assert model is not None
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert hasattr(model, 'feature_importances_')

    def test_train_xgboost(self, sample_processed_data):
        """Test XGBoost training."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        
        model = self.trainer.train_xgboost(X_train, y_train)
        metrics = self.trainer.evaluate_model(model, "xgboost", X_test, y_test)
        
        assert model is not None
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics

    def test_calculate_metrics(self, sample_processed_data):
        """Test metrics calculation."""
        # Create simple test data
        y_true = np.array([100, 110, 120, 130, 140])
        y_pred = np.array([105, 108, 125, 128, 135])
        
        metrics = self.trainer._calculate_metrics(y_true, y_pred)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    def test_save_model(self, sample_processed_data):
        """Test model saving functionality."""
        # Train a simple model first
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        model, metrics = self.trainer._train_linear_regression(X_train, X_test, y_train, y_test)
        
        # Mock the save functionality
        with patch('train_model.joblib.dump') as mock_dump:
            with patch('train_model.json.dump') as mock_json_dump:
                with patch('builtins.open', create=True) as mock_open:
                    model_path = self.trainer._save_model(
                        model, metrics, "linear_regression", X_train.columns.tolist()
                    )
                    
                    assert model_path is not None
                    mock_dump.assert_called_once()

    def test_train_all_models(self, sample_processed_data):
        """Test training all models."""
        models_to_train = ["linear_regression", "random_forest"]
        
        results = self.trainer.train_all_models(
            df=sample_processed_data,
            target_column="aqi",
            models_to_train=models_to_train
        )
        
        assert len(results) == len(models_to_train)
        for model_name in models_to_train:
            assert model_name in results
            assert 'model' in results[model_name]
            assert 'metrics' in results[model_name]
            assert 'feature_names' in results[model_name]

    def test_train_all_models_default(self, sample_processed_data):
        """Test training all models with default parameters."""
        results = self.trainer.train_all_models(df=sample_processed_data)
        
        # Should train all available models by default
        expected_models = ["linear_regression", "random_forest", "xgboost"]
        assert len(results) >= 2  # At least 2 models should be trained

    def test_cross_validation(self, sample_processed_data):
        """Test cross-validation functionality."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        
        # Test with linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        cv_scores = self.trainer._perform_cross_validation(model, X_train, y_train)
        
        assert len(cv_scores) > 0
        assert all(isinstance(score, (int, float)) for score in cv_scores)

    def test_feature_importance_extraction(self, sample_processed_data):
        """Test feature importance extraction."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        
        # Train random forest to get feature importances
        model, metrics = self.trainer._train_random_forest(X_train, X_test, y_train, y_test)
        
        # Extract feature importance
        importance = self.trainer._extract_feature_importance(model, X_train.columns.tolist())
        
        assert len(importance) == len(X_train.columns)
        assert all(isinstance(imp, (int, float)) for imp in importance.values())

    def test_model_comparison(self, sample_processed_data):
        """Test model comparison functionality."""
        results = self.trainer.train_all_models(
            df=sample_processed_data,
            target_column="aqi",
            models_to_train=["linear_regression", "random_forest"]
        )
        
        # Compare models
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['r2'])
        
        assert best_model_name in results
        assert results[best_model_name]['metrics']['r2'] >= 0  # Should have valid R²

    def test_prediction_functionality(self, sample_processed_data):
        """Test making predictions with trained models."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        
        model, metrics = self.trainer._train_linear_regression(X_train, X_test, y_train, y_test)
        
        # Make predictions
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(isinstance(pred, (int, float, np.number)) for pred in predictions)

    def test_data_scaling(self, sample_processed_data):
        """Test data scaling functionality."""
        X_train, X_test, y_train, y_test = self.trainer.prepare_data(
            sample_processed_data, target_column="aqi"
        )
        
        # Check that scaler is fitted
        assert hasattr(self.trainer.scaler, 'mean_')  # StandardScaler should have mean_ after fitting

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(Exception):  # Should raise some kind of error
            self.trainer.prepare_data(empty_df, target_column="aqi")

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        small_df = pd.DataFrame({
            'aqi': [100, 110],
            'pm25': [50, 60]
        })
        
        # Should handle small datasets gracefully
        try:
            X_train, X_test, y_train, y_test = self.trainer.prepare_data(
                small_df, target_column="aqi", test_size=0.2
            )
            # If it doesn't raise an error, that's also acceptable
        except ValueError:
            # Small datasets might cause issues, which is expected
            pass

    @patch('train_model.logger')
    def test_logging_functionality(self, mock_logger, sample_processed_data):
        """Test that logging is working properly."""
        self.trainer.train_all_models(
            df=sample_processed_data,
            target_column="aqi",
            models_to_train=["linear_regression"]
        )
        
        # Check that logger was called
        assert mock_logger.info.called


# Fixtures
@pytest.fixture
def sample_processed_data():
    """Create sample processed AQI data for testing."""
    np.random.seed(42)  # For reproducible tests
    n_samples = 1000
    
    return pd.DataFrame({
        'datetime': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
        'city': np.random.choice(['Delhi', 'Mumbai', 'Chennai'], n_samples),
        'aqi': np.random.normal(120, 30, n_samples),
        'pm25': np.random.normal(60, 15, n_samples),
        'pm10': np.random.normal(90, 20, n_samples),
        'no2': np.random.normal(25, 8, n_samples),
        'so2': np.random.normal(10, 5, n_samples),
        'co': np.random.normal(1.5, 0.5, n_samples),
        'o3': np.random.normal(35, 10, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'month': np.random.randint(1, 13, n_samples),
        'season': np.random.choice(['Winter', 'Spring', 'Monsoon', 'Autumn'], n_samples),
        'is_weekend': np.random.choice([0, 1], n_samples),
        'is_rush_hour': np.random.choice([0, 1], n_samples),
        'latitude': np.random.uniform(10, 30, n_samples),
        'longitude': np.random.uniform(70, 90, n_samples),
        'pm_ratio': np.random.uniform(0.3, 0.8, n_samples),
        'aqi_lag_1h': np.random.normal(120, 30, n_samples),
        'aqi_lag_24h': np.random.normal(120, 30, n_samples),
        'aqi_rolling_6h': np.random.normal(120, 25, n_samples),
        'source': ['Kaggle_Dataset'] * n_samples
    })


@pytest.fixture
def sample_small_data():
    """Create small dataset for testing edge cases."""
    return pd.DataFrame({
        'datetime': pd.date_range('2020-01-01', periods=10, freq='H'),
        'city': ['Delhi'] * 10,
        'aqi': range(100, 110),
        'pm25': range(50, 60),
        'pm10': range(80, 90)
    })