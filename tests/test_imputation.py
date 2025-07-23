"""Tests for imputation module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.imputation import SimpleMissingDataHandler


class TestSimpleMissingDataHandler:
    """Test SimpleMissingDataHandler class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = SimpleMissingDataHandler()

    def test_init(self):
        """Test SimpleMissingDataHandler initialization."""
        handler = SimpleMissingDataHandler()
        assert handler.imputers == {}
        assert handler.missing_info == {}

    def test_analyze_missing_data(self, sample_data_with_missing):
        """Test missing data analysis."""
        result = self.handler.analyze_missing_data(sample_data_with_missing)
        
        # Check that analysis results are returned
        assert isinstance(result, dict)
        assert "missing_summary" in result
        assert "column_details" in result
        
        # Check column details
        for col in sample_data_with_missing.columns:
            assert col in result["column_details"]
            assert "missing_count" in result["column_details"][col]
            assert "missing_percentage" in result["column_details"][col]

    def test_analyze_missing_data_no_missing(self, sample_aqi_data):
        """Test missing data analysis with no missing values."""
        result = self.handler.analyze_missing_data(sample_aqi_data)
        
        assert result["missing_summary"]["total_missing"] == 0
        assert result["missing_summary"]["percentage_missing"] == 0

    def test_impute_missing_data_mean(self, sample_data_with_missing):
        """Test mean imputation."""
        original_missing = sample_data_with_missing.isnull().sum().sum()
        
        result = self.handler.impute_missing_data(
            sample_data_with_missing, strategy="mean"
        )
        
        final_missing = result.isnull().sum().sum()
        assert final_missing < original_missing

    def test_impute_missing_data_median(self, sample_data_with_missing):
        """Test median imputation."""
        original_missing = sample_data_with_missing.isnull().sum().sum()
        
        result = self.handler.impute_missing_data(
            sample_data_with_missing, strategy="median"
        )
        
        final_missing = result.isnull().sum().sum()
        assert final_missing < original_missing

    def test_impute_missing_data_mode(self):
        """Test mode imputation with categorical data."""
        data = {
            "datetime": pd.date_range("2023-01-01", periods=10, freq="H"),
            "city": ["Delhi"] * 5 + [np.nan] * 5,
            "category": ["A"] * 3 + ["B"] * 2 + [np.nan] * 5
        }
        df = pd.DataFrame(data)
        
        result = self.handler.impute_missing_data(df, strategy="mode")
        
        # Mode imputation should fill missing values
        assert result["city"].isnull().sum() == 0
        assert result["category"].isnull().sum() == 0

    def test_impute_missing_data_group_wise(self, sample_data_with_missing):
        """Test group-wise imputation."""
        result = self.handler.impute_missing_data(
            sample_data_with_missing, 
            strategy="mean",
            group_by="city"
        )
        
        # Should have fewer missing values
        original_missing = sample_data_with_missing.isnull().sum().sum()
        final_missing = result.isnull().sum().sum()
        assert final_missing <= original_missing

    def test_impute_missing_data_with_indicators(self, sample_data_with_missing):
        """Test imputation with missing value indicators."""
        result = self.handler.impute_missing_data(
            sample_data_with_missing,
            strategy="mean",
            add_indicators=True
        )
        
        # Should have indicator columns
        indicator_cols = [col for col in result.columns if col.endswith("_missing")]
        assert len(indicator_cols) > 0

    def test_impute_missing_data_preserve_types(self, sample_data_with_missing):
        """Test that data types are preserved after imputation."""
        original_dtypes = sample_data_with_missing.dtypes.to_dict()
        
        result = self.handler.impute_missing_data(
            sample_data_with_missing, strategy="mean"
        )
        
        # Check that datetime columns are preserved
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])

    def test_impute_missing_data_invalid_strategy(self, sample_data_with_missing):
        """Test imputation with invalid strategy."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            self.handler.impute_missing_data(
                sample_data_with_missing, strategy="invalid"
            )

    def test_create_missing_indicators(self, sample_data_with_missing):
        """Test missing indicator creation."""
        result = self.handler.create_missing_indicators(sample_data_with_missing)
        
        # Should have indicator columns for columns with missing values
        missing_cols = sample_data_with_missing.columns[
            sample_data_with_missing.isnull().any()
        ].tolist()
        
        for col in missing_cols:
            indicator_col = f"{col}_missing"
            assert indicator_col in result.columns
            assert result[indicator_col].dtype == bool

    def test_get_missing_summary(self, sample_data_with_missing):
        """Test missing data summary."""
        summary = self.handler.get_missing_summary(sample_data_with_missing)
        
        assert "total_cells" in summary
        assert "missing_cells" in summary
        assert "percentage_missing" in summary
        assert "columns_with_missing" in summary

    def test_plot_missing_pattern(self, sample_data_with_missing):
        """Test missing pattern visualization (basic check)."""
        # This should not raise an exception
        try:
            self.handler.plot_missing_pattern(sample_data_with_missing)
        except ImportError:
            # If matplotlib/seaborn not available, skip
            pytest.skip("Visualization libraries not available")

    def test_recommend_strategy(self, sample_data_with_missing):
        """Test strategy recommendation."""
        recommendations = self.handler.recommend_strategy(sample_data_with_missing)
        
        assert isinstance(recommendations, dict)
        
        for col in sample_data_with_missing.columns:
            if col in recommendations:
                assert recommendations[col] in ["mean", "median", "mode", "drop"]

    @patch("src.imputation.logger")
    def test_logging_calls(self, mock_logger, sample_data_with_missing):
        """Test that logging is called appropriately."""
        self.handler.analyze_missing_data(sample_data_with_missing)
        assert mock_logger.info.called

    def test_handle_all_missing_column(self):
        """Test handling of columns with all missing values."""
        data = {
            "datetime": pd.date_range("2023-01-01", periods=5, freq="H"),
            "good_col": [1, 2, 3, 4, 5],
            "all_missing": [np.nan] * 5
        }
        df = pd.DataFrame(data)
        
        result = self.handler.impute_missing_data(df, strategy="mean")
        
        # All missing column should still be handled somehow
        assert "all_missing" in result.columns

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()
        
        result = self.handler.analyze_missing_data(df)
        assert result["missing_summary"]["total_missing"] == 0

    def test_single_row_dataframe(self):
        """Test handling of single row dataframe."""
        data = {"col1": [1], "col2": [np.nan]}
        df = pd.DataFrame(data)
        
        result = self.handler.impute_missing_data(df, strategy="mean")
        assert len(result) == 1