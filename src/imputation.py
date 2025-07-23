"""
Simplified missing data imputation module for Air Quality Prediction project.
Implements basic imputation strategies suitable for Master's level students.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal
from sklearn.impute import SimpleImputer

# Simple logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_imputation")


class SimpleMissingDataHandler:
    """Simple missing data handler for Master's level students."""
    
    def __init__(self):
        """Initialize the missing data handler."""
        self.imputers = {}
        self.missing_info = {}
    
    def analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns.
        
        Args:
            df: Input dataframe
        
        Returns:
            Dictionary with missing data information
        """
        logger.info("Analyzing missing data patterns...")
        
        missing_info = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            missing_info[column] = {
                "missing_count": missing_count,
                "missing_percentage": round(missing_percentage, 2),
                "data_type": str(df[column].dtype)
            }
        
        # Summary statistics
        total_missing = sum([info["missing_count"] for info in missing_info.values()])
        total_cells = len(df) * len(df.columns)
        
        summary = {
            "total_missing_values": total_missing,
            "total_cells": total_cells,
            "overall_missing_percentage": round((total_missing / total_cells) * 100, 2),
            "columns_with_missing": len([col for col, info in missing_info.items() 
                                       if info["missing_count"] > 0]),
            "column_details": missing_info
        }
        
        self.missing_info = summary
        
        # Print summary
        print(f"Missing Data Analysis:")
        print(f"- Total missing values: {total_missing:,}")
        print(f"- Overall missing percentage: {summary['overall_missing_percentage']}%")
        print(f"- Columns with missing data: {summary['columns_with_missing']}")
        print()
        
        # Show columns with high missing percentages
        high_missing = [(col, info["missing_percentage"]) 
                       for col, info in missing_info.items() 
                       if info["missing_percentage"] > 10]
        
        if high_missing:
            print("Columns with >10% missing data:")
            for col, pct in sorted(high_missing, key=lambda x: x[1], reverse=True):
                print(f"  {col}: {pct}%")
        
        return summary
    
    def impute_missing_data(
        self,
        df: pd.DataFrame,
        strategy: Literal["mean", "median", "mode", "forward_fill", "drop"] = "mean",
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Impute missing data using simple strategies.
        
        Args:
            df: Input dataframe
            strategy: Imputation strategy
            threshold: Drop columns with missing > threshold
        
        Returns:
            Dataframe with imputed values
        """
        logger.info(f"Imputing missing data using {strategy} strategy...")
        
        df_imputed = df.copy()
        
        # First, drop columns with too many missing values
        missing_percentages = df_imputed.isnull().sum() / len(df_imputed)
        cols_to_drop = missing_percentages[missing_percentages > threshold].index.tolist()
        
        if cols_to_drop:
            logger.info(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
            df_imputed = df_imputed.drop(columns=cols_to_drop)
        
        # Separate numeric and non-numeric columns
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
        categorical_columns = df_imputed.select_dtypes(include=['object']).columns
        
        # Handle numeric columns
        if len(numeric_columns) > 0:
            if strategy in ["mean", "median"]:
                imputer = SimpleImputer(strategy=strategy)
                df_imputed[numeric_columns] = imputer.fit_transform(df_imputed[numeric_columns])
                self.imputers["numeric"] = imputer
                
            elif strategy == "forward_fill":
                df_imputed[numeric_columns] = df_imputed[numeric_columns].fillna(method='ffill')
                # Fill remaining NaNs with mean
                df_imputed[numeric_columns] = df_imputed[numeric_columns].fillna(
                    df_imputed[numeric_columns].mean()
                )
        
        # Handle categorical columns
        if len(categorical_columns) > 0:
            if strategy == "mode":
                imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_columns] = imputer.fit_transform(df_imputed[categorical_columns])
                self.imputers["categorical"] = imputer
                
            elif strategy == "forward_fill":
                df_imputed[categorical_columns] = df_imputed[categorical_columns].fillna(method='ffill')
                # Fill remaining NaNs with mode
                for col in categorical_columns:
                    mode_value = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else "Unknown"
                    df_imputed[col] = df_imputed[col].fillna(mode_value)
            else:
                # Default to most frequent for categorical
                imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[categorical_columns] = imputer.fit_transform(df_imputed[categorical_columns])
                self.imputers["categorical"] = imputer
        
        # Drop strategy
        if strategy == "drop":
            df_imputed = df_imputed.dropna()
        
        logger.info(f"Imputation complete. Shape: {df.shape} -> {df_imputed.shape}")
        return df_imputed
    
    def impute_by_group(
        self,
        df: pd.DataFrame,
        group_column: str = "city",
        strategy: str = "mean",
    ) -> pd.DataFrame:
        """
        Impute missing data by groups (e.g., by city).
        
        Args:
            df: Input dataframe
            group_column: Column to group by
            strategy: Imputation strategy
        
        Returns:
            Dataframe with group-wise imputed values
        """
        logger.info(f"Imputing missing data by {group_column} using {strategy}...")
        
        df_imputed = df.copy()
        numeric_columns = df_imputed.select_dtypes(include=[np.number]).columns
        
        # Group-wise imputation for numeric columns
        for col in numeric_columns:
            if df_imputed[col].isnull().any():
                if strategy == "mean":
                    group_means = df_imputed.groupby(group_column)[col].mean()
                    df_imputed[col] = df_imputed[col].fillna(
                        df_imputed[group_column].map(group_means)
                    )
                elif strategy == "median":
                    group_medians = df_imputed.groupby(group_column)[col].median()
                    df_imputed[col] = df_imputed[col].fillna(
                        df_imputed[group_column].map(group_medians)
                    )
                
                # Fill any remaining NaNs with overall mean/median
                if df_imputed[col].isnull().any():
                    if strategy == "mean":
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
                    elif strategy == "median":
                        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
        
        logger.info("Group-wise imputation complete")
        return df_imputed
    
    def create_missing_indicators(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Create binary indicators for missing values.
        
        Args:
            df: Input dataframe
            columns: Columns to create indicators for
        
        Returns:
            Dataframe with missing indicators
        """
        if columns is None:
            # Only create indicators for columns with missing values
            columns = df.columns[df.isnull().any()].tolist()
        
        df_with_indicators = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].isnull().any():
                indicator_name = f"{col}_was_missing"
                df_with_indicators[indicator_name] = df[col].isnull().astype(int)
        
        logger.info(f"Created missing indicators for {len(columns)} columns")
        return df_with_indicators
    
    def get_imputation_summary(self) -> Dict:
        """
        Get summary of imputation process.
        
        Returns:
            Summary dictionary
        """
        return {
            "missing_info": self.missing_info,
            "imputers_used": list(self.imputers.keys()),
        }


def main():
    """Example usage of SimpleMissingDataHandler."""
    # Create sample data with missing values
    np.random.seed(42)
    data = {
        "city": ["Delhi", "Mumbai", "Delhi", "Mumbai", "Delhi"] * 20,
        "aqi": np.random.normal(100, 30, 100),
        "temperature": np.random.normal(25, 10, 100),
        "humidity": np.random.normal(60, 20, 100),
        "pressure": np.random.normal(1013, 10, 100),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[missing_indices[:10], "aqi"] = np.nan
    df.loc[missing_indices[10:], "temperature"] = np.nan
    
    print("Original data shape:", df.shape)
    print("Missing values per column:")
    print(df.isnull().sum())
    print()
    
    # Initialize handler and analyze
    handler = SimpleMissingDataHandler()
    missing_analysis = handler.analyze_missing_data(df)
    print()
    
    # Impute missing data
    df_imputed = handler.impute_missing_data(df, strategy="mean")
    print("After imputation:")
    print("Missing values per column:")
    print(df_imputed.isnull().sum())
    
    # Try group-wise imputation
    df_group_imputed = handler.impute_by_group(df, group_column="city", strategy="mean")
    print("\nAfter group-wise imputation:")
    print("Missing values per column:")
    print(df_group_imputed.isnull().sum())


if __name__ == "__main__":
    main()