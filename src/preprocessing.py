"""
Data Preprocessing Module
Handles data cleaning, merging, and feature engineering for air quality data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET_CITIES,
    CITY_COORDINATES, AQI_CATEGORIES, get_aqi_category
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIDataPreprocessor:
    """Main preprocessing class for AQI prediction data."""
    
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Load air quality data from Kaggle dataset."""
        logger.info("Loading air quality data from Kaggle dataset...")
        
        # Load Kaggle dataset files in order of preference
        kaggle_files = [
            ('kaggle_raw/city_hour.csv', 'hourly city data'),
            ('kaggle_raw/station_hour.csv', 'hourly station data'), 
            ('kaggle_raw/city_day.csv', 'daily city data'),
            ('kaggle_raw/station_day.csv', 'daily station data')
        ]
        
        for filename, description in kaggle_files:
            filepath = self.raw_data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    logger.info(f"Loaded Kaggle {description}: {len(df)} records")
                    
                    # Standardize column names for Kaggle data
                    df = self._standardize_kaggle_columns(df)
                    return df
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        # Fallback to other sample data if Kaggle data not available
        sample_files = [
            'aqi_weather_traffic_data.csv',
            'sample_aqi_data.csv'
        ]
        
        for filename in sample_files:
            filepath = self.raw_data_dir / filename
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    logger.info(f"Loaded fallback data from {filename}: {len(df)} records")
                    return df
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        # If no data found, create minimal sample
        logger.warning("No data files found, creating minimal sample...")
        return self._create_minimal_sample()
    
    def _standardize_kaggle_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Kaggle dataset column names to match expected format."""
        df = df.copy()
        
        # Map Kaggle columns to standard names
        column_mapping = {
            'City': 'city',
            'Date': 'date', 
            'Datetime': 'datetime',
            'PM2.5': 'pm25',
            'PM10': 'pm10',
            'NO': 'no',
            'NO2': 'no2',
            'NOx': 'nox', 
            'NH3': 'nh3',
            'CO': 'co',
            'SO2': 'so2',
            'O3': 'o3',
            'Benzene': 'benzene',
            'Toluene': 'toluene',
            'Xylene': 'xylene',
            'AQI': 'aqi',
            'AQI_Bucket': 'aqi_bucket',
            'StationId': 'station_id',
            'StationName': 'station_name',
            'State': 'state',
            'Status': 'status'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Handle datetime column creation for daily data
        if 'date' in df.columns and 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        
        # For station data, extract city from station name if city column missing
        if 'city' not in df.columns and 'station_name' in df.columns:
            # Try to extract city from station name (rough approach)
            df['city'] = df['station_name'].str.split(',').str[-1].str.strip()
            df['city'] = df['city'].str.replace(' - APPCB', '').str.replace(' - CPCB', '')
        
        # Add source column to track data origin
        df['source'] = 'Kaggle_Dataset'
        
        return df
    
    def _create_minimal_sample(self) -> pd.DataFrame:
        """Create minimal sample data for demonstration."""
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30),
            end=datetime.now(),
            freq='H'
        )
        
        data = []
        for date in dates:
            for city in TARGET_CITIES[:3]:  # Just 3 cities for minimal sample
                # Generate realistic but random AQI values
                base_aqi = {'Delhi': 150, 'Mumbai': 100, 'Bangalore': 80}.get(city, 100)
                aqi = base_aqi + np.random.normal(0, 30)
                aqi = max(10, min(400, aqi))
                
                data.append({
                    'datetime': date,
                    'city': city,
                    'location': f"{city}_Station_1",
                    'co': aqi * 0.02 + np.random.normal(0, 0.5),
                    'no2': aqi * 0.3 + np.random.normal(0, 8),
                    'o3': np.random.normal(40, 15),
                    'pm10': aqi * 0.6 + np.random.normal(0, 15),
                    'pm25': aqi * 0.4 + np.random.normal(0, 10),
                    'so2': np.random.normal(15, 8),
                    'source': 'Minimal_Sample'
                })
        
        df = pd.DataFrame(data)
        # Calculate AQI from PM2.5 (simplified)
        df['aqi'] = df['pm25'] * 2.5  # Rough conversion
        df['aqi'] = df['aqi'].clip(0, 500)
        
        logger.info(f"Created minimal sample: {len(df)} records")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize air quality data."""
        logger.info("Cleaning air quality data...")
        
        if df.empty:
            return df
        
        df = df.copy()
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Handle different datetime formats
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        elif 'collection_time' in df.columns:
            # For data that might use collection_time
            df['datetime'] = pd.to_datetime(df['collection_time'], errors='coerce')
        else:
            # Create datetime from current time if not available
            logger.warning("No datetime column found, using current time")
            df['datetime'] = datetime.now()
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['datetime'])
        
        # Ensure city column exists
        if 'city' not in df.columns:
            if 'location' in df.columns:
                # Try to extract city from location
                df['city'] = df['location'].str.split('_').str[0]
            else:
                logger.error("No city information found")
                return pd.DataFrame()
        
        # Clean city names
        df['city'] = df['city'].str.title().str.strip()
        
        # Filter for target cities (case insensitive) - now Top 8 cities only
        df = df[df['city'].str.lower().isin([city.lower() for city in TARGET_CITIES])]
        
        if df.empty:
            logger.warning("No data remaining after city filtering")
            return df
        
        # Clean pollutant columns
        pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co', 'aqi']
        weather_cols = ['temperature', 'pressure', 'humidity', 'wind_speed', 'wind_gust', 'dew_point']
        all_numeric_cols = pollutant_cols + weather_cols
        
        for col in all_numeric_cols:
            if col in df.columns:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove extreme outliers (values > 99.9th percentile)
                if df[col].notna().any():
                    upper_limit = df[col].quantile(0.999)
                    df.loc[df[col] > upper_limit, col] = np.nan
                    
                    # Remove negative values for pollutants (but allow negative temps)
                    if col not in ['temperature', 'dew_point']:
                        df.loc[df[col] < 0, col] = np.nan
        
        # Calculate AQI if not present
        if 'aqi' not in df.columns and 'pm25' in df.columns:
            # Simple AQI calculation from PM2.5 (US EPA formula simplified)
            df['aqi'] = df['pm25'].apply(self._pm25_to_aqi)
        
        # Remove rows where all pollutants are NaN
        pollutant_cols_present = [col for col in pollutant_cols if col in df.columns]
        if pollutant_cols_present:
            df = df.dropna(subset=pollutant_cols_present, how='all')
        
        logger.info(f"Cleaned air quality data: {len(df)} records remaining")
        return df
    
    def _pm25_to_aqi(self, pm25_value):
        """Convert PM2.5 to AQI using simplified US EPA formula."""
        if pd.isna(pm25_value):
            return np.nan
        
        # Simplified AQI breakpoints for PM2.5
        if pm25_value <= 12:
            return (50/12) * pm25_value
        elif pm25_value <= 35.4:
            return 50 + ((100-50)/(35.4-12)) * (pm25_value - 12)
        elif pm25_value <= 55.4:
            return 100 + ((150-100)/(55.4-35.4)) * (pm25_value - 35.4)
        elif pm25_value <= 150.4:
            return 150 + ((200-150)/(150.4-55.4)) * (pm25_value - 55.4)
        elif pm25_value <= 250.4:
            return 200 + ((300-200)/(250.4-150.4)) * (pm25_value - 150.4)
        else:
            return 300 + ((400-300)/(350.4-250.4)) * (pm25_value - 250.4)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for modeling."""
        logger.info("Creating features...")
        
        if df.empty:
            return df
        
        df = df.copy()
        
        # Time-based features
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['day_of_year'] = df['datetime'].dt.dayofyear
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Season mapping
            def get_season(month):
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8, 9]:
                    return 'Monsoon'
                else:
                    return 'Autumn'
            
            df['season'] = df['month'].apply(get_season)
            
            # Rush hour indicator
            df['is_rush_hour'] = df['hour'].apply(
                lambda x: 1 if x in [7, 8, 9, 17, 18, 19, 20] else 0
            )
            
            # Early morning clean air hours
            df['is_clean_hour'] = df['hour'].apply(
                lambda x: 1 if x in [2, 3, 4, 5] else 0
            )
        
        # Location-based features
        if 'city' in df.columns:
            # Add coordinates and state info
            for city, coords in CITY_COORDINATES.items():
                mask = df['city'] == city
                df.loc[mask, 'latitude'] = coords['lat']
                df.loc[mask, 'longitude'] = coords['lon'] 
                df.loc[mask, 'state'] = coords['state']
        
        # AQI category
        if 'aqi' in df.columns:
            df['aqi_category'] = df['aqi'].apply(get_aqi_category)
        
        # Pollutant ratios (if columns exist)
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)  # Avoid division by zero
        
        # Lag features for time series (if enough data)
        if len(df) > 48 and 'aqi' in df.columns:  # At least 48 hours of data
            df = df.sort_values(['city', 'datetime'])
            
            # Create lag features for each city
            for lag in [1, 6, 12, 24]:
                col_name = f'aqi_lag_{lag}h'
                df[col_name] = df.groupby('city')['aqi'].shift(lag)
            
            # Rolling averages
            for window in [6, 12, 24]:
                col_name = f'aqi_rolling_{window}h'
                df[col_name] = df.groupby('city')['aqi'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
        
        logger.info(f"Feature engineering complete: {df.shape[1]} features")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info("Handling missing values...")
        
        if df.empty:
            return df
        
        df = df.copy()
        
        # Forward fill for time series data (within each city)
        if 'city' in df.columns and 'datetime' in df.columns:
            df = df.sort_values(['city', 'datetime'])
            
            # Forward fill within each city group
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col not in ['latitude', 'longitude']:  # Don't fill coordinates
                    df[col] = df.groupby('city')[col].fillna(method='ffill')
        
        # Fill remaining missing values with appropriate strategies
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if col in ['latitude', 'longitude']:
                    continue  # Skip coordinates
                elif 'lag' in col or 'rolling' in col:
                    # Don't fill lag/rolling features aggressively
                    continue
                else:
                    # Fill with median for other numerical columns
                    df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['datetime', 'city']:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        logger.info(f"Missing value handling complete. Remaining NaN: {df.isnull().sum().sum()}")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'processed_aqi_data.csv'):
        """Save processed data to file."""
        filepath = self.processed_data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to: {filepath}")
        return filepath
    
    def run_preprocessing_pipeline(self) -> pd.DataFrame:
        """Run the complete preprocessing pipeline for air quality data."""
        logger.info("🔄 Starting air quality preprocessing pipeline...")
        
        # Load air quality data
        raw_data = self.load_data()
        
        if raw_data.empty:
            raise ValueError("No data available for processing")
        
        # Clean air quality data
        cleaned_data = self.clean_data(raw_data)
        
        # Create features
        featured_data = self.create_features(cleaned_data)
        
        # Handle missing values
        final_data = self.handle_missing_values(featured_data)
        
        # Save processed data
        self.save_processed_data(final_data)
        
        # Generate summary
        logger.info("📊 Preprocessing Summary:")
        logger.info(f"   Total records: {len(final_data):,}")
        logger.info(f"   Cities (Top 8): {final_data['city'].nunique() if 'city' in final_data.columns else 'N/A'}")
        if 'city' in final_data.columns:
            logger.info(f"   Cities included: {', '.join(sorted(final_data['city'].unique()))}")
        logger.info(f"   Date range: {final_data['datetime'].min()} to {final_data['datetime'].max()}" 
                   if 'datetime' in final_data.columns else "   No datetime column")
        logger.info(f"   Features: {final_data.shape[1]}")
        
        if 'aqi' in final_data.columns:
            aqi_stats = final_data['aqi'].describe()
            logger.info(f"   AQI range: {aqi_stats['min']:.1f} - {aqi_stats['max']:.1f}")
            logger.info(f"   AQI mean: {aqi_stats['mean']:.1f}")
        
        # Show data source
        if 'source' in final_data.columns:
            sources = final_data['source'].value_counts()
            logger.info(f"   Data sources: {dict(sources)}")
        
        return final_data

def main():
    """Main function to run preprocessing."""
    preprocessor = AQIDataPreprocessor()
    
    try:
        processed_data = preprocessor.run_preprocessing_pipeline()
        print("✅ Air quality data preprocessing completed successfully!")
        return processed_data
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()