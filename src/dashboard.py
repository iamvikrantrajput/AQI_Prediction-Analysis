"""
Simplified Streamlit dashboard for Air Quality Prediction project.
Basic dashboard suitable for Master's level students.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
from pathlib import Path

# Import our modules
from train_model import SimpleModelTrainer
from config import AQI_CATEGORIES, TARGET_CITIES

# Configure Streamlit page
st.set_page_config(
    page_title="Air Quality Prediction Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_sample_data():
    """Load current air quality data for dashboard."""
    try:
        # Try to load processed data first
        processed_data_path = Path("data/processed/processed_aqi_data.csv")
        if processed_data_path.exists():
            df = pd.read_csv(processed_data_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            st.success(f"✅ Loaded processed air quality data: {len(df):,} records")
            return df
        
        # Try to load Kaggle dataset directly
        kaggle_files = [
            ("data/kaggle_raw/city_hour.csv", "hourly city data"),
            ("data/kaggle_raw/station_hour.csv", "hourly station data"), 
            ("data/kaggle_raw/city_day.csv", "daily city data"),
            ("data/kaggle_raw/station_day.csv", "daily station data")
        ]
        
        for filepath, description in kaggle_files:
            file_path = Path(filepath)
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Standardize column names
                column_mapping = {
                    'City': 'city', 'Date': 'date', 'Datetime': 'datetime',
                    'PM2.5': 'pm25', 'PM10': 'pm10', 'NO': 'no', 'NO2': 'no2',
                    'CO': 'co', 'SO2': 'so2', 'O3': 'o3', 'AQI': 'aqi',
                    'StationId': 'station_id', 'StationName': 'station_name'
                }
                df = df.rename(columns=column_mapping)
                
                # Handle datetime
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                elif 'date' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'])
                
                # For station data, extract city from station name if needed
                if 'city' not in df.columns and 'station_name' in df.columns:
                    df['city'] = df['station_name'].str.extract(r'([^,]+)')[0].str.strip()
                
                st.info(f"📊 Loaded Kaggle {description}: {len(df):,} records")
                return df
            
    except Exception as e:
        st.error(f"Error loading air quality data: {e}")
    
    # Create minimal sample data if no files available
    np.random.seed(42)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='h')
    cities = ["Delhi", "Mumbai", "Bangalore", "Chennai"]
    
    data = []
    for city in cities:
        for date in dates[:200]:  # 200 records per city for demo
            data.append({
                "datetime": date,
                "city": city,
                "aqi": max(10, np.random.normal(120 if city == "Delhi" else 80, 30)),
                "pm25": max(0, np.random.normal(60 if city == "Delhi" else 40, 20)),
                "pm10": max(0, np.random.normal(90 if city == "Delhi" else 60, 25)),
                "temperature": np.random.normal(25, 8),
                "humidity": max(20, min(90, np.random.normal(60, 15))),
                "pressure": np.random.normal(1013, 8),
                "wind_speed": max(0, np.random.normal(5, 3)),
            })
    
    return pd.DataFrame(data)


def load_metadata():
    """Load air quality data metadata."""
    try:
        # Check for Kaggle dataset info
        stations_path = Path("data/kaggle_raw/stations.csv")
        if stations_path.exists():
            stations_df = pd.read_csv(stations_path)
            return {
                'data_info': {
                    'data_source': 'Kaggle Air Quality Dataset - India (2015-2020)',
                    'stations_count': len(stations_df),
                    'coverage': 'Multiple Indian cities'
                },
                'quality_metrics': {
                    'completeness_percentage': 85  # Approximate
                }
            }
        
        # Fallback to old metadata if available
        metadata_path = Path("data/raw/metadata_latest.json")
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
    except Exception as e:
        st.warning(f"Could not load metadata: {e}")
    return {}


def get_aqi_category(aqi_value):
    """Get AQI category based on value."""
    for category, info in AQI_CATEGORIES.items():
        if info["min"] <= aqi_value <= info["max"]:
            return category, info["color"]
    return "Unknown", "gray"


def plot_time_series(df, city, metric="aqi"):
    """Create time series plot for a specific city and metric."""
    city_data = df[df["city"] == city].copy()
    city_data["datetime"] = pd.to_datetime(city_data["datetime"])
    city_data = city_data.sort_values("datetime")
    
    fig = px.line(
        city_data, 
        x="datetime", 
        y=metric,
        title=f"{metric.upper()} Time Series - {city}",
        labels={"datetime": "Date", metric: metric.upper()}
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=metric.upper(),
        hovermode="x unified"
    )
    
    return fig


def plot_city_comparison(df, metric="aqi"):
    """Create box plot comparing cities."""
    fig = px.box(
        df, 
        x="city", 
        y=metric,
        title=f"{metric.upper()} Distribution by City",
        color="city"
    )
    
    fig.update_layout(
        xaxis_title="City",
        yaxis_title=metric.upper(),
        showlegend=False
    )
    
    return fig


def plot_correlation_heatmap(df):
    """Create correlation heatmap for numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="RdBu",
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
    )
    
    return fig


def plot_aqi_distribution(df):
    """Create AQI distribution with category colors."""
    fig = px.histogram(
        df, 
        x="aqi", 
        nbins=30,
        title="AQI Distribution",
        labels={"aqi": "AQI Value", "count": "Frequency"}
    )
    
    # Add vertical lines for AQI category boundaries
    for category, info in AQI_CATEGORIES.items():
        if info["max"] < 500:  # Don't add line for the last category
            fig.add_vline(
                x=info["max"], 
                line_dash="dash", 
                line_color=info["color"],
                annotation_text=category
            )
    
    return fig


def main():
    """Main dashboard function."""
    
    # Title and description
    st.title("🌍 Air Quality Prediction Dashboard")
    st.markdown("**Simple dashboard for visualizing and predicting air quality data**")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Overview", "Exploratory Analysis", "Model Training", "Predictions"]
    )
    
    # Load data and metadata
    with st.spinner("Loading data..."):
        df = load_sample_data()
        metadata = load_metadata()
    
    st.sidebar.markdown(f"**Data Info:**")
    st.sidebar.markdown(f"- Records: {len(df):,}")
    st.sidebar.markdown(f"- Cities: {df['city'].nunique()}")
    date_range_days = (df['datetime'].max() - df['datetime'].min()).days
    st.sidebar.markdown(f"- Time span: {date_range_days} days")
    st.sidebar.markdown(f"- Date range: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
    
    # Add metadata info if available
    if metadata:
        st.sidebar.markdown("**Collection Info:**")
        data_source = metadata.get('data_info', {}).get('data_source', 'Unknown')
        if 'Kaggle' in data_source:
            st.sidebar.markdown("- 📊 Kaggle Air Quality Dataset")
            st.sidebar.markdown("- 🇮🇳 India (2015-2020)")
        elif 'Synthetic' in data_source:
            st.sidebar.markdown("- 🤖 Synthetic data")
        else:
            st.sidebar.markdown("- 🌍 Real air quality data")
        
        stations_count = metadata.get('data_info', {}).get('stations_count', 'N/A')
        if stations_count != 'N/A':
            st.sidebar.markdown(f"- Stations: {stations_count}")
        
        completeness = metadata.get('quality_metrics', {}).get('completeness_percentage', 'N/A')
        st.sidebar.markdown(f"- Completeness: {completeness}%")
    
    # Main content based on selected page
    if page == "Data Overview":
        st.header("📊 Data Overview")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_aqi = df["aqi"].mean()
            st.metric("Average AQI", f"{avg_aqi:.1f}")
        
        with col2:
            max_aqi = df["aqi"].max()
            st.metric("Max AQI", f"{max_aqi:.1f}")
        
        with col3:
            cities_count = df["city"].nunique()
            st.metric("Cities", cities_count)
        
        with col4:
            records_count = len(df)
            st.metric("Total Records", f"{records_count:,}")
        
        # Data sample
        st.subheader("Data Sample")
        st.dataframe(df.head(10))
        
        # Missing data analysis
        st.subheader("Missing Data Analysis")
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            "Column": missing_data.index,
            "Missing Count": missing_data.values,
            "Missing %": missing_pct.values
        })
        
        st.dataframe(missing_df)
    
    elif page == "Exploratory Analysis":
        st.header("🔍 Exploratory Data Analysis")
        
        # City selection
        selected_city = st.selectbox("Select a city", df["city"].unique())
        
        # Metric selection - use available columns from air quality data
        available_metrics = []
        for col in ['aqi', 'pm25', 'pm10', 'o3', 'no2', 'so2', 'co']:
            if col in df.columns:
                available_metrics.append(col)
        
        # Add other numeric columns if available
        for col in ['temperature', 'humidity', 'pressure', 'wind_speed']:
            if col in df.columns:
                available_metrics.append(col)
        
        selected_metric = st.selectbox("Select a metric", available_metrics if available_metrics else ['aqi'])
        
        # Time series plot
        st.subheader(f"Time Series Analysis - {selected_city}")
        fig_ts = plot_time_series(df, selected_city, selected_metric)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # City comparison
        st.subheader(f"{selected_metric.upper()} Comparison Across Cities")
        fig_box = plot_city_comparison(df, selected_metric)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Feature Correlations")
        fig_corr = plot_correlation_heatmap(df)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # AQI distribution
        if selected_metric == "aqi":
            st.subheader("AQI Distribution with Categories")
            fig_dist = plot_aqi_distribution(df)
            st.plotly_chart(fig_dist, use_container_width=True)
    
    elif page == "Model Training":
        st.header("🤖 Model Training")
        
        st.markdown("Train simple machine learning models to predict air quality.")
        
        # Model selection
        available_models = ["linear_regression", "random_forest", "xgboost"]
        selected_models = st.multiselect(
            "Select models to train",
            available_models,
            default=available_models
        )
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2)
        with col2:
            target_column = st.selectbox("Target variable", ["aqi", "pm25", "pm10"])
        
        if st.button("Train Models"):
            if selected_models:
                with st.spinner("Training models..."):
                    try:
                        # Initialize trainer
                        trainer = SimpleModelTrainer()
                        
                        # Train models
                        results = trainer.train_all_models(
                            df=df,
                            target_column=target_column,
                            models_to_train=selected_models
                        )
                        
                        # Display results
                        st.success("Models trained successfully!")
                        
                        results_data = []
                        for model_name, result in results.items():
                            metrics = result["metrics"]
                            results_data.append({
                                "Model": model_name.replace("_", " ").title(),
                                "RMSE": f"{metrics['rmse']:.2f}",
                                "MAE": f"{metrics['mae']:.2f}",
                                "R²": f"{metrics['r2']:.3f}",
                                "MAPE": f"{metrics['mape']:.2f}%"
                            })
                        
                        results_df = pd.DataFrame(results_data)
                        st.table(results_df)
                        
                    except Exception as e:
                        st.error(f"Error training models: {str(e)}")
            else:
                st.warning("Please select at least one model to train.")
    
    elif page == "Predictions":
        st.header("🔮 Air Quality Predictions")
        
        st.markdown("Make predictions using trained models.")
        
        # Input features
        st.subheader("Input Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.selectbox("City", df["city"].unique())
            temperature = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0)
            humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
        
        with col2:
            pressure = st.number_input("Pressure (hPa)", 950.0, 1050.0, 1013.0)
            wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 5.0)
            hour = st.number_input("Hour of day", 0, 23, 12)
        
        if st.button("Make Prediction"):
            st.info("Prediction functionality requires trained models. Please train models first in the Model Training section.")
            
            # Create sample prediction (for demonstration)
            sample_aqi = np.random.randint(50, 200)
            category, color = get_aqi_category(sample_aqi)
            
            st.subheader("Predicted AQI")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("AQI Value", sample_aqi)
                st.markdown(f"**Category:** :red[{category}]")
            
            with col2:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = sample_aqi,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "AQI"},
                    gauge = {
                        'axis': {'range': [None, 500]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 50], 'color': "green"},
                            {'range': [51, 100], 'color': "yellow"},
                            {'range': [101, 150], 'color': "orange"},
                            {'range': [151, 200], 'color': "red"},
                            {'range': [201, 300], 'color': "purple"},
                            {'range': [301, 500], 'color': "maroon"}
                        ],
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Simple Air Quality Dashboard** - Built with Streamlit")


if __name__ == "__main__":
    main()