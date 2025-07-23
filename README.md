# Air Quality Prediction & Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Kaggle Dataset](https://img.shields.io/badge/data-Kaggle%20Dataset-blue.svg)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive data science project for predicting Air Quality Index (AQI) using historical air quality data from India (2015-2020) and machine learning. Features end-to-end ML pipeline, comprehensive data analysis, and interactive visualizations.

## 🚀 **Latest Updates**
- ✅ **Kaggle Dataset Integration**: Using comprehensive air quality dataset covering 6 major Indian cities (2015-2020)
- ✅ **Historical Data Analysis**: 242K+ records with rich pollutant and meteorological data
- ✅ **Enhanced Feature Engineering**: 37+ features including temporal patterns and location-based features
- ✅ **Production-Ready Pipeline**: One-command execution with robust data processing
- ✅ **Interactive Dashboard**: Streamlit-based visualization and analysis interface

## 🎯 Project Overview

This project analyzes historical air quality data from India's major cities to build predictive models for Air Quality Index (AQI) forecasting. The system provides comprehensive insights through interactive visualizations and supports multiple machine learning approaches.

### Key Features

- **Rich Historical Dataset**: 5+ years of air quality data (2015-2020) from multiple Indian cities
- **Comprehensive Pollutant Coverage**: PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene
- **Multiple ML Models**: Linear Regression, Random Forest, XGBoost with performance comparison
- **Advanced Analytics**: Time series analysis, correlation studies, city-wise comparisons
- **Interactive Dashboard**: Real-time visualizations and prediction interface
- **Automated Pipeline**: End-to-end processing from raw data to trained models

## 🏗️ Project Structure

```
AQI_Prediction&Analysis/
├── data/                           # All datasets
│   ├── kaggle_raw/                # Kaggle dataset files
│   │   ├── city_hour.csv          # Hourly city-level data
│   │   ├── city_day.csv           # Daily city-level data
│   │   ├── station_hour.csv       # Hourly station-level data
│   │   ├── station_day.csv        # Daily station-level data
│   │   └── stations.csv           # Station information
│   ├── processed/                 # Cleaned and feature-engineered data
│   │   └── processed_aqi_data.csv
│   └── raw/                      # Symlink to kaggle_raw
├── src/                          # Python source code
│   ├── config.py                # Configuration and settings
│   ├── preprocessing.py         # Data cleaning and feature engineering
│   ├── train_model.py          # ML training pipeline
│   ├── dashboard.py            # Interactive Streamlit dashboard
│   └── imputation.py           # Missing data handling
├── notebooks/                    # Jupyter analysis notebooks
│   └── AQI_Analysis_and_Modeling.ipynb
├── models/                       # Saved model artifacts
├── reports/                      # Generated reports and figures
├── tests/                        # Test suite
├── run_pipeline.sh              # Pipeline automation script
├── setup_dataset.py            # Kaggle credentials setup
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Kaggle account and API credentials
- Git

### Setup Instructions

1. **Clone Repository**:
```bash
git clone <your-repo-url>
cd AQI_Prediction&Analysis
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup Kaggle Credentials**:
```bash
# Interactive setup (recommended)
python setup_dataset.py

# Or manually place kaggle.json in ~/.kaggle/
```

4. **Download Dataset**:
```bash
# Download Kaggle air quality dataset
kaggle datasets download -d rohanrao/air-quality-data-in-india

# Extract to data/kaggle_raw/
unzip air-quality-data-in-india.zip -d data/kaggle_raw/
```

5. **Run Complete Pipeline**:
```bash
# Automated pipeline execution
./run_pipeline.sh

# Or run components individually:
python src/preprocessing.py
python src/train_model.py
```

### What Happens:
1. 📊 **Data Loading**: Processes Kaggle dataset (242K+ records)
2. 🧹 **Data Cleaning**: Standardizes formats and handles missing values
3. ⚙️ **Feature Engineering**: Creates 37+ features including temporal and location features
4. 🤖 **Model Training**: Trains and compares multiple ML models
5. 📈 **Results**: Generates performance metrics and saves trained models

## 📊 Dataset Information

### Kaggle Air Quality Dataset
- **Source**: [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Time Period**: 2015-2020
- **Coverage**: 6 major Indian cities (Ahmedabad, Chennai, Delhi, Hyderabad, Kolkata, Mumbai)
- **Records**: 242,416 processed records
- **Data Types**: Both hourly and daily measurements

### Pollutants Measured
- **Particulate Matter**: PM2.5, PM10
- **Gaseous Pollutants**: NO, NO2, NOx, NH3, CO, SO2, O3
- **Organic Compounds**: Benzene, Toluene, Xylene
- **Air Quality Index**: Pre-calculated AQI values with categories

## 🔍 Analysis & Visualizations

### Interactive Dashboard
```bash
streamlit run src/dashboard.py
```

**Dashboard Features:**
- 📊 **Data Overview**: Dataset statistics and quality metrics
- 🔍 **Exploratory Analysis**: Time series plots, city comparisons, correlation analysis
- 🤖 **Model Training**: Interactive model training and comparison
- 🔮 **Predictions**: AQI prediction interface

### Jupyter Notebook Analysis
```bash
jupyter lab
# Open notebooks/AQI_Analysis_and_Modeling.ipynb
```

**Analysis Includes:**
- Time series patterns and trends
- City-wise pollution comparisons
- Pollutant correlation analysis
- Seasonal and temporal patterns
- Model performance comparison

## 🤖 Machine Learning Models

### Available Models
1. **Linear Regression**: Baseline interpretable model
2. **Random Forest**: Ensemble method with feature importance
3. **XGBoost**: Gradient boosting for high accuracy

### Feature Engineering (37+ Features)
- **Temporal Features**: Hour, day of week, month, season
- **Location Features**: City, coordinates, state information
- **Lag Features**: Previous AQI values (1h, 6h, 12h, 24h)
- **Rolling Statistics**: Moving averages over different windows
- **Categorical Features**: AQI categories, rush hour indicators
- **Pollutant Ratios**: PM2.5/PM10 ratios for enhanced predictions

### Typical Performance
| Model | RMSE | R² | Features Used |
|-------|------|----|----|
| Linear Regression | 35-45 | 0.85-0.90 | All features |
| Random Forest | 25-35 | 0.90-0.95 | Feature selection |
| XGBoost | 20-30 | 0.92-0.97 | Optimized features |

## 🧪 Testing & Quality

Run the test suite to ensure everything works correctly:

```bash
# Run all tests
pytest tests/ -v

# Run specific components
pytest tests/test_preprocessing.py -v
pytest tests/test_train_model.py -v
```

## 🛠️ Configuration

Key configuration options in `src/config.py`:
- **Target Cities**: Focus on specific Indian cities
- **Model Parameters**: Hyperparameters for each ML model
- **Data Paths**: Directory structure configuration
- **AQI Categories**: Standard air quality classifications

## 📈 Results & Insights

### Key Findings
- Delhi consistently shows highest pollution levels
- Strong seasonal patterns with winter months showing elevated AQI
- Rush hour effects visible in hourly data
- PM2.5 and PM10 are strongest predictors of overall AQI

### Model Performance
- XGBoost achieves best accuracy with R² > 0.95
- Feature importance analysis reveals temporal features are crucial
- City-specific patterns improve prediction accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Run tests: `pytest tests/ -v`
4. Submit pull request with clear description

## 🛠️ Troubleshooting

### Common Issues
1. **Kaggle API Issues**: Ensure kaggle.json is properly configured
2. **Memory Issues**: Large dataset may require sufficient RAM (4GB+ recommended)
3. **Missing Data**: Pipeline handles missing values automatically
4. **Port Conflicts**: Use `--server.port` flag for Streamlit if needed

### Getting Help
- Check logs for detailed error messages
- Verify dataset is properly extracted in `data/kaggle_raw/`
- Ensure all dependencies are installed

## 📚 Educational Value

**Perfect for Learning:**
- 🎓 **Data Science**: Complete ML pipeline from data to deployment
- 📊 **Environmental Analysis**: Real-world air quality data insights  
- 🤖 **Machine Learning**: Multi-model comparison and evaluation
- 📈 **Visualization**: Interactive dashboards and analysis

**Use Cases:**
- Academic research projects
- Environmental monitoring systems
- Public health impact studies
- Urban planning and policy making

## 📄 License

MIT License - feel free to use for academic and commercial purposes.

## 🙏 Acknowledgments

- **Kaggle & Data Contributors**: For providing comprehensive air quality dataset
- **Central Pollution Control Board (India)**: Original data source
- **Open Source Community**: For excellent libraries (pandas, scikit-learn, Streamlit)

## 📊 Project Status

✅ **Fully Functional**: Complete end-to-end pipeline  
✅ **Rich Dataset**: 5+ years of historical data  
✅ **Multiple Models**: Comprehensive ML approach  
✅ **Interactive Analysis**: Dashboard and notebooks  
✅ **Well Tested**: Comprehensive test coverage  
✅ **Documentation**: Complete setup guides  

---

## 🚀 **Ready to Analyze Air Quality Data?**

```bash
# Get started in 3 commands:
git clone <your-repo-url>
cd AQI_Prediction&Analysis
./run_pipeline.sh
```

**🌍 Comprehensive Air Quality Analysis with Historical Data**

*End-to-end solution for air quality prediction and analysis using 5+ years of real environmental data from India's major cities.*