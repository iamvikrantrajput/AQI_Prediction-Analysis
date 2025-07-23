#!/bin/bash

# Air Quality Prediction Pipeline Automation Script
# This script runs the complete end-to-end pipeline

set -e  # Exit on any error

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Main pipeline execution
main() {
    echo "🚀 Air Quality Prediction Pipeline"
    echo "=================================="
    echo ""
    
    # Step 1: Verify Kaggle Data Available
    log "🔄 Step 1: Verifying Kaggle Dataset"
    
    if [ ! -d "data/kaggle_raw" ] || [ ! -f "data/kaggle_raw/city_hour.csv" ]; then
        warning "Kaggle dataset not found in data/kaggle_raw/"
        warning "Please download the Kaggle Air Quality dataset first"
        warning "Run: kaggle datasets download -d rohanrao/air-quality-data-in-india"
        exit 1
    fi
    
    success "Kaggle dataset verified"
    
    # Step 2: Data Preprocessing
    log "🔄 Step 2: Data Preprocessing"
    cd src
    python3 preprocessing.py || { echo "Preprocessing failed"; exit 1; }
    cd ..
    success "Data preprocessing completed"
    
    # Step 3: Model Training
    log "🔄 Step 3: Model Training"
    cd src
    python3 train_model.py --data-path "../data/processed/processed_aqi_data.csv" --target "aqi" || { echo "Training failed"; exit 1; }
    cd ..
    success "Model training completed"
    
    echo ""
    success "🎉 Pipeline completed successfully!"
    
    # Show results
    log "📊 Results in:"
    echo "   📁 data/processed/ - Processed datasets"
    echo "   🤖 models/ - Trained models"
    echo "   📈 reports/ - Analysis reports"
}

# Run main function
main "$@"