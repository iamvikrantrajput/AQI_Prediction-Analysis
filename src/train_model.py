"""
Simplified model training module for Air Quality Prediction project.
Implements basic ML models for AQI prediction - suitable for Master's level.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import joblib
import json
from datetime import datetime
from pathlib import Path

# ML Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from config import MODEL_CONFIG, TRAINING_CONFIG, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, get_model_path

# Simple logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_model")


class SimpleModelTrainer:
    """Simple model training class for Master's level students."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config or TRAINING_CONFIG
        self.model_config = MODEL_CONFIG
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = "aqi",
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: Input dataframe
            target_column: Target variable column name
            test_size: Test set size
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")
        
        # Handle missing data with simple mean imputation
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Remove non-feature columns
        exclude_columns = ["datetime", "city", target_column, "source", "collection_time", "location", "location_id"]
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle categorical variables with simple label encoding
        for col in feature_columns:
            if df[col].dtype == 'object':
                df[col] = pd.Categorical(df[col]).codes
        
        X = df[feature_columns].values
        y = df[target_column].values
        
        self.feature_names = feature_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.config["random_state"]
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        logger.info(f"Data prepared - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
        """Train Linear Regression model."""
        logger.info("Training Linear Regression model...")
        
        config = self.model_config["linear_regression"]
        model = LinearRegression(**config)
        model.fit(X_train, y_train)
        
        self.models["linear_regression"] = model
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        config = self.model_config["random_forest"]
        model = RandomForestRegressor(**config)
        model.fit(X_train, y_train)
        
        self.models["random_forest"] = model
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        config = self.model_config["xgboost"]
        model = xgb.XGBRegressor(**config)
        model.fit(X_train, y_train)
        
        self.models["xgboost"] = model
        return model
    
    def evaluate_model(
        self,
        model: Any,
        model_name: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            model_name: Name of the model
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (avoiding division by zero)
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape
        }
        
        logger.info(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        return metrics
    
    def cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> Dict[str, float]:
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv_folds: Number of cross-validation folds
        
        Returns:
            Cross-validation scores
        """
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                   scoring='neg_mean_squared_error')
        
        return {
            "cv_rmse_mean": np.sqrt(-cv_scores.mean()),
            "cv_rmse_std": np.sqrt(cv_scores.std()),
        }
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metrics: Optional[Dict] = None,
        version: Optional[str] = None,
    ) -> Path:
        """
        Save trained model.
        
        Args:
            model: Trained model
            model_name: Name of the model
            metrics: Model performance metrics
            version: Model version
        
        Returns:
            Path where model was saved
        """
        # Create models directory if it doesn't exist
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = get_model_path("regression", model_name, version)
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = model_path.with_name(f"{model_name}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version or "v1.0.0",
            "created_at": datetime.now().isoformat(),
            "feature_names": self.feature_names,
            "metrics": metrics,
            "scaler_path": str(scaler_path),
        }
        
        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def train_all_models(
        self,
        df: pd.DataFrame,
        target_column: str = "aqi",
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all specified models.
        
        Args:
            df: Input dataframe
            target_column: Target variable
            models_to_train: List of models to train
        
        Returns:
            Dictionary with trained models and their metrics
        """
        if models_to_train is None:
            models_to_train = ["linear_regression", "random_forest", "xgboost"]
        
        logger.info(f"Training models: {models_to_train}")
        
        # Prepare data once for all models
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_column)
        X_all = np.vstack([X_train, X_test])
        y_all = np.hstack([y_train, y_test])
        
        results = {}
        
        for model_name in models_to_train:
            try:
                logger.info(f"Training {model_name}...")
                
                # Train model
                if model_name == "linear_regression":
                    model = self.train_linear_regression(X_train, y_train)
                elif model_name == "random_forest":
                    model = self.train_random_forest(X_train, y_train)
                elif model_name == "xgboost":
                    model = self.train_xgboost(X_train, y_train)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                # Evaluate model
                metrics = self.evaluate_model(model, model_name, X_test, y_test)
                
                # Cross-validation
                cv_metrics = self.cross_validate_model(model, X_all, y_all)
                metrics.update(cv_metrics)
                
                # Save model
                model_path = self.save_model(model, model_name, metrics)
                
                results[model_name] = {
                    "model": model,
                    "metrics": metrics,
                    "model_path": model_path,
                    "feature_names": self.feature_names
                }
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        # Generate and save comprehensive results report
        if results:
            self._save_training_results(results, df, target_column, X_test, y_test)
        
        return results
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Any, StandardScaler]:
        """
        Load a saved model and its scaler.
        
        Args:
            model_name: Name of the model
            version: Model version
        
        Returns:
            Tuple of (model, scaler)
        """
        model_path = get_model_path("regression", model_name, version)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Load scaler
        scaler_path = model_path.with_name(f"{model_name}_scaler.pkl")
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    def predict(
        self,
        model_name: str,
        X: np.ndarray,
        version: Optional[str] = None,
    ) -> np.ndarray:
        """
        Make predictions using a saved model.
        
        Args:
            model_name: Name of the model
            X: Features to predict on
            version: Model version
        
        Returns:
            Predictions
        """
        model, scaler = self.load_model(model_name, version)
        X_scaled = scaler.transform(X)
        return model.predict(X_scaled)

    def _save_training_results(
        self, 
        results: Dict[str, Dict[str, Any]], 
        df: pd.DataFrame, 
        target_column: str,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> None:
        """Save comprehensive training results with plots and metrics."""
        # Create directories
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save metrics comparison
        metrics_df = pd.DataFrame({
            model_name: result['metrics'] for model_name, result in results.items()
        }).T
        
        metrics_path = REPORTS_DIR / f"model_comparison_{timestamp}.csv"
        metrics_df.to_csv(metrics_path)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # 2. Create and save plots
        self._create_performance_plots(results, X_test, y_test, timestamp)
        
        # 3. Create feature importance plots
        self._create_feature_importance_plots(results, timestamp)
        
        # 4. Save comprehensive report
        self._create_comprehensive_report(results, df, target_column, timestamp)
    
    def _create_performance_plots(
        self, 
        results: Dict[str, Dict[str, Any]], 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        timestamp: str
    ) -> None:
        """Create performance comparison plots."""
        plt.style.use('seaborn-v0_8')
        
        # Model comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        model_names = list(results.keys())
        metrics = ['rmse', 'mae', 'r2', 'mape']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [results[model]['metrics'][metric] for model in model_names]
            
            bars = ax.bar(model_names, values, 
                         color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
            ax.set_title(f'{metric.upper()} Comparison', fontweight='bold')
            ax.set_ylabel(metric.upper())
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = FIGURES_DIR / f"model_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Performance plots saved to: {plot_path}")
        
        # Prediction vs Actual plots
        fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            model = result['model']
            y_pred = model.predict(X_test)
            
            ax = axes[idx]
            ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                   'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual AQI')
            ax.set_ylabel('Predicted AQI')
            ax.set_title(f'{model_name.replace("_", " ").title()}\nR² = {result["metrics"]["r2"]:.3f}')
            ax.legend()
        
        plt.tight_layout()
        pred_plot_path = FIGURES_DIR / f"prediction_vs_actual_{timestamp}.png"
        plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Prediction plots saved to: {pred_plot_path}")
    
    def _create_feature_importance_plots(
        self, 
        results: Dict[str, Dict[str, Any]], 
        timestamp: str
    ) -> None:
        """Create feature importance plots for tree-based models."""
        tree_models = {k: v for k, v in results.items() 
                      if k in ['random_forest', 'xgboost'] and hasattr(v['model'], 'feature_importances_')}
        
        if not tree_models:
            return
        
        fig, axes = plt.subplots(1, len(tree_models), figsize=(8*len(tree_models), 6))
        if len(tree_models) == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(tree_models.items()):
            model = result['model']
            feature_names = result['feature_names']
            
            # Get top 15 features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True).tail(15)
            
            ax = axes[idx]
            ax.barh(importance_df['feature'], importance_df['importance'], color='lightblue')
            ax.set_title(f'{model_name.replace("_", " ").title()}\nFeature Importance (Top 15)')
            ax.set_xlabel('Importance')
            
            # Rotate labels if needed
            plt.setp(ax.get_yticklabels(), rotation=0, ha='right')
        
        plt.tight_layout()
        importance_path = FIGURES_DIR / f"feature_importance_{timestamp}.png"
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance plots saved to: {importance_path}")
    
    def _create_comprehensive_report(
        self, 
        results: Dict[str, Dict[str, Any]], 
        df: pd.DataFrame, 
        target_column: str,
        timestamp: str
    ) -> None:
        """Create comprehensive HTML report."""
        
        # Create HTML report content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AQI Model Training Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; color: #2c3e50; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ 
                    border: 1px solid #ddd; padding: 8px; text-align: center; 
                }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .best-model {{ background-color: #d4edda; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🌍 AQI Model Training Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Dataset Summary</h2>
                <ul>
                    <li><strong>Total Records:</strong> {len(df):,}</li>
                    <li><strong>Target Variable:</strong> {target_column}</li>
                    <li><strong>Features:</strong> {len(self.feature_names)}</li>
                    <li><strong>Cities:</strong> {df['city'].nunique() if 'city' in df.columns else 'N/A'}</li>
                    <li><strong>Date Range:</strong> {df['datetime'].min()} to {df['datetime'].max() if 'datetime' in df.columns else 'N/A'}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🤖 Model Performance Comparison</h2>
                <table class="metrics-table">
                    <tr>
                        <th>Model</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>R²</th>
                        <th>MAPE (%)</th>
                        <th>Cross-Val RMSE</th>
                    </tr>
        """
        
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['metrics']['r2'])
        
        for model_name, result in results.items():
            metrics = result['metrics']
            row_class = 'best-model' if model_name == best_model else ''
            cv_rmse = f"{metrics.get('cv_rmse_mean', 0):.2f} ± {metrics.get('cv_rmse_std', 0):.2f}"
            
            html_content += f"""
                    <tr class="{row_class}">
                        <td><strong>{model_name.replace('_', ' ').title()}</strong></td>
                        <td>{metrics['rmse']:.3f}</td>
                        <td>{metrics['mae']:.3f}</td>
                        <td>{metrics['r2']:.3f}</td>
                        <td>{metrics['mape']:.2f}</td>
                        <td>{cv_rmse}</td>
                    </tr>
            """
        
        html_content += f"""
                </table>
                <p><em>🏆 Best Model: <strong>{best_model.replace('_', ' ').title()}</strong> (R² = {results[best_model]['metrics']['r2']:.3f})</em></p>
            </div>
            
            <div class="section">
                <h2>📈 Visualizations</h2>
                <h3>Model Performance Comparison</h3>
                <img src="../figures/model_comparison_{timestamp}.png" alt="Model Comparison">
                
                <h3>Prediction vs Actual</h3>
                <img src="../figures/prediction_vs_actual_{timestamp}.png" alt="Prediction vs Actual">
                
                <h3>Feature Importance</h3>
                <img src="../figures/feature_importance_{timestamp}.png" alt="Feature Importance">
            </div>
            
            <div class="section">
                <h2>💡 Key Insights</h2>
                <ul>
                    <li>Best performing model: <strong>{best_model.replace('_', ' ').title()}</strong></li>
                    <li>R² Score: <strong>{results[best_model]['metrics']['r2']:.3f}</strong></li>
                    <li>RMSE: <strong>{results[best_model]['metrics']['rmse']:.2f}</strong> AQI units</li>
                    <li>Mean Absolute Error: <strong>{results[best_model]['metrics']['mae']:.2f}</strong> AQI units</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>🔧 Technical Details</h2>
                <ul>
                    <li><strong>Training Method:</strong> Train-Test Split (80/20)</li>
                    <li><strong>Cross-Validation:</strong> 5-Fold CV</li>
                    <li><strong>Feature Scaling:</strong> StandardScaler</li>
                    <li><strong>Missing Value Handling:</strong> Mean Imputation</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        report_path = REPORTS_DIR / f"training_report_{timestamp}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive report saved to: {report_path}")


def main():
    """Main function for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Air Quality Model Training")
    parser.add_argument("--data-path", required=True, help="Path to preprocessed data")
    parser.add_argument("--target", default="aqi", help="Target variable column name")
    parser.add_argument("--models", nargs="+", 
                       default=["linear_regression", "random_forest", "xgboost"], 
                       help="Models to train")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    
    # Initialize trainer
    trainer = SimpleModelTrainer()
    
    # Train models
    results = trainer.train_all_models(
        df=df,
        target_column=args.target,
        models_to_train=args.models,
    )
    
    # Print results
    print("\nModel Training Results:")
    print("-" * 50)
    for model_name, result in results.items():
        metrics = result["metrics"]
        print(f"{model_name.replace('_', ' ').title()}:")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        if "cv_rmse_mean" in metrics:
            print(f"  Cross-Val RMSE: {metrics['cv_rmse_mean']:.2f} ± {metrics['cv_rmse_std']:.2f}")
        print(f"  Model saved to: {result['model_path']}")
        print()


if __name__ == "__main__":
    main()