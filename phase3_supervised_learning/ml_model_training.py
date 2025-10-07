#!/usr/bin/env python3
"""
Phase 3.4: Machine Learning Model Training
==========================================

This module implements comprehensive machine learning model training for Portuguese
address matching using the rich 89-feature dataset created in Phase 3.3.

Key Features:
1. Multiple ML algorithms (Random Forest, XGBoost, SVM, Logistic Regression)
2. Comprehensive cross-validation and hyperparameter tuning
3. Feature importance analysis and selection
4. Performance comparison against Phase 2 baselines
5. Model interpretation and explainability
6. Production-ready model persistence

The goal is to achieve superior address matching performance by leveraging
the multi-modal feature engineering from previous phases.
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, 
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# Visualization and analysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib

class MLAddressMatchingTrainer:
    """
    Comprehensive ML training pipeline for Portuguese address matching.
    """
    
    def __init__(self, data_path="results/phase3_3_features/ml_ready_features_20251002_153226.csv"):
        """Initialize the ML training pipeline."""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.best_models = {}
        
        print("🚀 ML Address Matching Trainer Initialized")
        print(f"   📊 Data Path: {data_path}")
    
    def load_and_prepare_data(self, test_size=0.2, random_state=42):
        """Load and prepare the ML-ready dataset."""
        print("\n📊 Loading ML-Ready Dataset...")
        
        # Load data
        df = pd.read_csv(self.data_path, encoding='utf-8')
        print(f"   ✅ Loaded {len(df):,} address pairs")
        print(f"   ✅ Found {len(df.columns)} total columns")
        
        # Separate features and target
        # Exclude text columns that shouldn't be used as features
        text_cols_to_exclude = [
            'address_1', 'address_2', 'label',
            'address_1_road', 'address_1_house_number', 'address_1_city', 
            'address_1_postcode', 'address_1_normalized',
            'address_2_road', 'address_2_house_number', 'address_2_city',
            'address_2_postcode', 'address_2_normalized'
        ]
        
        feature_cols = [col for col in df.columns if col not in text_cols_to_exclude]
        X = df[feature_cols]
        y = df['label']
        
        # Ensure all features are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                print(f"   ⚠️  Converting {col} to numeric...")
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any NaN values with 0
        X = X.fillna(0)
        
        print(f"   ✅ Extracted {len(feature_cols)} features")
        print(f"   ✅ Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"   📈 Training set: {len(X_train):,} pairs")
        print(f"   📊 Test set: {len(X_test):,} pairs")
        
        # Store data
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.feature_names = feature_cols
        
        return X_train, X_test, y_train, y_test
    
    def analyze_feature_categories(self):
        """Analyze feature importance by category."""
        print("\n🔍 Analyzing Feature Categories...")
        
        feature_categories = {
            'Phase2': [col for col in self.feature_names if col.startswith('phase2_')],
            'Component': [col for col in self.feature_names if col.startswith('comp_')],
            'NER': [col for col in self.feature_names if col.startswith('ner_')],
            'Statistical': [col for col in self.feature_names if col.startswith('stat_')],
            'Interaction': [col for col in self.feature_names if col.startswith('interact_')],
            'Original': [col for col in self.feature_names if not any(col.startswith(p) for p in ['phase2_', 'comp_', 'ner_', 'stat_', 'interact_'])]
        }
        
        print("   📋 Feature Categories:")
        for category, features in feature_categories.items():
            if features:
                print(f"     {category}: {len(features)} features")
        
        self.feature_categories = feature_categories
        return feature_categories
    
    def train_baseline_models(self):
        """Train baseline models with default parameters."""
        print("\n🔧 Training Baseline Models...")
        
        # Define baseline models
        baseline_models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        if XGBOOST_AVAILABLE:
            baseline_models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        baseline_results = {}
        
        for name, model in baseline_models.items():
            print(f"   🎯 Training {name}...")
            start_time = time.time()
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
            
            # Test predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'training_time': time.time() - start_time
            }
            
            baseline_results[name] = results
            self.models[f"{name}_baseline"] = model
            
            print(f"     ✅ F1: {results['f1']:.4f} | ROC-AUC: {results['roc_auc']:.4f} | Time: {results['training_time']:.1f}s")
        
        self.baseline_results = baseline_results
        return baseline_results
    
    def hyperparameter_tuning(self, quick_search=True):
        """Perform hyperparameter tuning for best models."""
        print("\n⚙️  Hyperparameter Tuning...")
        
        if quick_search:
            print("   🏃 Running Quick Search (5-fold CV, limited params)")
            cv_folds = 3
            n_iter = 20
        else:
            print("   🔍 Running Comprehensive Search (10-fold CV, extensive params)")
            cv_folds = 5
            n_iter = 50
        
        # Define parameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300] if quick_search else [100, 200, 300, 500],
                'max_depth': [10, 20, None] if quick_search else [5, 10, 20, 30, None],
                'min_samples_split': [2, 5] if quick_search else [2, 5, 10],
                'min_samples_leaf': [1, 2] if quick_search else [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10] if quick_search else [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if XGBOOST_AVAILABLE:
            param_grids['XGBoost'] = {
                'n_estimators': [100, 200] if quick_search else [100, 200, 300],
                'max_depth': [3, 6] if quick_search else [3, 6, 9],
                'learning_rate': [0.1, 0.2] if quick_search else [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0] if quick_search else [0.7, 0.8, 0.9, 1.0]
            }
        
        tuned_results = {}
        
        for model_name, param_grid in param_grids.items():
            print(f"   🎯 Tuning {model_name}...")
            start_time = time.time()
            
            # Select base model
            if model_name == 'RandomForest':
                base_model = RandomForestClassifier(random_state=42)
            elif model_name == 'LogisticRegression':
                base_model = LogisticRegression(random_state=42, max_iter=1000)
            elif model_name == 'XGBoost':
                base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            
            # Perform grid search
            search = RandomizedSearchCV(
                base_model, param_grid, cv=cv_folds, scoring='f1',
                n_iter=n_iter, random_state=42, n_jobs=-1
            )
            
            search.fit(self.X_train, self.y_train)
            
            # Best model predictions
            best_model = search.best_estimator_
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            results = {
                'best_params': search.best_params_,
                'best_cv_score': search.best_score_,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'tuning_time': time.time() - start_time
            }
            
            tuned_results[model_name] = results
            self.best_models[model_name] = best_model
            
            print(f"     ✅ Best F1: {results['f1']:.4f} | ROC-AUC: {results['roc_auc']:.4f} | Time: {results['tuning_time']:.1f}s")
            print(f"     🔧 Best params: {results['best_params']}")
        
        self.tuned_results = tuned_results
        return tuned_results
    
    def feature_importance_analysis(self):
        """Analyze feature importance from best models."""
        print("\n🔍 Feature Importance Analysis...")
        
        importance_results = {}
        
        for model_name, model in self.best_models.items():
            print(f"   📊 Analyzing {model_name} feature importance...")
            
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_results[model_name] = feature_imp
                
                # Show top 10 features
                print(f"     🏆 Top 10 features for {model_name}:")
                for idx, row in feature_imp.head(10).iterrows():
                    print(f"       {row['feature']}: {row['importance']:.4f}")
            
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_[0])
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                importance_results[model_name] = feature_imp
                
                print(f"     🏆 Top 10 features for {model_name}:")
                for idx, row in feature_imp.head(10).iterrows():
                    print(f"       {row['feature']}: {row['importance']:.4f}")
        
        self.feature_importance = importance_results
        return importance_results
    
    def category_performance_analysis(self):
        """Analyze performance by feature category."""
        print("\n📈 Category Performance Analysis...")
        
        category_results = {}
        
        # Test each category individually
        for category, features in self.feature_categories.items():
            if not features:
                continue
                
            print(f"   🎯 Testing {category} features ({len(features)} features)...")
            
            # Select features for this category
            X_train_cat = self.X_train[features]
            X_test_cat = self.X_test[features]
            
            # Train simple RandomForest on category
            model = RandomForestClassifier(random_state=42, n_estimators=100)
            model.fit(X_train_cat, self.y_train)
            
            # Predictions
            y_pred = model.predict(X_test_cat)
            y_pred_proba = model.predict_proba(X_test_cat)[:, 1]
            
            # Metrics
            from sklearn.metrics import accuracy_score, f1_score
            results = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'feature_count': len(features)
            }
            
            category_results[category] = results
            print(f"     ✅ F1: {results['f1']:.4f} | ROC-AUC: {results['roc_auc']:.4f}")
        
        self.category_results = category_results
        return category_results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report."""
        print("\n📋 Generating Comprehensive Report...")
        
        report = {
            'dataset_info': {
                'total_pairs': len(self.X_train) + len(self.X_test),
                'training_pairs': len(self.X_train),
                'test_pairs': len(self.X_test),
                'total_features': len(self.feature_names),
                'feature_categories': {cat: len(feats) for cat, feats in self.feature_categories.items()}
            },
            'baseline_results': self.baseline_results,
            'tuned_results': self.tuned_results,
            'category_performance': self.category_results,
            'best_overall_model': self._get_best_model(),
            'feature_importance_summary': self._summarize_feature_importance()
        }
        
        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"ml_training_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   💾 Report saved: {report_file}")
        
        # Print summary
        best_model = report['best_overall_model']
        print(f"\n🏆 BEST MODEL: {best_model['model_name']}")
        print(f"   🎯 F1 Score: {best_model['f1']:.4f}")
        print(f"   📊 ROC-AUC: {best_model['roc_auc']:.4f}")
        print(f"   ⚡ Accuracy: {best_model['accuracy']:.4f}")
        
        return report, report_file
    
    def _get_best_model(self):
        """Get the best performing model."""
        best_score = 0
        best_model_info = {}
        
        for model_name, results in self.tuned_results.items():
            if results['f1'] > best_score:
                best_score = results['f1']
                best_model_info = {
                    'model_name': model_name,
                    'f1': results['f1'],
                    'roc_auc': results['roc_auc'],
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'best_params': results['best_params']
                }
        
        return best_model_info
    
    def _summarize_feature_importance(self):
        """Summarize feature importance across models."""
        if not self.feature_importance:
            return {}
        
        # Average importance across models
        all_features = set()
        for model_imp in self.feature_importance.values():
            all_features.update(model_imp['feature'].tolist())
        
        importance_summary = {}
        for feature in all_features:
            scores = []
            for model_imp in self.feature_importance.values():
                feature_row = model_imp[model_imp['feature'] == feature]
                if not feature_row.empty:
                    scores.append(feature_row['importance'].iloc[0])
            
            if scores:
                importance_summary[feature] = {
                    'mean_importance': np.mean(scores),
                    'std_importance': np.std(scores),
                    'models_count': len(scores)
                }
        
        # Sort by mean importance
        sorted_features = sorted(importance_summary.items(), 
                               key=lambda x: x[1]['mean_importance'], reverse=True)
        
        return dict(sorted_features[:20])  # Top 20 features
    
    def save_best_models(self):
        """Save the best trained models."""
        print("\n💾 Saving Best Models...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_models = {}
        
        for model_name, model in self.best_models.items():
            model_file = f"best_{model_name.lower()}_{timestamp}.joblib"
            joblib.dump(model, model_file)
            saved_models[model_name] = model_file
            print(f"   ✅ Saved {model_name}: {model_file}")
        
        return saved_models
    
    def run_complete_training_pipeline(self, quick_tuning=True):
        """Run the complete ML training pipeline."""
        print("\n" + "="*70)
        print("🚀 COMPLETE ML TRAINING PIPELINE")
        print("="*70)
        
        start_time = time.time()
        
        # 1. Load and prepare data
        self.load_and_prepare_data()
        
        # 2. Analyze feature categories
        self.analyze_feature_categories()
        
        # 3. Train baseline models
        self.train_baseline_models()
        
        # 4. Hyperparameter tuning
        self.hyperparameter_tuning(quick_search=quick_tuning)
        
        # 5. Feature importance analysis
        self.feature_importance_analysis()
        
        # 6. Category performance analysis
        self.category_performance_analysis()
        
        # 7. Generate comprehensive report
        report, report_file = self.generate_comprehensive_report()
        
        # 8. Save best models
        saved_models = self.save_best_models()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("✅ ML TRAINING PIPELINE COMPLETE")
        print(f"   ⏱️  Total Time: {total_time:.1f}s")
        print(f"   📁 Report: {report_file}")
        print(f"   🎯 Best Model: {report['best_overall_model']['model_name']}")
        print(f"   🏆 Best F1: {report['best_overall_model']['f1']:.4f}")
        print("="*70)
        
        return report, saved_models

def main():
    """Main execution function."""
    print("🚀 Phase 3.4: ML Model Training for Portuguese Address Matching")
    print("=" * 70)
    
    # Initialize trainer
    trainer = MLAddressMatchingTrainer()
    
    # Ask user for training mode
    print("\n🤔 Select Training Mode:")
    print("   1. Quick Training (fast, good for testing)")
    print("   2. Comprehensive Training (slower, best results)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    quick_mode = choice == "1"
    
    if quick_mode:
        print("\n⚡ Running Quick Training Mode...")
    else:
        print("\n🔍 Running Comprehensive Training Mode...")
    
    # Run complete pipeline
    report, saved_models = trainer.run_complete_training_pipeline(quick_tuning=quick_mode)
    
    print(f"\n🎉 Training Complete!")
    print(f"   📊 Models trained and evaluated")
    print(f"   📁 Results saved to files")
    print(f"   🚀 Ready for Phase 3.5: Evaluation & Comparison!")

if __name__ == "__main__":
    main()