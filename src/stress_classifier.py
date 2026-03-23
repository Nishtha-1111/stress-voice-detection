# """
# stress_classifier.py
# Purpose: Train and manage stress detection models
# - Random Forest
# - SVM
# - Gradient Boosting
# - Model selection based on F1-score
# - Probability calibration
# """

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
#                            f1_score, roc_auc_score, confusion_matrix,
#                            classification_report)
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline
# import joblib
# import json
# import os
# from datetime import datetime

# class StressClassifier:
#     """
#     Manages stress detection models with probability calibration
#     """
    
#     def __init__(self, random_state=42):
#         """
#         Initialize classifier manager
        
#         Args:
#             random_state (int): Random seed for reproducibility
#         """
#         self.random_state = random_state
#         self.models = {}
#         self.best_model = None
#         self.best_model_name = None
#         self.cv_folds = 5
#         self.scaler = None
        
#     def get_model_params(self):
#         """
#         Define hyperparameter grids for each model
        
#         Returns:
#             dict: Model names and their parameter grids
#         """
#         param_grids = {
#             'random_forest': {
#                 'n_estimators': [100, 200, 300],
#                 'max_depth': [10, 20, 30, None],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4],
#                 'class_weight': ['balanced', None]
#             },
            
#             'svm': {
#                 'C': [0.1, 1, 10, 100],
#                 'gamma': ['scale', 'auto', 0.1, 0.01],
#                 'kernel': ['rbf'],
#                 'class_weight': ['balanced', None],
#                 'probability': [True]  # Need probability for calibration
#             },
            
#             'gradient_boosting': {
#                 'n_estimators': [100, 200, 300],
#                 'max_depth': [3, 5, 7, 10],
#                 'learning_rate': [0.01, 0.1, 0.2],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4]
#             }
#         }
        
#         return param_grids
    
#     def create_models(self):
#         """
#         Create base model instances
        
#         Returns:
#             dict: Model names and their instances
#         """
#         models = {
#             'random_forest': RandomForestClassifier(
#                 random_state=self.random_state,
#                 n_jobs=-1
#             ),
            
#             'svm': SVC(
#                 random_state=self.random_state,
#                 cache_size=1000,
#                 probability=True  # Enable probability estimates
#             ),
            
#             'gradient_boosting': GradientBoostingClassifier(
#                 random_state=self.random_state
#             )
#         }
        
#         return models
    
#     def train_with_gridsearch(self, X_train, y_train, model_name, cv=5):
#         """
#         Train a specific model with grid search
        
#         Args:
#             X_train (np.array): Training features
#             y_train (np.array): Training labels
#             model_name (str): Name of model to train
#             cv (int): Number of cross-validation folds
            
#         Returns:
#             tuple: (best_model, best_params, best_score)
#         """
#         print(f"\n🎯 Training {model_name} with GridSearchCV...")
        
#         # Get models and parameters
#         models = self.create_models()
#         param_grids = self.get_model_params()
        
#         if model_name not in models:
#             raise ValueError(f"Unknown model: {model_name}")
        
#         model = models[model_name]
#         param_grid = param_grids[model_name]
        
#         # Create grid search object
#         grid_search = GridSearchCV(
#             estimator=model,
#             param_grid=param_grid,
#             cv=cv,
#             scoring='f1_weighted',
#             n_jobs=-1,
#             verbose=1,
#             return_train_score=True
#         )
        
#         # Fit grid search
#         grid_search.fit(X_train, y_train)
        
#         print(f"\n✅ Best parameters for {model_name}:")
#         for param, value in grid_search.best_params_.items():
#             print(f"   {param}: {value}")
#         print(f"   Best F1-score: {grid_search.best_score_:.4f}")
        
#         return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
#     def apply_smote(self, X, y):
#         """
#         Apply SMOTE to handle class imbalance
        
#         Args:
#             X (np.array): Features
#             y (np.array): Labels
            
#         Returns:
#             tuple: (X_resampled, y_resampled)
#         """
#         print("\n🔄 Applying SMOTE for class imbalance...")
        
#         # Count before SMOTE
#         unique, counts = np.unique(y, return_counts=True)
#         print(f"   Before SMOTE - No Stress: {counts[0]}, Stress: {counts[1]}")
        
#         # Apply SMOTE
#         smote = SMOTE(random_state=self.random_state)
#         X_resampled, y_resampled = smote.fit_resample(X, y)
        
#         # Count after SMOTE
#         unique, counts = np.unique(y_resampled, return_counts=True)
#         print(f"   After SMOTE  - No Stress: {counts[0]}, Stress: {counts[1]}")
        
#         return X_resampled, y_resampled
    
#     def calibrate_probabilities(self, model, X_calib, y_calib):
#         """
#         Calibrate model probabilities using sigmoid method
        
#         Args:
#             model: Trained model
#             X_calib (np.array): Calibration features
#             y_calib (np.array): Calibration labels
            
#         Returns:
#             CalibratedClassifierCV: Calibrated model
#         """
#         print("\n📊 Calibrating probabilities...")
        
#         calibrated_model = CalibratedClassifierCV(
#             estimator=model,
#             method='sigmoid',  # Platt scaling
#             cv='prefit'  # Use pre-fitted model
#         )
        
#         calibrated_model.fit(X_calib, y_calib)
        
#         print("✅ Probability calibration complete")
        
#         return calibrated_model
    
#     def evaluate_model(self, model, X_test, y_test, model_name="Model"):
#         """
#         Comprehensive model evaluation
        
#         Args:
#             model: Trained model
#             X_test (np.array): Test features
#             y_test (np.array): Test labels
#             model_name (str): Name of model for printing
            
#         Returns:
#             dict: Evaluation metrics
#         """
#         print(f"\n📈 Evaluating {model_name}...")
        
#         # Get predictions
#         y_pred = model.predict(X_test)
        
#         # Get probabilities (if available)
#         if hasattr(model, 'predict_proba'):
#             y_proba = model.predict_proba(X_test)
#         else:
#             y_proba = None
        
#         # Calculate metrics
#         metrics = {
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred, average='weighted'),
#             'recall': recall_score(y_test, y_pred, average='weighted'),
#             'f1_score': f1_score(y_test, y_pred, average='weighted'),
#             'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
#         }
        
#         # Calculate ROC-AUC if probabilities available
#         if y_proba is not None and len(np.unique(y_test)) == 2:
#             metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        
#         # Print results
#         print(f"\n{model_name} Performance:")
#         print(f"   Accuracy:  {metrics['accuracy']:.4f}")
#         print(f"   Precision: {metrics['precision']:.4f}")
#         print(f"   Recall:    {metrics['recall']:.4f}")
#         print(f"   F1-Score:  {metrics['f1_score']:.4f}")
#         if 'roc_auc' in metrics:
#             print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
#         print("\nConfusion Matrix:")
#         print(metrics['confusion_matrix'])
        
#         # Detailed classification report
#         print("\nDetailed Classification Report:")
#         print(classification_report(y_test, y_pred, 
#                                    target_names=['No Stress', 'Stress']))
        
#         return metrics
    
#     def train_all_models(self, X_train, y_train, X_val, y_val, use_smote=True):
#         """
#         Train all models and select the best one
        
#         Args:
#             X_train (np.array): Training features
#             y_train (np.array): Training labels
#             X_val (np.array): Validation features
#             y_val (np.array): Validation labels
#             use_smote (bool): Whether to apply SMOTE
            
#         Returns:
#             dict: Training results for all models
#         """
#         print("\n" + "="*60)
#         print("🚀 TRAINING ALL MODELS")
#         print("="*60)
        
#         # Apply SMOTE if requested
#         if use_smote:
#             X_train_balanced, y_train_balanced = self.apply_smote(X_train, y_train)
#         else:
#             X_train_balanced, y_train_balanced = X_train, y_train
        
#         results = {}
#         best_f1 = 0
        
#         # Train each model
#         for model_name in ['random_forest', 'svm', 'gradient_boosting']:
#             print("\n" + "-"*40)
            
#             # Train with grid search
#             best_model, best_params, cv_score = self.train_with_gridsearch(
#                 X_train_balanced, y_train_balanced, model_name, cv=self.cv_folds
#             )
            
#             # Calibrate probabilities
#             calibrated_model = self.calibrate_probabilities(best_model, X_val, y_val)
            
#             # Evaluate on validation set
#             metrics = self.evaluate_model(calibrated_model, X_val, y_val, model_name)
            
#             # Store results
#             results[model_name] = {
#                 'model': calibrated_model,
#                 'best_params': best_params,
#                 'cv_score': cv_score,
#                 'validation_metrics': metrics,
#                 'model_size': self.estimate_model_size(calibrated_model)
#             }
            
#             # Track best model based on F1-score
#             if metrics['f1_score'] > best_f1:
#                 best_f1 = metrics['f1_score']
#                 self.best_model = calibrated_model
#                 self.best_model_name = model_name
        
#         print("\n" + "="*60)
#         print(f"🏆 Best Model: {self.best_model_name}")
#         print(f"   F1-Score: {best_f1:.4f}")
#         print("="*60)
        
#         return results
    
#     def estimate_model_size(self, model):
#         """
#         Estimate model size in MB
        
#         Args:
#             model: Trained model
            
#         Returns:
#             float: Estimated size in MB
#         """
#         import sys
#         import pickle
        
#         # Get size by pickling
#         size_bytes = len(pickle.dumps(model))
#         size_mb = size_bytes / (1024 * 1024)
        
#         return size_mb
    
#     def save_model(self, model, scaler, filepath, performance_summary=None):
#         """
#         Save model and scaler to disk
        
#         Args:
#             model: Trained model
#             scaler: Fitted scaler
#             filepath (str): Path to save model
#             performance_summary (dict): Model performance metrics
#         """
#         print(f"\n💾 Saving model to {filepath}")
        
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
#         # Save model
#         model_path = filepath.replace('.pkl', '') + '_model.pkl'
#         joblib.dump(model, model_path)
        
#         # Save scaler
#         scaler_path = filepath.replace('.pkl', '') + '_scaler.pkl'
#         joblib.dump(scaler, scaler_path)
        
#         # Save performance summary
#         if performance_summary:
#             summary_path = filepath.replace('.pkl', '') + '_summary.json'
#             with open(summary_path, 'w') as f:
#                 json.dump(performance_summary, f, indent=4)
        
#         # Check model size
#         model_size = os.path.getsize(model_path) / (1024 * 1024)
#         print(f"   Model size: {model_size:.2f} MB")
        
#         if model_size > 100:
#             print("   ⚠️  Warning: Model size exceeds 100MB limit!")
        
#         return {
#             'model_path': model_path,
#             'scaler_path': scaler_path,
#             'model_size_mb': model_size
#         }
    
#     def load_model(self, model_path, scaler_path):
#         """
#         Load saved model and scaler
        
#         Args:
#             model_path (str): Path to saved model
#             scaler_path (str): Path to saved scaler
            
#         Returns:
#             tuple: (model, scaler)
#         """
#         print(f"\n📂 Loading model from {model_path}")
        
#         model = joblib.load(model_path)
#         scaler = joblib.load(scaler_path)
        
#         print("✅ Model loaded successfully")
        
#         return model, scaler
    
#     def predict_with_confidence(self, model, scaler, features):
#         """
#         Make prediction with confidence score
        
#         Args:
#             model: Trained model
#             scaler: Fitted scaler
#             features (np.array): Feature vector
            
#         Returns:
#             tuple: (prediction, confidence, probabilities)
#         """
#         # Scale features
#         features_scaled = scaler.transform(features.reshape(1, -1))
        
#         # Get prediction
#         prediction = model.predict(features_scaled)[0]
        
#         # Get probabilities
#         if hasattr(model, 'predict_proba'):
#             probabilities = model.predict_proba(features_scaled)[0]
#             confidence = probabilities[prediction]
#         else:
#             # Fallback if no probability
#             probabilities = np.array([0.5, 0.5])
#             confidence = 0.5
        
#         # Map prediction to label
#         label = "Stress Detected" if prediction == 1 else "No Stress Detected"
        
#         return label, confidence, probabilities


# # Test the classifier
# if __name__ == "__main__":
#     print("🤖 Testing Stress Classifier")
#     print("="*50)
    
#     # Create dummy data for testing
#     np.random.seed(42)
#     n_samples = 1000
#     n_features = 2120  # Match our feature size
    
#     X = np.random.randn(n_samples, n_features)
#     y = np.random.choice([0, 1], n_samples, p=[0.67, 0.33])  # Imbalanced like real data
    
#     # Split data
#     from sklearn.model_selection import train_test_split
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
#     print(f"\n📊 Data splits:")
#     print(f"   Training: {X_train.shape[0]} samples")
#     print(f"   Validation: {X_val.shape[0]} samples")
#     print(f"   Testing: {X_test.shape[0]} samples")
    
#     # Initialize classifier
#     classifier = StressClassifier(random_state=42)
    
#     # Create and test scaler (we'll add this in next step)
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_val_scaled = scaler.transform(X_val)
    
#     # Train all models (with small dummy data for testing)
#     print("\n⚠️  Testing with dummy data (small sample)...")
#     results = classifier.train_all_models(
#         X_train_scaled[:100], y_train[:100],
#         X_val_scaled[:50], y_val[:50],
#         use_smote=True
#     )
    
#     print("\n✅ Classifier test complete!")


# """
# stress_classifier.py
# Stress detection classifier with SMOTE handling
# """

# import numpy as np
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.dummy import DummyClassifier


# class StressClassifier:

#     def __init__(self, random_state=42):
#         self.random_state = random_state
#         self.best_model = None
#         self.best_model_name = None
#         self.cv_folds = 3

#     # -----------------------------
#     # SMOTE Handling - FIXED VERSION
#     # -----------------------------
#     def apply_smote(self, X, y):
#         """
#         Apply SMOTE to handle class imbalance
#         Handles cases where only one class is present
#         """
#         print("\n🔄 Applying SMOTE for class imbalance...")
        
#         # Count classes
#         unique, counts = np.unique(y, return_counts=True)
        
#         # Safe printing of class distribution
#         class_dist = {}
#         for u, c in zip(unique, counts):
#             if u == 0:
#                 class_dist['No Stress'] = c
#             elif u == 1:
#                 class_dist['Stress'] = c
#         print(f"   Class distribution: {class_dist}")
        
#         # If only one class exists
#         if len(unique) < 2:
#             print(f"   ⚠️ Only one class present. Skipping SMOTE.")
#             return X, y
        
#         # Get counts safely - FIXED THIS PART
#         count_0 = 0
#         count_1 = 0
#         for u, c in zip(unique, counts):
#             if u == 0:
#                 count_0 = c
#             elif u == 1:
#                 count_1 = c
        
#         print(f"   Before SMOTE - No Stress: {count_0}, Stress: {count_1}")
        
#         # Check minimum samples
#         min_samples_needed = 6
#         minority_count = min(counts)
        
#         if minority_count < min_samples_needed:
#             print(f"   ⚠️ Minority class has only {minority_count} samples "
#                   f"(need {min_samples_needed}). Skipping SMOTE.")
#             return X, y
        
#         try:
#             from imblearn.over_sampling import SMOTE
#             smote = SMOTE(random_state=self.random_state)
            
#             X_resampled, y_resampled = smote.fit_resample(X, y)
            
#             # Count after SMOTE
#             unique, counts = np.unique(y_resampled, return_counts=True)
#             count_0 = 0
#             count_1 = 0
#             for u, c in zip(unique, counts):
#                 if u == 0:
#                     count_0 = c
#                 elif u == 1:
#                     count_1 = c
#             print(f"   After SMOTE - No Stress: {count_0}, Stress: {count_1}")
            
#             return X_resampled, y_resampled
            
#         except Exception as e:
#             print(f"   ⚠️ SMOTE failed: {e}. Returning original data.")
#             return X, y

#     # -----------------------------
#     # Model Selection
#     # -----------------------------
#     def get_model_and_params(self, model_name):

#         if model_name == "random_forest":
#             model = RandomForestClassifier(random_state=self.random_state)
#             params = {
#                 "n_estimators": [50, 100],
#                 "max_depth": [None, 10],
#                 "min_samples_split": [2, 5]
#             }

#         elif model_name == "svm":
#             model = SVC(probability=True, random_state=self.random_state)
#             params = {
#                 "C": [0.1, 1, 10],
#                 "kernel": ["linear", "rbf"]
#             }

#         elif model_name == "gradient_boosting":
#             model = GradientBoostingClassifier(random_state=self.random_state)
#             params = {
#                 "n_estimators": [50, 100],
#                 "learning_rate": [0.01, 0.1],
#                 "max_depth": [3, 5]
#             }

#         else:
#             raise ValueError("Unsupported model")

#         return model, params

#     # -----------------------------
#     # Grid Search Training
#     # -----------------------------
#     def train_with_gridsearch(self, X, y, model_name, cv=3):

#         print(f"\nTraining {model_name} with GridSearch...")

#         model, params = self.get_model_and_params(model_name)

#         grid = GridSearchCV(
#             model,
#             params,
#             cv=cv,
#             scoring="f1",
#             n_jobs=-1,
#             verbose=0
#         )

#         grid.fit(X, y)

#         print(f"   Best params: {grid.best_params_}")
#         print(f"   Best score: {grid.best_score_:.4f}")

#         return grid.best_estimator_, grid.best_params_, grid.best_score_

#     # -----------------------------
#     # Probability Calibration
#     # -----------------------------
#     def calibrate_probabilities(self, model, X_val, y_val):

#         print("   Calibrating probabilities...")

#         calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
#         calibrated.fit(X_val, y_val)

#         return calibrated

#     # -----------------------------
#     # Evaluation
#     # -----------------------------
#     def evaluate_model(self, model, X_val, y_val, name):

#         print(f"\nEvaluating {name}...")

#         y_pred = model.predict(X_val)
        
#         # Calculate metrics
#         accuracy = accuracy_score(y_val, y_pred)
#         precision = precision_score(y_val, y_pred, zero_division=0)
#         recall = recall_score(y_val, y_pred, zero_division=0)
#         f1 = f1_score(y_val, y_pred, zero_division=0)
#         cm = confusion_matrix(y_val, y_pred).tolist()

#         metrics = {
#             "accuracy": float(accuracy),
#             "precision": float(precision),
#             "recall": float(recall),
#             "f1_score": float(f1),
#             "confusion_matrix": cm
#         }

#         print(f"   Accuracy: {accuracy:.4f}")
#         print(f"   Precision: {precision:.4f}")
#         print(f"   Recall: {recall:.4f}")
#         print(f"   F1-Score: {f1:.4f}")

#         return metrics

#     # -----------------------------
#     # Estimate model size
#     # -----------------------------
#     def estimate_model_size(self, model):
#         import sys
#         import pickle
#         size_bytes = len(pickle.dumps(model))
#         size_mb = size_bytes / (1024 * 1024)
#         return size_mb

#     # -----------------------------
#     # Train All Models
#     # -----------------------------
#     def train_all_models(self, X_train, y_train, X_val, y_val, use_smote=True):

#         print("\n" + "=" * 60)
#         print("🚀 TRAINING ALL MODELS")
#         print("=" * 60)

#         # Handle small dataset
#         if len(X_train) < 10:
#             print(f"⚠️ Very small dataset ({len(X_train)} samples). Using simplified training.")
#             self.cv_folds = min(2, len(np.unique(y_train)))

#         # Apply SMOTE
#         if use_smote and len(np.unique(y_train)) > 1:
#             X_train_balanced, y_train_balanced = self.apply_smote(X_train, y_train)
#         else:
#             if len(np.unique(y_train)) == 1:
#                 print("⚠️ Only one class present. Skipping SMOTE.")
#             X_train_balanced, y_train_balanced = X_train, y_train

#         results = {}
#         best_f1 = 0

#         for model_name in ["random_forest", "svm", "gradient_boosting"]:

#             print("\n" + "-" * 40)

#             if model_name == "svm" and len(X_train) < 20:
#                 print("⚠️ Skipping SVM (requires more samples)")
#                 continue

#             try:
#                 best_model, best_params, cv_score = self.train_with_gridsearch(
#                     X_train_balanced,
#                     y_train_balanced,
#                     model_name,
#                     cv=min(self.cv_folds, max(2, len(np.unique(y_train_balanced))))
#                 )

#                 # Calibrate if validation set has both classes
#                 if len(X_val) >= 5 and len(np.unique(y_val)) > 1:
#                     calibrated_model = self.calibrate_probabilities(best_model, X_val, y_val)
#                 else:
#                     print("   ⚠️ Validation set too small or single class, skipping calibration")
#                     calibrated_model = best_model

#                 metrics = self.evaluate_model(calibrated_model, X_val, y_val, model_name)

#                 results[model_name] = {
#                     "model": calibrated_model,
#                     "best_params": best_params,
#                     "cv_score": float(cv_score),
#                     "validation_metrics": metrics,
#                     "model_size": self.estimate_model_size(calibrated_model)
#                 }

#                 if metrics["f1_score"] > best_f1:
#                     best_f1 = metrics["f1_score"]
#                     self.best_model = calibrated_model
#                     self.best_model_name = model_name

#             except Exception as e:
#                 print(f"❌ Error training {model_name}: {e}")
#                 continue

#         if self.best_model is None:
#             print("\n⚠️ No models trained successfully. Using dummy classifier.")
#             self.best_model = DummyClassifier(strategy="most_frequent")
#             self.best_model.fit(X_train, y_train)
#             self.best_model_name = "dummy"
#             best_f1 = 0

#         print("\n" + "=" * 60)
#         print(f"🏆 Best Model: {self.best_model_name}")
#         print(f"   F1 Score: {best_f1:.4f}")
#         print("=" * 60)

#         return results


































"""
stress_classifier.py
Stress detection classifier with SMOTE handling
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier


class StressClassifier:

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.best_model = None
        self.best_model_name = None
        self.cv_folds = 3

    # -----------------------------
    # SMOTE Handling
    # -----------------------------
    def apply_smote(self, X, y):
        """
        Apply SMOTE to handle class imbalance
        Handles cases where only one class is present
        """
        print("\n🔄 Applying SMOTE for class imbalance...")
        
        # Count classes
        unique, counts = np.unique(y, return_counts=True)
        
        # Safe printing of class distribution
        class_dist = {}
        for u, c in zip(unique, counts):
            if u == 0:
                class_dist['No Stress'] = c
            elif u == 1:
                class_dist['Stress'] = c
        print(f"   Class distribution: {class_dist}")
        
        # If only one class exists
        if len(unique) < 2:
            print(f"   ⚠️ Only one class present. Skipping SMOTE.")
            return X, y
        
        # Get counts safely
        count_0 = 0
        count_1 = 0
        for u, c in zip(unique, counts):
            if u == 0:
                count_0 = c
            elif u == 1:
                count_1 = c
        
        print(f"   Before SMOTE - No Stress: {count_0}, Stress: {count_1}")
        
        # Check minimum samples
        min_samples_needed = 6
        minority_count = min(counts)
        
        if minority_count < min_samples_needed:
            print(f"   ⚠️ Minority class has only {minority_count} samples "
                  f"(need {min_samples_needed}). Skipping SMOTE.")
            return X, y
        
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=self.random_state)
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Count after SMOTE
            unique, counts = np.unique(y_resampled, return_counts=True)
            count_0 = 0
            count_1 = 0
            for u, c in zip(unique, counts):
                if u == 0:
                    count_0 = c
                elif u == 1:
                    count_1 = c
            print(f"   After SMOTE - No Stress: {count_0}, Stress: {count_1}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"   ⚠️ SMOTE failed: {e}. Returning original data.")
            return X, y

    # -----------------------------
    # Model Selection
    # -----------------------------
    def get_model_and_params(self, model_name):
        """
        Get model and hyperparameter grid based on model name
        """
        if model_name == "random_forest":
            model = RandomForestClassifier(random_state=self.random_state)
            params = {
                "n_estimators": [50, 100],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5]
            }

        elif model_name == "svm":
            model = SVC(probability=True, random_state=self.random_state)
            params = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }

        elif model_name == "gradient_boosting":
            model = GradientBoostingClassifier(random_state=self.random_state)
            params = {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5]
            }

        else:
            raise ValueError("Unsupported model")

        return model, params

    # -----------------------------
    # Grid Search Training
    # -----------------------------
    def train_with_gridsearch(self, X, y, model_name, cv=3):
        """
        Train model with grid search, handling small datasets
        """
        print(f"\nTraining {model_name} with GridSearch...")
        
        # Check if we have enough samples for CV
        n_samples = len(X)
        n_classes = len(np.unique(y))
        
        # Adjust CV folds based on available data
        if n_samples < 5 or n_classes < 2:
            print(f"   ⚠️ Too few samples ({n_samples}) or classes ({n_classes}) for CV. Using simple training.")
            model, params = self.get_model_and_params(model_name)
            # Use default parameters without grid search
            model.fit(X, y)
            # Return a simple score (accuracy) since we can't do CV
            score = model.score(X, y)
            return model, params, score
        
        # Ensure cv is at least 2 and not more than samples
        cv_actual = min(max(2, cv), n_samples)
        if cv_actual < 2:
            cv_actual = 2
        
        model, params = self.get_model_and_params(model_name)
        
        try:
            grid = GridSearchCV(
                model,
                params,
                cv=cv_actual,
                scoring="f1",
                n_jobs=-1,
                verbose=0
            )
            
            grid.fit(X, y)
            
            print(f"   Best params: {grid.best_params_}")
            print(f"   Best score: {grid.best_score_:.4f}")
            
            return grid.best_estimator_, grid.best_params_, grid.best_score_
        
        except Exception as e:
            print(f"   ⚠️ Grid search failed: {e}. Using simple training.")
            model.fit(X, y)
            score = model.score(X, y)
            return model, params, score

    # -----------------------------
    # Probability Calibration
    # -----------------------------
    def calibrate_probabilities(self, model, X_val, y_val):
        """
        Calibrate model probabilities using sigmoid method
        """
        print("   Calibrating probabilities...")
        
        # Check if we have both classes for calibration
        if len(np.unique(y_val)) < 2:
            print("   ⚠️ Only one class in validation set. Skipping calibration.")
            return model

        try:
            calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
            calibrated.fit(X_val, y_val)
            return calibrated
        except Exception as e:
            print(f"   ⚠️ Calibration failed: {e}. Using uncalibrated model.")
            return model

    # -----------------------------
    # Evaluation
    # -----------------------------
    def evaluate_model(self, model, X_val, y_val, name):
        """
        Evaluate model performance
        """
        print(f"\nEvaluating {name}...")

        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        
        # For precision, recall, f1 - handle cases with single class
        if len(np.unique(y_val)) > 1:
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        cm = confusion_matrix(y_val, y_pred).tolist()

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm
        }

        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")

        return metrics

    # -----------------------------
    # Estimate model size
    # -----------------------------
    def estimate_model_size(self, model):
        """
        Estimate model size in MB
        """
        import sys
        import pickle
        try:
            size_bytes = len(pickle.dumps(model))
            size_mb = size_bytes / (1024 * 1024)
            return size_mb
        except:
            return 0.0

    # -----------------------------
    # Train All Models
    # -----------------------------
    def train_all_models(self, X_train, y_train, X_val, y_val, use_smote=True):
        """
        Train all models and select the best one
        """
        print("\n" + "=" * 60)
        print("🚀 TRAINING ALL MODELS")
        print("=" * 60)

        # Handle small dataset
        if len(X_train) < 10:
            print(f"⚠️ Very small dataset ({len(X_train)} samples). Using simplified training.")
            self.cv_folds = 2

        # Apply SMOTE
        if use_smote and len(np.unique(y_train)) > 1:
            X_train_balanced, y_train_balanced = self.apply_smote(X_train, y_train)
        else:
            if len(np.unique(y_train)) == 1:
                print("⚠️ Only one class present. Skipping SMOTE.")
            X_train_balanced, y_train_balanced = X_train, y_train

        results = {}
        best_f1 = 0

        for model_name in ["random_forest", "svm", "gradient_boosting"]:

            print("\n" + "-" * 40)

            # Skip SVM for very small datasets
            if model_name == "svm" and len(X_train) < 20:
                print("⚠️ Skipping SVM (requires more samples)")
                continue

            try:
                # Determine CV folds based on data
                n_classes = len(np.unique(y_train_balanced))
                cv_folds = min(self.cv_folds, max(2, n_classes))
                
                best_model, best_params, cv_score = self.train_with_gridsearch(
                    X_train_balanced,
                    y_train_balanced,
                    model_name,
                    cv=cv_folds
                )

                # Calibrate if validation set has both classes
                if len(X_val) >= 5:
                    calibrated_model = self.calibrate_probabilities(best_model, X_val, y_val)
                else:
                    print("   ⚠️ Validation set too small, skipping calibration")
                    calibrated_model = best_model

                metrics = self.evaluate_model(calibrated_model, X_val, y_val, model_name)

                results[model_name] = {
                    "model": calibrated_model,
                    "best_params": best_params,
                    "cv_score": float(cv_score) if cv_score else 0.0,
                    "validation_metrics": metrics,
                    "model_size": self.estimate_model_size(calibrated_model)
                }

                if metrics["f1_score"] > best_f1:
                    best_f1 = metrics["f1_score"]
                    self.best_model = calibrated_model
                    self.best_model_name = model_name

            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                continue

        if self.best_model is None:
            print("\n⚠️ No models trained successfully. Using dummy classifier.")
            self.best_model = DummyClassifier(strategy="most_frequent")
            self.best_model.fit(X_train, y_train)
            self.best_model_name = "dummy"
            best_f1 = 0.0

        print("\n" + "=" * 60)
        print(f"🏆 Best Model: {self.best_model_name}")
        print(f"   F1 Score: {best_f1:.4f}")
        print("=" * 60)

        return results