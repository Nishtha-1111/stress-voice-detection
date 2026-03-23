#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
training_pipeline.py
Complete training pipeline for stress detection
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import modules
try:
    from src.audio_processor import AudioProcessor
    from src.feature_extractor import FeatureExtractor
    from src.stress_classifier import StressClassifier
    from src.data_mapper import RAVDESSMapper
    print("✅ Successfully imported src modules")
except ImportError as e:
    print(f"⚠️ Import warning: {e}")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingPipeline:
    """
    End-to-end training pipeline for stress detection
    """
    
    def __init__(self, data_path="data/ravdess_raw", output_path="models"):
        """
        Initialize training pipeline
        """
        self.base_path = parent_dir
        self.data_path = self.base_path / data_path
        self.output_path = self.base_path / output_path
        self.output_path.mkdir(exist_ok=True)
        
        print(f"\n📂 Initializing pipeline:")
        print(f"   Data path: {self.data_path}")
        print(f"   Output path: {self.output_path}")
        
        # Initialize components
        self.audio_processor = AudioProcessor(target_sr=16000)
        self.feature_extractor = FeatureExtractor(sr=16000, n_segments=10)
        self.classifier = StressClassifier(random_state=42)
        
        # Try to initialize mapper, but don't fail if it doesn't exist
        try:
            self.mapper = RAVDESSMapper(str(self.data_path))
        except NameError:
            print("⚠️  RAVDESSMapper not available, using simple scanning")
            self.mapper = None
            
        self.scaler = StandardScaler()
        
        # Storage
        self.metadata = None
        self.features = None
        self.labels = None
        self.train_indices = None
        self.test_indices = None
    
    def prepare_dataset(self, test_size=0.3, random_seed=42):
        """
        Prepare dataset with speaker-independent split
        """
        print("\n" + "="*60)
        print("📊 STEP 1: PREPARING DATASET")
        print("="*60)
        
        if not self.data_path.exists():
            print(f"❌ Data path does not exist: {self.data_path}")
            return None
        
        # If mapper is available, use it
        if self.mapper is not None:
            self.metadata = self.mapper.scan_dataset()
            if self.metadata is not None:
                split = self.mapper.get_actor_split(
                    train_ratio=1-test_size, 
                    random_seed=random_seed
                )
                if split:
                    self.train_indices = split['train_indices']
                    self.test_indices = split['test_indices']
                    print(f"\n✅ Dataset prepared with speaker-independent split")
                    print(f"   Total files: {len(self.metadata)}")
                    print(f"   Training files: {len(self.train_indices)}")
                    print(f"   Testing files: {len(self.test_indices)}")
                    return self.metadata
        
        # Fallback: simple scan
        print("Using simple dataset scan...")
        files_data = []
        actor_folders = sorted(self.data_path.glob('Actor_*'))
        
        for actor_folder in actor_folders:
            audio_files = list(actor_folder.glob('*.wav'))
            for file_path in audio_files:
                filename = file_path.stem
                parts = filename.split('-')
                if len(parts) >= 3:
                    emotion_code = parts[2]
                    stress_label = 1 if emotion_code in ['05', '06'] else 0
                    files_data.append({
                        'file_path': str(file_path),
                        'filename': file_path.name,
                        'actor': actor_folder.name,
                        'stress_label': stress_label
                    })
        
        self.metadata = pd.DataFrame(files_data)
        
        # Simple random split
        indices = np.arange(len(self.metadata))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_seed,
            stratify=self.metadata['stress_label']
        )
        self.train_indices = train_idx
        self.test_indices = test_idx
        
        print(f"\n✅ Dataset prepared with random split")
        print(f"   Total files: {len(self.metadata)}")
        print(f"   Training files: {len(self.train_indices)}")
        print(f"   Testing files: {len(self.test_indices)}")
        print(f"   Stress distribution: 0: {sum(self.metadata['stress_label']==0)}, "
              f"1: {sum(self.metadata['stress_label']==1)}")
        
        return self.metadata
    
    def extract_all_features(self, max_files=None):
        """
        Extract features from all audio files
        """
        print("\n" + "="*60)
        print("🔧 STEP 2: FEATURE EXTRACTION")
        print("="*60)
        
        if self.metadata is None:
            self.prepare_dataset()
        
        # Get files to process
        if max_files:
            files_to_process = self.metadata.iloc[:max_files]
            print(f"⚠️  Processing only {max_files} files (test mode)")
        else:
            files_to_process = self.metadata
            print(f"Processing all {len(files_to_process)} files")
        
        features_list = []
        labels_list = []
        failed_files = []
        
        for idx, row in files_to_process.iterrows():
            try:
                file_path = row['file_path']
                stress_label = row['stress_label']
                
                filename = Path(file_path).name
                print(f"\nProcessing [{idx+1}/{len(files_to_process)}]: {filename}")
                
                if not os.path.exists(file_path):
                    print(f"   ❌ File not found")
                    failed_files.append(file_path)
                    continue
                
                # Preprocess audio
                audio_result = self.audio_processor.preprocess_for_training(
                    file_path, augment=False
                )
                
                if audio_result is None or 'original' not in audio_result:
                    print(f"   ❌ Preprocessing failed")
                    failed_files.append(file_path)
                    continue
                
                # Extract features
                features = self.feature_extractor.extract_all_features(
                    audio_result['original']
                )
                
                features_list.append(features)
                labels_list.append(stress_label)
                
                print(f"   ✅ Features shape: {features.shape}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                failed_files.append(file_path)
        
        if len(features_list) == 0:
            print("❌ No features extracted")
            return None, None
        
        self.features = np.array(features_list)
        self.labels = np.array(labels_list)
        
        print(f"\n✅ Feature extraction complete:")
        print(f"   Processed: {len(features_list)} files")
        print(f"   Failed: {len(failed_files)} files")
        print(f"   Features shape: {self.features.shape}")
        print(f"   Labels distribution: 0: {sum(self.labels==0)}, 1: {sum(self.labels==1)}")
        
        # Save features
        np.save(self.output_path / 'features.npy', self.features)
        np.save(self.output_path / 'labels.npy', self.labels)
        print(f"\n💾 Features saved to {self.output_path}")
        
        return self.features, self.labels
    
    def prepare_train_test_split(self, test_mode=False):
        """
        Prepare train/test split
        """
        print("\n" + "="*60)
        print("📂 STEP 3: TRAIN TEST SPLIT")
        print("="*60)
        
        if self.features is None or self.labels is None:
            print("❌ No features/labels available")
            return None, None, None, None, None, None
        
        # If in test mode or small dataset, use simple random split
        if test_mode or len(self.features) < 100:
            print("⚠️  Test mode: using random split")
            
            # Ensure we have at least 2 classes for stratification
            if len(np.unique(self.labels)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    self.features, self.labels, test_size=0.3, 
                    random_state=42, stratify=self.labels
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_test, y_test, test_size=0.5, 
                    random_state=42, stratify=y_test
                )
            else:
                # If only one class, don't use stratification
                X_train, X_test, y_train, y_test = train_test_split(
                    self.features, self.labels, test_size=0.3, 
                    random_state=42
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_test, y_test, test_size=0.5, 
                    random_state=42
                )
        else:
            # Use stored indices
            train_idx = [i for i in self.train_indices if i < len(self.features)]
            test_idx = [i for i in self.test_indices if i < len(self.features)]
            
            X_train = self.features[train_idx]
            y_train = self.labels[train_idx]
            X_test = self.features[test_idx]
            y_test = self.labels[test_idx]
            
            # Further split test into validation and test
            if len(np.unique(y_test)) > 1:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_test, y_test, test_size=0.5, 
                    random_state=42, stratify=y_test
                )
            else:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_test, y_test, test_size=0.5, 
                    random_state=42
                )
        
        print(f"\n📊 Data splits:")
        print(f"   Train: {X_train.shape[0]} samples")
        print(f"   Val: {X_val.shape[0]} samples")
        print(f"   Test: {X_test.shape[0]} samples")
        
        if len(np.unique(y_train)) > 1:
            print(f"\n   Train distribution - No Stress: {sum(y_train==0)}, Stress: {sum(y_train==1)}")
        if len(np.unique(y_val)) > 1:
            print(f"   Val distribution - No Stress: {sum(y_val==0)}, Stress: {sum(y_val==1)}")
        if len(np.unique(y_test)) > 1:
            print(f"   Test distribution - No Stress: {sum(y_test==0)}, Stress: {sum(y_test==1)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """
        Scale features
        """
        print("\n" + "="*60)
        print("📏 STEP 4: SCALING FEATURES")
        print("="*60)
        
        self.scaler.fit(X_train)
        
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Features scaled successfully")
        print(f"   Mean of scaled train: {X_train_scaled.mean():.4f}")
        print(f"   Std of scaled train: {X_train_scaled.std():.4f}")
        
        # Save scaler
        joblib.dump(self.scaler, self.output_path / 'scaler.pkl')
        print(f"💾 Scaler saved to {self.output_path / 'scaler.pkl'}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train all models
        """
        print("\n" + "="*60)
        print("🤖 STEP 5: TRAINING MODELS")
        print("="*60)
        
        results = self.classifier.train_all_models(
            X_train, y_train, X_val, y_val, use_smote=True
        )
        
        return results
    
    def evaluate_best_model(self, X_test, y_test):
        """
        Evaluate best model
        """
        print("\n" + "="*60)
        print("📈 STEP 6: FINAL EVALUATION")
        print("="*60)
        
        if self.classifier.best_model is None:
            print("❌ No best model found")
            return None
        
        metrics = self.classifier.evaluate_model(
            self.classifier.best_model, X_test, y_test,
            f"Best Model ({self.classifier.best_model_name})"
        )
        
        return metrics
    
    def plot_results(self, metrics):
        """
        Plot confusion matrix and feature importance
        Handles single-class cases gracefully
        """
        print("\n" + "="*60)
        print("📊 STEP 7: GENERATING PLOTS")
        print("="*60)
        
        if metrics is None:
            print("❌ No metrics to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot confusion matrix
        cm = np.array(metrics['confusion_matrix'])
        
        # Check if it's a 2x2 matrix (both classes present)
        if cm.shape == (2, 2):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title('Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            axes[0].set_xticklabels(['No Stress', 'Stress'])
            axes[0].set_yticklabels(['No Stress', 'Stress'])
        else:
            # Handle single-class case
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title('Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            
            # Set labels based on the actual class
            if len(cm) == 1:
                if len(np.unique(self.labels)) == 1:
                    only_class = self.labels[0]
                    label = "No Stress" if only_class == 0 else "Stress"
                    axes[0].set_xticklabels([label])
                    axes[0].set_yticklabels([label])
        
        # Plot feature importance (if available)
        if hasattr(self.classifier.best_model, 'feature_importances_'):
            importances = self.classifier.best_model.feature_importances_
            top_idx = np.argsort(importances)[-10:]
            top_importances = importances[top_idx]
            
            axes[1].barh(range(10), top_importances)
            axes[1].set_yticks(range(10))
            axes[1].set_yticklabels([f'F{i}' for i in top_idx])
            axes[1].set_title('Top 10 Feature Importances')
            axes[1].set_xlabel('Importance')
        else:
            axes[1].text(0.5, 0.5, 'Feature importance\nnot available', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('Feature Importances')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_path / 'performance_plots.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"💾 Plots saved to {plot_path}")
        
        plt.show()
    
    def save_performance_summary(self, metrics, results):
        """
        Save performance summary
        """
        print("\n" + "="*60)
        print("💾 STEP 8: SAVING RESULTS")
        print("="*60)
        
        if metrics is None:
            return None
        
        summary = {
            'best_model': self.classifier.best_model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_stats': {
                'total_files': len(self.metadata) if self.metadata is not None else 0,
                'features_shape': list(self.features.shape) if self.features is not None else [0,0]
            },
            'best_model_metrics': {
                k: float(v) if isinstance(v, (np.floating, float, np.integer)) else v 
                for k, v in metrics.items() if k != 'confusion_matrix'
            },
            'confusion_matrix': metrics.get('confusion_matrix', [])
        }
        
        summary_path = self.output_path / 'performance_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"✅ Summary saved to {summary_path}")
        
        return summary
    
    def run_pipeline(self, max_files=None):
        """
        Run complete training pipeline
        """
        print("\n" + "🚀"*20)
        print("STARTING TRAINING PIPELINE")
        print("🚀"*20)
        
        start_time = datetime.now()
        test_mode = max_files is not None and max_files < 100
        
        # Step 1: Prepare dataset
        if self.prepare_dataset() is None:
            return None
        
        # Step 2: Extract features
        features, labels = self.extract_all_features(max_files)
        if features is None:
            return None
        
        # Step 3: Split data
        split = self.prepare_train_test_split(test_mode=test_mode)
        if split[0] is None:
            return None
        X_train, X_val, X_test, y_train, y_val, y_test = split
        
        # Step 4: Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        # Step 5: Train models
        results = self.train_models(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Step 6: Evaluate
        metrics = self.evaluate_best_model(X_test_scaled, y_test)
        
        # Step 7: Plot results
        self.plot_results(metrics)
        
        # Step 8: Save results
        summary = self.save_performance_summary(metrics, results)
        
        # Save best model
        if self.classifier.best_model is not None:
            model_path = self.output_path / 'stress_model.pkl'
            joblib.dump(self.classifier.best_model, model_path)
            print(f"💾 Best model saved to {model_path}")
            
            # Check model size
            if os.path.exists(model_path):
                model_size = os.path.getsize(model_path) / (1024 * 1024)
                print(f"   Model size: {model_size:.2f} MB")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "✨"*20)
        print(f"✨ PIPELINE COMPLETE!")
        print(f"✨ Total time: {duration:.2f} minutes")
        print("✨"*20)
        
        return summary


if __name__ == "__main__":
    print("="*60)
    print("🎯 STRESS DETECTION TRAINING PIPELINE")
    print("="*60)
    
    # For testing with small sample (5 files)
    print("\n🧪 Running test with 5 files...")
    pipeline = TrainingPipeline()
    summary = pipeline.run_pipeline(max_files=None)
    
    if summary:
        print("\n✅ Test successful!")
        print(f"   Best model: {summary['best_model']}")
        
        # Uncomment below to run full training
        print("\n" + "="*60)
        print("To run full training on all 1440 files:")
        print("1. Edit this file and change max_files=5 to max_files=None")
        print("2. Or create a new script with:")
        print("   pipeline = TrainingPipeline()")
        print("   summary = pipeline.run_pipeline(max_files=None)")
        print("="*60)
    else:
        print("\n❌ Test failed. Please check the errors above.")

        