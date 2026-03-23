"""
tests/test_classifier.py
Test the stress classifier module
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stress_classifier import StressClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler

print("🧪 Testing Stress Classifier")
print("="*50)

# Create classifier instance
classifier = StressClassifier(random_state=42)

# Create dummy data
print("\n📊 Creating dummy data...")
n_samples = 500
n_features = 2120

X = np.random.randn(n_samples, n_features)
y = np.array([0]*335 + [1]*165)  # Simulate class imbalance
np.random.shuffle(y)

print(f"   Features shape: {X.shape}")
print(f"   Labels distribution: 0: {sum(y==0)}, 1: {sum(y==1)}")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"\n📈 Data splits:")
print(f"   Training: {len(X_train)}")
print(f"   Validation: {len(X_val)}")
print(f"   Testing: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Test individual model training
print("\n🔍 Testing individual model training...")
model, params, score = classifier.train_with_gridsearch(
    X_train_scaled[:200], y_train[:200], 
    'random_forest', cv=3
)
print(f"✅ Model training successful")

# Test calibration
print("\n🔍 Testing probability calibration...")
calibrated = classifier.calibrate_probabilities(model, X_val_scaled[:100], y_val[:100])
print("✅ Calibration successful")

# Test evaluation
print("\n🔍 Testing model evaluation...")
metrics = classifier.evaluate_model(calibrated, X_test_scaled[:100], y_test[:100])
print("✅ Evaluation successful")

# Test prediction with confidence
print("\n🔍 Testing prediction...")
label, confidence, probs = classifier.predict_with_confidence(
    calibrated, scaler, X_test_scaled[0]
)
print(f"   Prediction: {label}")
print(f"   Confidence: {confidence:.2f}")
print(f"   Probabilities: [No Stress: {probs[0]:.2f}, Stress: {probs[1]:.2f}]")

print("\n" + "="*50)
print("✅ All classifier tests passed!")