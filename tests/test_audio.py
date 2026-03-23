"""
tests/test_audio.py
Test the audio processing and feature extraction pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor
import glob
import numpy as np

def test_pipeline():
    """Test the complete audio processing pipeline"""
    
    print("🧪 Testing Complete Audio Pipeline")
    print("="*60)
    
    # Initialize modules
    processor = AudioProcessor(target_sr=16000)
    extractor = FeatureExtractor(sr=16000, n_segments=10)
    
    # Find a sample file
    sample_files = glob.glob("data/ravdess_raw/Actor_01/*.wav")
    
    if not sample_files:
        print("❌ No test files found!")
        return
    
    test_file = sample_files[0]
    print(f"\n📁 Test file: {test_file}")
    
    # Step 1: Validate audio
    print("\n🔍 Step 1: Audio Validation")
    is_valid, msg, duration = processor.validate_audio(test_file)
    print(f"   {msg}")
    
    if not is_valid:
        return
    
    # Step 2: Preprocess audio
    print("\n🔧 Step 2: Audio Preprocessing")
    audio = processor.preprocess_for_inference(test_file, apply_noise_reduction=True)
    
    if audio is None:
        print("❌ Preprocessing failed")
        return
    
    print(f"   ✅ Preprocessing successful")
    print(f"   Original duration: {duration:.2f}s")
    print(f"   Processed duration: {len(audio)/processor.target_sr:.2f}s")
    
    # Step 3: Extract features
    print("\n📊 Step 3: Feature Extraction")
    features = extractor.extract_all_features(audio)
    
    print(f"   ✅ Feature extraction successful")
    print(f"   Feature vector shape: {features.shape}")
    print(f"   Number of features: {len(features)}")
    
    # Step 4: Verify feature consistency
    print("\n✅ Step 4: Consistency Check")
    print(f"   All features finite: {np.all(np.isfinite(features))}")
    print(f"   Feature range: [{np.min(features):.4f}, {np.max(features):.4f}]")
    
    print("\n" + "="*60)
    print("🎉 Pipeline test complete!")
    
    return features

if __name__ == "__main__":
    features = test_pipeline()