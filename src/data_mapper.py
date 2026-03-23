"""
data_mapper.py
Purpose: Map and analyze the RAVDESS dataset structure
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RAVDESSMapper:
    """
    Maps the RAVDESS dataset to understand file structure and labels
    """
    
    # Emotion codes from RAVDESS filename convention
    EMOTION_MAP = {
        '01': 'neutral',    # No stress
        '02': 'calm',       # No stress
        '03': 'happy',      # No stress
        '04': 'sad',        # No stress
        '05': 'angry',      # STRESS DETECTED
        '06': 'fearful',    # STRESS DETECTED
        '07': 'disgust',    # No stress
        '08': 'surprised'   # No stress
    }
    
    # Stress mapping based on requirements
    STRESS_MAP = {
        '05': 1,  # angry -> stress
        '06': 1,  # fearful -> stress
        '01': 0, '02': 0, '03': 0, '04': 0, '07': 0, '08': 0  # others -> no stress
    }
    
    def __init__(self, data_path):
        """
        Initialize mapper with path to ravdess_raw folder
        
        Args:
            data_path (str): Path to the ravdess_raw directory
        """
        self.data_path = Path(data_path)
        self.metadata = None
        
    def scan_dataset(self):
        """
        Scan all audio files and extract metadata from filenames
        """
        files_data = []
        
        # Check if path exists
        if not self.data_path.exists():
            print(f"❌ Error: Path {self.data_path} does not exist!")
            return None
            
        # Walk through all actor folders
        actor_folders = sorted(self.data_path.glob('Actor_*'))
        
        if len(actor_folders) == 0:
            print(f"❌ No Actor_* folders found in {self.data_path}")
            print("Please make sure your dataset is in the correct location:")
            print(f"  Expected: {self.data_path.absolute()}")
            return None
        
        print(f"Found {len(actor_folders)} actor folders")
        
        for actor_folder in actor_folders:
            actor_id = actor_folder.name
            
            # Get all audio files in this actor's folder
            audio_files = list(actor_folder.glob('*.wav'))
            audio_files.extend(list(actor_folder.glob('*.mp3')))  # Also look for mp3 just in case
            
            if len(audio_files) == 0:
                print(f"⚠️  Warning: No audio files found in {actor_folder}")
                continue
                
            for file_path in audio_files:
                # Parse filename
                filename = file_path.stem  # without extension
                parts = filename.split('-')
                
                if len(parts) >= 7:
                    # Extract relevant information
                    emotion_code = parts[2]
                    intensity = parts[3]
                    statement = parts[4]
                    repetition = parts[5]
                    
                    # Get emotion and stress labels
                    emotion = self.EMOTION_MAP.get(emotion_code, 'unknown')
                    stress_label = self.STRESS_MAP.get(emotion_code, -1)
                    
                    files_data.append({
                        'file_path': str(file_path),
                        'filename': file_path.name,
                        'actor': actor_id,
                        'actor_num': int(actor_id.split('_')[1]),
                        'emotion_code': emotion_code,
                        'emotion': emotion,
                        'stress_label': stress_label,
                        'intensity': intensity,
                        'statement': statement,
                        'repetition': repetition
                    })
        
        # Create DataFrame
        if len(files_data) > 0:
            self.metadata = pd.DataFrame(files_data)
            print(f"✅ Successfully scanned {len(files_data)} files")
        else:
            print("❌ No files found with valid filenames")
            
        return self.metadata
    
    def get_dataset_stats(self):
        """Print statistics about the dataset"""
        if self.metadata is None or len(self.metadata) == 0:
            print("❌ No metadata available. Run scan_dataset() first.")
            return
        
        print("\n" + "="*60)
        print("📊 RAVDESS DATASET STATISTICS")
        print("="*60)
        print(f"📍 Dataset location: {self.data_path}")
        print(f"📁 Total files: {len(self.metadata)}")
        print(f"🎭 Total actors: {self.metadata['actor'].nunique()}")
        print(f"   Actor IDs: {sorted(self.metadata['actor_num'].unique())}")
        
        print("\n📊 Emotion distribution:")
        emotion_counts = self.metadata['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            percentage = (count/len(self.metadata))*100
            print(f"   {emotion:10}: {count:4} files ({percentage:.1f}%)")
        
        print("\n📊 Stress distribution:")
        stress_counts = self.metadata['stress_label'].value_counts()
        for stress, count in stress_counts.items():
            label = "STRESS" if stress == 1 else "NO STRESS"
            percentage = (count/len(self.metadata))*100
            print(f"   {label:10}: {count:4} files ({percentage:.1f}%)")
        
        print("\n📊 Files per actor (first 5):")
        actor_counts = self.metadata.groupby('actor').size().head()
        for actor, count in actor_counts.items():
            print(f"   {actor}: {count} files")
            
        # Check for class imbalance
        stress_ratio = stress_counts[1] / stress_counts[0] if 0 in stress_counts else 0
        print(f"\n📈 Class imbalance ratio (stress:no-stress): 1:{1/stress_ratio:.2f}")
        print("="*60 + "\n")
        
    def get_actor_split(self, train_ratio=0.7, random_seed=42):
        """
        Get speaker-independent train/test split by actor IDs
        """
        if self.metadata is None or len(self.metadata) == 0:
            print("❌ No metadata available. Run scan_dataset() first.")
            return None
            
        # Get unique actors
        actors = sorted(self.metadata['actor_num'].unique())
        np.random.seed(random_seed)
        
        # Shuffle actors
        shuffled_actors = actors.copy()
        np.random.shuffle(shuffled_actors)
        
        # Split actors
        split_idx = int(len(actors) * train_ratio)
        train_actors = sorted(shuffled_actors[:split_idx])
        test_actors = sorted(shuffled_actors[split_idx:])
        
        print("\n" + "="*60)
        print(f"🎯 TRAIN/TEST SPLIT (Speaker-Independent)")
        print("="*60)
        print(f"Train actors ({len(train_actors)}): {train_actors}")
        print(f"Test actors ({len(test_actors)}): {test_actors}")
        
        # Get indices for train/test
        train_idx = self.metadata[self.metadata['actor_num'].isin(train_actors)].index
        test_idx = self.metadata[self.metadata['actor_num'].isin(test_actors)].index
        
        train_files = len(train_idx)
        test_files = len(test_idx)
        print(f"\n📁 Training files: {train_files} ({train_files/len(self.metadata)*100:.1f}%)")
        print(f"📁 Testing files: {test_files} ({test_files/len(self.metadata)*100:.1f}%)")
        
        # Check stress distribution in train/test
        train_stress = self.metadata.loc[train_idx, 'stress_label'].value_counts()
        test_stress = self.metadata.loc[test_idx, 'stress_label'].value_counts()
        
        print("\n📊 Stress distribution in training:")
        print(f"   No Stress: {train_stress.get(0, 0)} files")
        print(f"   Stress:    {train_stress.get(1, 0)} files")
        
        print("\n📊 Stress distribution in testing:")
        print(f"   No Stress: {test_stress.get(0, 0)} files")
        print(f"   Stress:    {test_stress.get(1, 0)} files")
        
        return {
            'train_indices': train_idx,
            'test_indices': test_idx,
            'train_actors': train_actors,
            'test_actors': test_actors
        }


# Quick test when script runs directly
if __name__ == "__main__":
    print("🔍 RAVDESS Dataset Mapper")
    print("-" * 40)
    
    # Test the mapper with relative path
    data_path = Path("data/ravdess_raw")
    
    print(f"Looking for dataset at: {data_path.absolute()}")
    print("-" * 40)
    
    mapper = RAVDESSMapper(data_path)
    metadata = mapper.scan_dataset()
    
    if metadata is not None and len(metadata) > 0:
        mapper.get_dataset_stats()
        split = mapper.get_actor_split()
        
        # Save metadata for later use
        processed_dir = Path("data/processed")
        processed_dir.mkdir(exist_ok=True)
        
        metadata_path = processed_dir / "metadata.csv"
        metadata.to_csv(metadata_path, index=False)
        print(f"\n✅ Metadata saved to {metadata_path}")
    else:
        print("\n❌ Failed to scan dataset. Please check:")
        print("   1. Dataset path is correct")
        print("   2. Dataset files exist in the ravdess_raw folder")
        print("   3. File names follow the RAVDESS format")