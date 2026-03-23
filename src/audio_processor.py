"""
audio_processor.py
Purpose: Handle all audio preprocessing operations
- Load audio from various formats
- Convert stereo to mono
- Resample to 16kHz
- Silence trimming
- Noise reduction
- Audio augmentation
"""

import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import io
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """
    Handles all audio preprocessing operations
    Ensures consistent preprocessing between training and inference
    """
    
    def __init__(self, target_sr=16000):
        """
        Initialize audio processor
        
        Args:
            target_sr (int): Target sample rate (16kHz as per requirements)
        """
        self.target_sr = target_sr
        self.silence_threshold = 30  # top_db value for silence detection
        self.silence_padding = 0.1   # Keep 0.1s of silence at boundaries
        
    def load_audio(self, file_path, duration=None):
        """
        Load audio file with robust format handling
        
        Args:
            file_path (str): Path to audio file
            duration (float): Maximum duration to load (seconds)
            
        Returns:
            tuple: (audio_time_series, sample_rate)
        """
        try:
            # Load audio with librosa (supports wav, mp3, m4a, flac)
            audio, sr = librosa.load(
                file_path, 
                sr=self.target_sr,  # Resample to target rate
                duration=duration,   # Limit duration if specified
                mono=True            # Convert stereo to mono
            )
            
            print(f"✅ Loaded: {file_path}")
            print(f"   Duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")
            
            return audio, sr
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {str(e)}")
            return None, None
    
    def trim_silence(self, audio, sr):
        """
        Conservative silence trimming
        Keeps natural speech pauses and 0.1s of silence at boundaries
        
        Args:
            audio (np.array): Audio time series
            sr (int): Sample rate
            
        Returns:
            np.array: Trimmed audio
        """
        # Detect non-silent intervals
        non_silent_intervals = librosa.effects.split(
            audio, 
            top_db=self.silence_threshold
        )
        
        if len(non_silent_intervals) == 0:
            print("⚠️  Warning: No speech detected, returning original audio")
            return audio
        
        # Keep first and last intervals with padding
        start = max(0, non_silent_intervals[0][0] - int(self.silence_padding * sr))
        end = min(len(audio), non_silent_intervals[-1][1] + int(self.silence_padding * sr))
        
        trimmed_audio = audio[start:end]
        
        print(f"   Trimmed: {len(audio)/sr:.2f}s → {len(trimmed_audio)/sr:.2f}s")
        
        return trimmed_audio
    
    def reduce_noise(self, audio, sr, stationary=True):
        """
        Simple noise reduction using spectral gating
        
        Args:
            audio (np.array): Audio time series
            sr (int): Sample rate
            stationary (bool): If True, assumes stationary noise
            
        Returns:
            np.array: Noise-reduced audio
        """
        if stationary:
            # Use first 0.5s as noise sample
            noise_sample = audio[:int(0.5 * sr)]
            
            # Compute STFT of noise
            noise_stft = librosa.stft(noise_sample)
            noise_mag = np.abs(noise_stft)
            noise_phase = np.angle(noise_stft)
            
            # Compute average noise magnitude
            noise_profile = np.mean(noise_mag, axis=1)
            
            # Compute STFT of full audio
            audio_stft = librosa.stft(audio)
            audio_mag = np.abs(audio_stft)
            audio_phase = np.angle(audio_stft)
            
            # Spectral subtraction
            mag_reduced = np.maximum(audio_mag - noise_profile[:, np.newaxis], 0)
            
            # Reconstruct audio
            reduced_stft = mag_reduced * np.exp(1j * audio_phase)
            reduced_audio = librosa.istft(reduced_stft)
            
            # Ensure same length
            if len(reduced_audio) < len(audio):
                reduced_audio = np.pad(reduced_audio, (0, len(audio) - len(reduced_audio)))
            else:
                reduced_audio = reduced_audio[:len(audio)]
                
            return reduced_audio
        else:
            # For non-stationary noise, return original (simpler approach)
            return audio
    
    def augment_audio(self, audio, sr, augment_type='all'):
        """
        Apply data augmentation for training
        
        Args:
            audio (np.array): Audio time series
            sr (int): Sample rate
            augment_type (str): Type of augmentation ('noise', 'pitch', 'speed', 'reverb', 'all')
            
        Returns:
            list: List of augmented audio samples
        """
        augmented = []
        
        if augment_type in ['noise', 'all']:
            # Add Gaussian noise
            noise = np.random.normal(0, 0.005, len(audio))
            augmented.append(('noise', audio + noise))
        
        if augment_type in ['pitch', 'all']:
            # Pitch shift by ±2 semitones
            for n_steps in [-2, 2]:
                pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
                augmented.append((f'pitch_{n_steps}', pitch_shifted))
        
        if augment_type in ['speed', 'all']:
            # Speed perturbation ±10%
            for rate in [0.9, 1.1]:
                speed_changed = librosa.effects.time_stretch(audio, rate=rate)
                augmented.append((f'speed_{rate}', speed_changed))
        
        if augment_type in ['reverb', 'all']:
            # Simple reverb simulation (using convolution with impulse response)
            # Create simple impulse response
            ir_length = int(0.1 * sr)  # 100ms reverb
            ir = np.exp(-np.linspace(0, 5, ir_length))  # Exponential decay
            reverb_audio = signal.convolve(audio, ir, mode='same')[:len(audio)]
            augmented.append(('reverb', reverb_audio))
        
        print(f"   Created {len(augmented)} augmented versions")
        return augmented
    
    def preprocess_for_inference(self, file_path, apply_noise_reduction=True):
        """
        Complete preprocessing pipeline for inference (single file)
        
        Args:
            file_path (str): Path to audio file
            apply_noise_reduction (bool): Whether to apply noise reduction
            
        Returns:
            np.array: Preprocessed audio
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        # Apply noise reduction if requested
        if apply_noise_reduction:
            audio = self.reduce_noise(audio, sr)
        
        # Trim silence
        audio = self.trim_silence(audio, sr)
        
        return audio
    
    def preprocess_for_training(self, file_path, augment=True):
        """
        Complete preprocessing pipeline for training (with augmentation)
        
        Args:
            file_path (str): Path to audio file
            augment (bool): Whether to apply augmentation
            
        Returns:
            dict: Original and augmented audio samples
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        # Apply noise reduction
        audio = self.reduce_noise(audio, sr)
        
        # Trim silence
        audio = self.trim_silence(audio, sr)
        
        result = {
            'original': audio,
            'sr': sr,
            'augmented': []
        }
        
        # Create augmented versions for training
        if augment:
            result['augmented'] = self.augment_audio(audio, sr)
        
        return result
    
    def validate_audio(self, file_path, max_duration=30):
        """
        Validate if audio file meets requirements
        
        Args:
            file_path (str): Path to audio file
            max_duration (float): Maximum allowed duration in seconds
            
        Returns:
            tuple: (is_valid, message, duration)
        """
        try:
            # Quick load to get duration
            duration = librosa.get_duration(filename=file_path)
            
            if duration > max_duration:
                return False, f"Audio too long: {duration:.2f}s > {max_duration}s", duration
            
            # Try to load a small portion to check format
            librosa.load(file_path, duration=1)
            
            return True, "Valid audio file", duration
            
        except Exception as e:
            return False, f"Invalid audio: {str(e)}", 0


# Test the audio processor
if __name__ == "__main__":
    print("🎵 Testing Audio Processor")
    print("="*50)
    
    # Initialize processor
    processor = AudioProcessor(target_sr=16000)
    
    # Test with a sample file (if available)
    import glob
    sample_files = glob.glob("data/ravdess_raw/Actor_01/*.wav")
    
    if sample_files:
        test_file = sample_files[0]
        print(f"\nTesting with: {test_file}")
        
        # Test validation
        is_valid, msg, duration = processor.validate_audio(test_file)
        print(f"Validation: {msg}")
        
        # Test inference preprocessing
        audio = processor.preprocess_for_inference(test_file)
        if audio is not None:
            print(f"Inference preprocessing successful")
            print(f"Final audio shape: {audio.shape}")
        
        # Test training preprocessing with augmentation
        result = processor.preprocess_for_training(test_file, augment=True)
        if result:
            print(f"\nTraining preprocessing:")
            print(f"  Original: {len(result['original'])/result['sr']:.2f}s")
            print(f"  Augmented versions: {len(result['augmented'])}")
            for aug_type, aug_audio in result['augmented']:
                print(f"    - {aug_type}: {len(aug_audio)/result['sr']:.2f}s")
    else:
        print("No test files found. Make sure dataset is in data/ravdess_raw/")