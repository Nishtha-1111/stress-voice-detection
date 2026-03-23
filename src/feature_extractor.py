import numpy as np
import librosa


class FeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=13, n_segments=3):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_segments = n_segments

    def segment_audio(self, audio):
        """
        Split audio into equal segments
        """
        segment_length = len(audio) // self.n_segments
        segments = []

        for i in range(self.n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < self.n_segments - 1 else len(audio)
            segments.append(audio[start:end])

        return segments

    def extract_mfcc_features(self, audio):
        """
        Extract MFCC features with deltas and delta-deltas
        Handles short audio segments gracefully
        """
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )

        min_frames_needed = 9

        if mfccs.shape[1] < min_frames_needed:
            pad_width = min_frames_needed - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='edge')

        try:
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        except:
            print("⚠️ Warning: Delta computation failed, using zeros")
            mfcc_delta = np.zeros_like(mfccs)
            mfcc_delta2 = np.zeros_like(mfccs)

        mfcc_features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])

        return mfcc_features

    def extract_spectral_features(self, audio):
        """
        Extract spectral features
        """
        features = {}

        try:
            features['centroid'] = librosa.feature.spectral_centroid(
                y=audio, sr=self.sr
            )[0]
        except:
            features['centroid'] = np.zeros(10)

        try:
            features['bandwidth'] = librosa.feature.spectral_bandwidth(
                y=audio, sr=self.sr
            )[0]
        except:
            features['bandwidth'] = np.zeros(10)

        try:
            contrast = librosa.feature.spectral_contrast(
                y=audio, sr=self.sr
            )

            for i in range(contrast.shape[0]):
                features[f'contrast_{i}'] = contrast[i]

        except:
            for i in range(6):
                features[f'contrast_{i}'] = np.zeros(10)

        try:
            features['rolloff'] = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sr
            )[0]
        except:
            features['rolloff'] = np.zeros(10)

        try:
            features['flatness'] = librosa.feature.spectral_flatness(
                y=audio
            )[0]
        except:
            features['flatness'] = np.zeros(10)

        return features

    def extract_prosodic_features(self, audio):
        """
        Extract prosodic features (pitch, energy, zero-crossing rate)
        With robust error handling
        """
        features = {}

        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sr,
                fill_na=0.0
            )

            f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
            features['pitch'] = f0

        except Exception as e:
            print(f"⚠️ Pitch extraction failed: {e}")
            n_frames = len(audio) // 512 + 1
            features['pitch'] = np.zeros(n_frames)

        try:
            features['rms'] = librosa.feature.rms(y=audio)[0]
        except:
            features['rms'] = np.zeros(10)

        try:
            features['zcr'] = librosa.feature.zero_crossing_rate(audio)[0]
        except:
            features['zcr'] = np.zeros(10)

        return features

    def compute_segment_statistics(self, features_per_segment):
        """
        Compute statistics for each segment with robust error handling
        """
        feature_vector = []

        for segment_idx, segment_features in enumerate(features_per_segment):

            for feature_name, feature_values in segment_features.items():

                try:

                    if isinstance(feature_values, np.ndarray):

                        if feature_values.ndim > 1:

                            for dim_values in feature_values:

                                if len(dim_values) > 0:
                                    stats_vector = [
                                        np.mean(dim_values),
                                        np.std(dim_values) if len(dim_values) > 1 else 0,
                                        np.min(dim_values),
                                        np.max(dim_values)
                                    ]
                                else:
                                    stats_vector = [0, 0, 0, 0]

                                feature_vector.extend(stats_vector)

                        else:

                            if len(feature_values) > 0:
                                stats_vector = [
                                    np.mean(feature_values),
                                    np.std(feature_values) if len(feature_values) > 1 else 0,
                                    np.min(feature_values),
                                    np.max(feature_values)
                                ]
                            else:
                                stats_vector = [0, 0, 0, 0]

                            feature_vector.extend(stats_vector)

                    else:

                        feature_vector.extend([float(feature_values), 0, 0, 0])

                except Exception as e:
                    print(f"⚠️ Error processing {feature_name}: {e}")
                    feature_vector.extend([0, 0, 0, 0])

        return np.array(feature_vector)

    def extract_all_features(self, audio):
        """
        Extract all features with temporal segmentation
        Includes error handling for robustness
        """

        min_audio_length = 0.5 * self.sr

        if len(audio) < min_audio_length:
            print(f"⚠️ Audio too short ({len(audio)/self.sr:.2f}s), padding...")
            padding = int(min_audio_length - len(audio))
            audio = np.pad(audio, (0, padding), mode='constant')

        segments = self.segment_audio(audio)

        features_per_segment = []

        for segment_idx, segment in enumerate(segments):

            segment_features = {}

            try:

                mfcc = self.extract_mfcc_features(segment)
                segment_features['mfcc'] = mfcc

                spectral = self.extract_spectral_features(segment)
                segment_features.update(spectral)

                prosodic = self.extract_prosodic_features(segment)
                segment_features.update(prosodic)

            except Exception as e:

                print(f"⚠️ Error extracting features from segment {segment_idx}: {e}")

                segment_features = {
                    'mfcc': np.zeros((39, 10)),
                    'centroid': np.zeros(10),
                    'bandwidth': np.zeros(10),
                    'contrast_0': np.zeros(10),
                    'contrast_1': np.zeros(10),
                    'contrast_2': np.zeros(10),
                    'contrast_3': np.zeros(10),
                    'contrast_4': np.zeros(10),
                    'contrast_5': np.zeros(10),
                    'rolloff': np.zeros(10),
                    'flatness': np.zeros(10),
                    'pitch': np.zeros(10),
                    'rms': np.zeros(10),
                    'zcr': np.zeros(10)
                }

            features_per_segment.append(segment_features)

        feature_vector = self.compute_segment_statistics(features_per_segment)

        return feature_vector