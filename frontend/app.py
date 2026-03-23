"""
frontend/app.py
Stunning Voice Stress Detection App with Pastel Dark Theme - FIXED VERSION
"""

import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Voice Stress Analyzer ✨",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# All other imports
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import joblib
import json
from pathlib import Path
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import uuid
import tempfile

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.audio_processor import AudioProcessor
from src.feature_extractor import FeatureExtractor

# Try to import sounddevice for recording
try:
    import sounddevice as sd
    import soundfile as sf
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False

# =============================================================================
# CUSTOM PASTEL DARK THEME - FIXED COLORS
# =============================================================================

# Define custom color palette (without alpha for Plotly compatibility)
PASTEL_DARK = {
    "bg_dark": "#1a1b2f",        # Deep dark blue-purple background
    "bg_card": "#252a41",         # Slightly lighter card background
    "bg_hover": "#2f3550",         # Hover state
    "accent_purple": "#b980f0",    # Soft pastel purple
    "accent_pink": "#ff9fdb",      # Soft pastel pink
    "accent_blue": "#80b3ff",      # Soft pastel blue
    "accent_green": "#9fdfb2",     # Soft pastel green
    "accent_red": "#ff9f9f",       # Soft pastel red
    "accent_yellow": "#ffe08c",    # Soft pastel yellow
    "text_primary": "#ffffff",     # White text
    "text_secondary": "#b4b9d6",   # Soft purple-gray text
    "text_muted": "#6b7399",       # Muted text
    "success": "#9fdfb2",          # Pastel green for success
    "warning": "#ffe08c",           # Pastel yellow for warning
    "danger": "#ff9f9f",            # Pastel red for danger
    "info": "#80b3ff"               # Pastel blue for info
}

# CSS with fixed color format
st.markdown(f"""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');
    
    /* Global styles */
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main app background */
    .stApp {{
        background: linear-gradient(135deg, {PASTEL_DARK['bg_dark']} 0%, #1e1f35 100%);
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {PASTEL_DARK['bg_card']};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {PASTEL_DARK['accent_purple']};
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {PASTEL_DARK['accent_pink']};
    }}
    
    /* Headers */
    .main-header {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, {PASTEL_DARK['accent_purple']}, {PASTEL_DARK['accent_pink']}, {PASTEL_DARK['accent_blue']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease;
        text-shadow: 0 0 30px rgba(185, 128, 240, 0.3);
    }}
    
    .sub-header {{
        text-align: center;
        color: {PASTEL_DARK['text_secondary']};
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease;
        letter-spacing: 0.5px;
    }}
    
    /* Cards */
    .stCard {{
        background: {PASTEL_DARK['bg_card']};
        border-radius: 25px;
        padding: 25px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        margin: 15px 0;
        border: 1px solid rgba(185, 128, 240, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeIn 0.8s ease;
        backdrop-filter: blur(10px);
    }}
    
    .stCard:hover {{
        transform: translateY(-5px) scale(1.02);
        border-color: {PASTEL_DARK['accent_purple']};
        box-shadow: 0 30px 60px rgba(185, 128, 240, 0.3);
    }}
    
    /* Result cards */
    .stress-high {{
        background: linear-gradient(135deg, rgba(255, 159, 159, 0.2), rgba(255, 159, 219, 0.2));
        color: {PASTEL_DARK['accent_red']};
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        border: 2px solid rgba(255, 159, 159, 0.3);
        animation: pulse 2s infinite;
        backdrop-filter: blur(10px);
        letter-spacing: 2px;
    }}
    
    .stress-low {{
        background: linear-gradient(135deg, rgba(159, 223, 178, 0.2), rgba(128, 179, 255, 0.2));
        color: {PASTEL_DARK['accent_green']};
        padding: 30px;
        border-radius: 25px;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        border: 2px solid rgba(159, 223, 178, 0.3);
        backdrop-filter: blur(10px);
        letter-spacing: 2px;
    }}
    
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.02); opacity: 0.9; }}
        100% {{ transform: scale(1); opacity: 1; }}
    }}
    
    /* Confidence bar */
    .confidence-container {{
        background: {PASTEL_DARK['bg_dark']};
        border-radius: 30px;
        height: 30px;
        margin: 20px 0;
        overflow: hidden;
        border: 1px solid rgba(185, 128, 240, 0.3);
    }}
    
    .confidence-bar {{
        height: 100%;
        border-radius: 30px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    
    /* Metric cards */
    .metric-card {{
        background: linear-gradient(135deg, {PASTEL_DARK['bg_card']}, {PASTEL_DARK['bg_hover']});
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(185, 128, 240, 0.2);
        transition: all 0.3s ease;
        animation: slideInUp 0.6s ease;
    }}
    
    .metric-card:hover {{
        border-color: {PASTEL_DARK['accent_purple']};
        transform: translateY(-3px);
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {PASTEL_DARK['accent_purple']}, {PASTEL_DARK['accent_pink']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }}
    
    .metric-label {{
        color: {PASTEL_DARK['text_secondary']};
        font-size: 1rem;
        font-weight: 500;
        margin-top: 8px;
    }}
    
    /* Recording animation */
    .recording-indicator {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 12px;
        margin: 30px 0;
    }}
    
    .recording-dot {{
        width: 20px;
        height: 20px;
        background: {PASTEL_DARK['accent_red']};
        border-radius: 50%;
        animation: recordingPulse 1.2s ease-in-out infinite;
        box-shadow: 0 0 20px {PASTEL_DARK['accent_red']};
    }}
    
    .recording-dot:nth-child(2) {{
        animation-delay: 0.2s;
    }}
    
    .recording-dot:nth-child(3) {{
        animation-delay: 0.4s;
    }}
    
    @keyframes recordingPulse {{
        0%, 100% {{ transform: scale(1); opacity: 1; }}
        50% {{ transform: scale(1.5); opacity: 0.7; }}
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {PASTEL_DARK['accent_purple']}, {PASTEL_DARK['accent_pink']}) !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 10px 20px rgba(185, 128, 240, 0.3) !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 30px rgba(185, 128, 240, 0.5) !important;
    }}
    
    /* File uploader */
    .uploadfile {{
        border: 2px dashed rgba(185, 128, 240, 0.3) !important;
        border-radius: 30px !important;
        padding: 40px !important;
        text-align: center;
        background: {PASTEL_DARK['bg_card']} !important;
        transition: all 0.3s ease;
    }}
    
    .uploadfile:hover {{
        border-color: {PASTEL_DARK['accent_purple']} !important;
        background: {PASTEL_DARK['bg_hover']} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background: {PASTEL_DARK['bg_card']};
        padding: 15px;
        border-radius: 40px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 30px;
        padding: 10px 25px;
        color: {PASTEL_DARK['text_secondary']};
        transition: all 0.3s ease;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(185, 128, 240, 0.3), rgba(255, 159, 219, 0.3)) !important;
        color: white !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {PASTEL_DARK['accent_purple']}, {PASTEL_DARK['accent_pink']}) !important;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    @keyframes fadeInDown {{
        from {{
            opacity: 0;
            transform: translateY(-30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(30px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    @keyframes slideInUp {{
        from {{
            transform: translateY(30px);
            opacity: 0;
        }}
        to {{
            transform: translateY(0);
            opacity: 1;
        }}
    }}
    
    /* Floating animation */
    .floating {{
        animation: floating 3s ease-in-out infinite;
    }}
    
    @keyframes floating {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    /* Responsive design */
    @media (max-width: 768px) {{
        .main-header {{
            font-size: 2.5rem;
        }}
        .stress-high, .stress-low {{
            font-size: 1.8rem;
            padding: 20px;
        }}
        .metric-value {{
            font-size: 1.8rem;
        }}
    }}
    
    @media (max-width: 480px) {{
        .main-header {{
            font-size: 2rem;
        }}
        .sub-header {{
            font-size: 1rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS
# =============================================================================

@st.cache_resource(show_spinner="Loading AI models... ✨")
def load_models():
    """Load trained model and scaler with caching"""
    models_path = Path(__file__).parent.parent / "models"
    
    model_path = models_path / "stress_model.pkl"
    scaler_path = models_path / "scaler.pkl"
    summary_path = models_path / "performance_summary.json"
    
    if not model_path.exists() or not scaler_path.exists():
        st.error("❌ Model files not found! Please train the model first.")
        return None, None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Load performance summary if exists
    summary = None
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    return model, scaler, summary

@st.cache_resource
def init_processors():
    """Initialize audio processors with caching"""
    audio_processor = AudioProcessor(target_sr=16000)
    feature_extractor = FeatureExtractor(sr=16000, n_segments=10)
    return audio_processor, feature_extractor

# =============================================================================
# AUDIO PROCESSING FUNCTIONS - FIXED TEMP FILE HANDLING
# =============================================================================

def record_audio(duration=5, fs=16000):
    """Record audio from microphone with animation"""
    try:
        # Create recording animation
        with st.container():
            st.markdown("<div class='recording-indicator'>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    st.markdown(f"<div class='recording-dot'></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Countdown
            for i in range(duration):
                status_text.markdown(f"<h3 style='text-align: center; color: {PASTEL_DARK['text_secondary']};'>Recording... {duration-i}s</h3>", unsafe_allow_html=True)
                progress_bar.progress((i + 1) / duration)
                time.sleep(1)
        
        # Record audio
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
        
        progress_bar.empty()
        status_text.empty()
        st.success("✅ Recording complete!")
        st.balloons()
        
        return recording.flatten(), fs
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None, None

def process_audio_data(audio_data, sr, audio_processor, feature_extractor, scaler, model):
    """Process audio data and make prediction"""
    try:
        # Preprocess audio
        audio = audio_processor.reduce_noise(audio_data, sr)
        audio = audio_processor.trim_silence(audio, sr)
        
        # Extract features
        features = feature_extractor.extract_all_features(audio)
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]
        else:
            probabilities = np.array([0.5, 0.5])
            confidence = 0.5
        
        return audio, features, prediction, confidence, probabilities, len(audio)/sr
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None, None, None, None, None, None

def process_audio_file(uploaded_file, audio_processor, feature_extractor, scaler, model):
    """Process uploaded audio file and make prediction - FIXED VERSION"""
    
    # Use tempfile for automatic cleanup
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        temp_path = tmp_file.name
    
    try:
        # Load audio
        audio, sr = audio_processor.load_audio(temp_path)
        if audio is None:
            return None, None, None, None, None, None
        
        # Process the audio data
        result = process_audio_data(audio, sr, audio_processor, feature_extractor, scaler, model)
        return result
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None, None, None, None
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

# =============================================================================
# VISUALIZATION FUNCTIONS - FIXED PLOTLY COLORS
# =============================================================================

def create_gauge_chart(confidence, prediction):
    """Create beautiful gauge chart - FIXED COLOR FORMAT"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'size': 24, 'color': PASTEL_DARK['text_primary']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': PASTEL_DARK['text_secondary']},
            'bar': {'color': PASTEL_DARK['accent_purple'] if prediction == 0 else PASTEL_DARK['accent_red']},
            'bgcolor': PASTEL_DARK['bg_card'],
            'borderwidth': 2,
            'bordercolor': PASTEL_DARK['accent_purple'],  # FIXED: removed alpha
            'steps': [
                {'range': [0, 50], 'color': PASTEL_DARK['accent_green']},
                {'range': [50, 100], 'color': PASTEL_DARK['accent_red']}
            ],
            'threshold': {
                'line': {'color': PASTEL_DARK['accent_yellow'], 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': PASTEL_DARK['text_primary']},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def plot_waveform(audio, sr):
    """Plot audio waveform with pastel theme"""
    fig, ax = plt.subplots(figsize=(12, 3), facecolor=PASTEL_DARK['bg_card'])
    ax.set_facecolor(PASTEL_DARK['bg_dark'])
    
    time = np.linspace(0, len(audio)/sr, len(audio))
    ax.plot(time, audio, color=PASTEL_DARK['accent_purple'], alpha=0.8, linewidth=1.5)
    ax.fill_between(time, audio, alpha=0.2, color=PASTEL_DARK['accent_purple'])
    
    ax.set_xlabel('Time (s)', color=PASTEL_DARK['text_secondary'], fontsize=12)
    ax.set_ylabel('Amplitude', color=PASTEL_DARK['text_secondary'], fontsize=12)
    ax.set_title('Waveform', color=PASTEL_DARK['text_primary'], fontsize=14, pad=15)
    
    ax.tick_params(colors=PASTEL_DARK['text_secondary'])
    for spine in ax.spines.values():
        spine.set_color(PASTEL_DARK['accent_purple'])
    
    ax.grid(True, alpha=0.2, color=PASTEL_DARK['text_muted'])
    fig.patch.set_facecolor(PASTEL_DARK['bg_card'])
    
    return fig

def plot_spectrogram(audio, sr):
    """Plot spectrogram with pastel theme"""
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=PASTEL_DARK['bg_card'])
    ax.set_facecolor(PASTEL_DARK['bg_dark'])
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', ax=ax, sr=sr,
                                   cmap='plasma')
    
    ax.set_title('Spectrogram', color=PASTEL_DARK['text_primary'], fontsize=14, pad=15)
    ax.set_xlabel('Time (s)', color=PASTEL_DARK['text_secondary'])
    ax.set_ylabel('Frequency (Hz)', color=PASTEL_DARK['text_secondary'])
    
    ax.tick_params(colors=PASTEL_DARK['text_secondary'])
    for spine in ax.spines.values():
        spine.set_color(PASTEL_DARK['accent_purple'])
    
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Intensity (dB)', color=PASTEL_DARK['text_secondary'])
    cbar.ax.yaxis.set_tick_params(color=PASTEL_DARK['text_secondary'])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=PASTEL_DARK['text_secondary'])
    
    fig.patch.set_facecolor(PASTEL_DARK['bg_card'])
    
    return fig

def plot_pitch_contour(audio, sr):
    """Plot pitch contour with pastel theme"""
    fig, ax = plt.subplots(figsize=(12, 3), facecolor=PASTEL_DARK['bg_card'])
    ax.set_facecolor(PASTEL_DARK['bg_dark'])
    
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr,
        fill_na=0.0
    )
    times = librosa.times_like(f0)
    
    ax.plot(times, f0, color=PASTEL_DARK['accent_pink'], alpha=0.9, linewidth=2)
    ax.fill_between(times, f0, alpha=0.2, color=PASTEL_DARK['accent_pink'])
    
    ax.set_xlabel('Time (s)', color=PASTEL_DARK['text_secondary'], fontsize=12)
    ax.set_ylabel('Frequency (Hz)', color=PASTEL_DARK['text_secondary'], fontsize=12)
    ax.set_title('Pitch Contour', color=PASTEL_DARK['text_primary'], fontsize=14, pad=15)
    
    ax.tick_params(colors=PASTEL_DARK['text_secondary'])
    for spine in ax.spines.values():
        spine.set_color(PASTEL_DARK['accent_purple'])
    
    ax.grid(True, alpha=0.2, color=PASTEL_DARK['text_muted'])
    fig.patch.set_facecolor(PASTEL_DARK['bg_card'])
    
    return fig

def save_history(prediction, confidence, duration):
    """Save prediction to history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'prediction': "Stress Detected" if prediction == 1 else "No Stress",
        'confidence': f"{confidence:.2%}",
        'duration': f"{duration:.2f}s"
    })

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main Streamlit app"""
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Header with floating animation
    st.markdown("""
    <div style='text-align: center;'>
        <h1 class='main-header floating'>🎤 Voice Stress Analyzer Pro</h1>
        <p class='sub-header'>✨ AI-Powered Emotional State Detection from Voice ✨</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    model, scaler, summary = load_models()
    audio_processor, feature_extractor = init_processors()
    
    if model is None or scaler is None:
        st.error("❌ Models not loaded. Please train the model first.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "🎙️ Record", "📊 History", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.container():
                st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                st.subheader("📂 Upload Audio File")
                
                uploaded_file = st.file_uploader(
                    "Choose an audio file",
                    type=['wav', 'mp3', 'm4a', 'flac'],
                    help="Upload a clear voice recording (max 30 seconds)",
                    key="file_uploader"
                )
                
                if uploaded_file is not None:
                    st.audio(uploaded_file, format='audio/wav')
                    
                    if st.button("🔍 Analyze Audio", key="analyze_upload", use_container_width=True):
                        with st.spinner("🔄 Processing audio... Please wait..."):
                            result = process_audio_file(
                                uploaded_file, audio_processor, feature_extractor, 
                                scaler, model
                            )
                            
                            if result[0] is not None:
                                audio, features, prediction, confidence, probabilities, duration = result
                                
                                st.session_state['audio'] = audio
                                st.session_state['prediction'] = prediction
                                st.session_state['confidence'] = confidence
                                st.session_state['probabilities'] = probabilities
                                st.session_state['duration'] = duration
                                
                                save_history(prediction, confidence, duration)
                                st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                st.subheader("ℹ️ Quick Guide")
                
                st.info("""
                **📌 How to use:**
                1. Click 'Browse files'
                2. Select your audio file
                3. Click 'Analyze'
                4. View results
                
                **✅ Best results:**
                • Clear voice
                • No background noise
                • 3-5 seconds duration
                • Natural speech
                
                **📁 Supported formats:**
                • WAV, MP3, M4A, FLAC
                """)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        if not RECORDING_AVAILABLE:
            st.warning("⚠️ Recording feature requires additional packages. Install with:")
            st.code("pip install sounddevice soundfile")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.container():
                    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                    st.subheader("🎙️ Live Voice Recording")
                    
                    duration = st.slider("Recording duration (seconds)", 2, 10, 5)
                    
                    if st.button("⏺️ Start Recording", key="record_btn", use_container_width=True):
                        audio_data, sr = record_audio(duration)
                        
                        if audio_data is not None:
                            with st.spinner("🔄 Analyzing your voice..."):
                                result = process_audio_data(
                                    audio_data, sr, audio_processor, feature_extractor,
                                    scaler, model
                                )
                                
                                if result[0] is not None:
                                    audio, features, prediction, confidence, probabilities, duration = result
                                    
                                    st.session_state['audio'] = audio
                                    st.session_state['prediction'] = prediction
                                    st.session_state['confidence'] = confidence
                                    st.session_state['probabilities'] = probabilities
                                    st.session_state['duration'] = duration
                                    
                                    save_history(prediction, confidence, duration)
                                    st.rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                with st.container():
                    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                    st.subheader("🎯 Tips")
                    
                    st.info("""
                    **For accurate results:**
                    
                    😌 **No Stress:**
                    • Speak calmly
                    • Normal pace
                    • Relaxed tone
                    
                    😰 **Stress Test:**
                    • Speak faster
                    • Higher pitch
                    • More energy
                    
                    🎯 **Pro tip:** Try both styles!
                    """)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display results if available
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center; color: {PASTEL_DARK['text_primary']}'>📊 Analysis Results</h2>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            with st.container():
                st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                
                # Result
                if st.session_state.prediction == 1:
                    st.markdown(f"<div class='stress-high'>🔴 STRESS DETECTED</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='stress-low'>🟢 NO STRESS DETECTED</div>", unsafe_allow_html=True)
                
                # Confidence gauge
                fig = create_gauge_chart(st.session_state.confidence, st.session_state.prediction)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                
                # Metrics
                st.subheader("📈 Detailed Analysis")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{st.session_state.confidence:.1%}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Confidence</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col_b:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{st.session_state.duration:.1f}s</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Duration</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Probability bars
                st.subheader("Probability Distribution")
                
                prob_no_stress = st.session_state.probabilities[0]
                prob_stress = st.session_state.probabilities[1]
                
                # No Stress bar
                st.markdown(f"<p style='color: {PASTEL_DARK['text_secondary']}'>No Stress</p>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='confidence-container'>
                    <div class='confidence-bar' style='width: {prob_no_stress*100}%; background: linear-gradient(90deg, {PASTEL_DARK['accent_green']}, {PASTEL_DARK['accent_blue']});'>
                        {prob_no_stress:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Stress bar
                st.markdown(f"<p style='color: {PASTEL_DARK['text_secondary']}'>Stress</p>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='confidence-container'>
                    <div class='confidence-bar' style='width: {prob_stress*100}%; background: linear-gradient(90deg, {PASTEL_DARK['accent_red']}, {PASTEL_DARK['accent_pink']});'>
                        {prob_stress:.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualizations
        if 'audio' in st.session_state:
            st.markdown("---")
            st.subheader("🎨 Audio Visualizations")
            
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Waveform", "Spectrogram", "Pitch Contour"])
            
            with viz_tab1:
                fig = plot_waveform(st.session_state.audio, 16000)
                st.pyplot(fig)
                plt.close()
            
            with viz_tab2:
                fig = plot_spectrogram(st.session_state.audio, 16000)
                st.pyplot(fig)
                plt.close()
            
            with viz_tab3:
                fig = plot_pitch_contour(st.session_state.audio, 16000)
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        with st.container():
            st.markdown("<div class='stCard'>", unsafe_allow_html=True)
            st.subheader("📊 Analysis History")
            
            if st.session_state.history:
                # Summary metrics
                df = pd.DataFrame(st.session_state.history)
                total = len(df)
                stress_count = len(df[df['prediction'] == 'Stress Detected'])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{total}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Total Analyses</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{stress_count}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Stress Detected</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    stress_rate = (stress_count/total)*100
                    st.markdown(f"<div class='metric-value'>{stress_rate:.1f}%</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Stress Rate</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # History table
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "timestamp": "Time",
                        "prediction": "Result",
                        "confidence": "Confidence",
                        "duration": "Duration"
                    }
                )
                
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()
            else:
                st.info("No analysis history yet. Upload or record audio to get started!")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.container():
                st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                st.subheader("🎯 About This Project")
                
                st.markdown("""
                ### Voice Stress Detection using AI
                
                This application uses machine learning to detect stress levels from voice patterns. 
                It analyzes various acoustic features including:
                
                * **MFCCs** - Mel-frequency cepstral coefficients
                * **Spectral Features** - Centroid, bandwidth, contrast
                * **Prosodic Features** - Pitch, energy, rhythm
                * **Temporal Dynamics** - Changes over time
                
                ### How It Works
                1. Audio is preprocessed (noise reduction, silence trimming)
                2. 2120 features are extracted from 10 temporal segments
                3. SVM model classifies as stress/no-stress
                4. Confidence scores are calculated
                
                ### Dataset
                Trained on RAVDESS dataset with 1440 audio samples
                """)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                st.subheader("📊 Model Performance")
                
                if summary:
                    metrics = summary.get('best_model_metrics', {})
                    
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                    st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
                    st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
                    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
                    
                    st.caption(f"🏆 Best Model: {summary.get('best_model', 'N/A')}")
                    
                    # Confusion matrix
                    if 'confusion_matrix' in summary:
                        st.subheader("Confusion Matrix")
                        cm = np.array(summary['confusion_matrix'])
                        fig, ax = plt.subplots(figsize=(6, 4), facecolor=PASTEL_DARK['bg_card'])
                        ax.set_facecolor(PASTEL_DARK['bg_dark'])
                        sns.heatmap(cm, annot=True, fmt='d', cmap='plasma', ax=ax,
                                   cbar_kws={'label': 'Count'})
                        ax.set_xlabel('Predicted', color=PASTEL_DARK['text_secondary'])
                        ax.set_ylabel('Actual', color=PASTEL_DARK['text_secondary'])
                        ax.set_xticklabels(['No Stress', 'Stress'], color=PASTEL_DARK['text_secondary'])
                        ax.set_yticklabels(['No Stress', 'Stress'], color=PASTEL_DARK['text_secondary'])
                        fig.patch.set_facecolor(PASTEL_DARK['bg_card'])
                        st.pyplot(fig)
                        plt.close()
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"""
        <p style='text-align: center; color: {PASTEL_DARK['text_muted']}; padding: 20px;'>
            Built with ❤️ using Streamlit | RAVDESS Dataset | SVM Model | Pastel Dark Theme ✨
        </p>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()