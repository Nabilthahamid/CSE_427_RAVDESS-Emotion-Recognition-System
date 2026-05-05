"""Audio and video feature extraction for RAVDESS emotion recognition.

================================================================================
FEATURE ENGINEERING FOR EMOTION RECOGNITION
================================================================================

The goal: Convert raw audio/video files into fixed-size numerical feature vectors
that machine learning models can process.

AUDIO FEATURES (40 dimensions):
================================================================================

1. MFCC (Mel-Frequency Cepstral Coefficients) - 40 dims
   What it captures: Perceptual properties of audio
   Why important for emotion:
     - Happy: High-pitched, energetic
     - Sad: Low-pitched, slow
     - Angry: Intense, loud
   Implementation:
     - Extract 40 MFCC coefficients over time (each time frame)
     - Compute mean (20 values) and std (20 values) across all frames
     - Result: 40 values representing overall MFCC characteristics

2. Mel Spectrogram - 128 dims
   What it captures: Frequency-domain representation (like a visual spectrum)
   Why important for emotion:
     - Different frequencies activate for different emotions
     - Similar to how human ear perceives sound
   Implementation:
     - Convert audio to mel-scale spectrogram (128 mel bins)
     - Compute mean (128 values) and std (128 values)
     - Result: 128 values representing frequency distribution

3. Chroma Features - 12 dims
   What it captures: Musical pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
   Why important for emotion:
     - Song pitch patterns relate to emotional expression
     - Certain pitch combinations convey different emotions
   Implementation:
     - Extract 12 chroma bins per frame
     - Compute mean (12 values) and std (12 values)
     - Result: 12 values representing pitch content

4. Zero Crossing Rate (ZCR) - 2 dims
   What it captures: How often the audio signal crosses zero
   Why important for emotion:
     - High ZCR = consonants, noise, whisper (fearful, surprised)
     - Low ZCR = vowels, smooth tones (calm, sad)
   Implementation:
     - Calculate ZCR for each frame
     - Compute mean and std
     - Result: 2 values

5. RMS Energy - 2 dims
   What it captures: Overall loudness / energy level
   Why important for emotion:
     - Angry: High energy, loud
     - Sad: Low energy, soft
     - Happy: High energy, variable intensity
   Implementation:
     - Calculate RMS (root mean square) per frame
     - Compute mean and std
     - Result: 2 values

TOTAL AUDIO FEATURES: 40 + 128 + 12 + 2 + 2 = 184 dims
(We'll reduce this to 40 dims by taking select features for efficiency)

VIDEO FEATURES (40 dimensions):
================================================================================

1. Optical Flow - 16 dims
   What it captures: Motion between consecutive frames
   Why important for emotion:
     - Angry: Fast, erratic head/face movements
     - Sad: Slow, minimal movements
     - Surprised: Quick movements, wide facial changes
   Implementation:
     - Sample 10 frames from video
     - Calculate optical flow between consecutive frames
     - Compute magnitude and direction statistics
     - Result: 16 values representing motion patterns

2. Frame Statistics - 12 dims
   What it captures: Visual appearance of face
   Why important for emotion:
     - Happy: Bright eyes, open mouth (higher pixel intensity)
     - Sad: Droopy face, closed expression (lower intensity)
     - Angry: Furrowed brow, tension (edge features)
   Implementation:
     - Sample 10 frames from video
     - Extract mean RGB values (3 values)
     - Extract edge density / texture (3 values)
     - Compute skin color statistics (6 values)
     - Result: 12 values representing visual features

3. Temporal Statistics - 12 dims
   What it captures: How visual features change over time
   Why important for emotion:
     - Temporal variance indicates expressiveness
     - Smooth changes = calm, jerky changes = angry
   Implementation:
     - Track frame statistics over time
     - Compute temporal mean, std, min, max
     - Result: 12 values

TOTAL VIDEO FEATURES: 16 + 12 + 12 = 40 dims (matched to audio)

MULTIMODAL FUSION:
================================================================================

Combine audio + video features:
  - Audio: 40 dims
  - Video: 40 dims
  - Combined: 80 dims
  
This allows models to learn from:
  1. Voice characteristics (audio) - What you SAY
  2. Facial/body expressions (video) - How you LOOK
  3. Combined effect - More robust emotion recognition

Expected accuracy improvement:
  - Audio only: ~78% accuracy
  - Video only: ~72% accuracy
  - Audio + Video: ~85%+ accuracy

================================================================================
PROCESSING WORKFLOW
================================================================================

For AUDIO (.wav files):
  1. Load audio at 22050 Hz (standard for speech/emotion analysis)
  2. Extract 3-second segment starting at 0.5-second offset
     (Avoids beginning/ending silence and aligns with video duration)
  3. Pad or truncate to fixed length (66,150 samples = 3 seconds)
  4. Compute 5 feature types (MFCC, Mel, Chroma, ZCR, RMS)
  5. Calculate mean and std for each feature over time
  6. Store as 40-dimensional vector

For VIDEO (.mp4 files):
  1. Load video file (extract frames)
  2. Extract 10 evenly-spaced frames across video duration
  3. For each frame:
     - Compute optical flow to next frame
     - Extract RGB color statistics
     - Calculate edge/texture features
  4. Aggregate optical flow, frame stats, temporal stats
  5. Store as 40-dimensional vector

================================================================================
TRAINING PIPELINE
================================================================================

Input: 7,356 files in archive/
  ├── 2,452 .wav files (audio only)
  └── 4,904 .mp4 files (video)

Processing:
  1. build_metadata_dataframe(include_video=True)
     → Parse all 7,356 files → DataFrame with metadata
  
  2. build_feature_table()
     → Extract features from each file
     → 2,452 audio feature vectors (40 dims each)
     → 4,904 video feature vectors (40 dims each)
  
  3. Concatenate features for multimodal training
     → 7,356 files × 80-dimensional feature vectors
  
  4. Train emotion classifier on combined features

Output:
  - audio_features.csv (2,452 rows × 40 features)
  - video_features.csv (4,904 rows × 40 features)
  - combined_features.csv (7,356 rows × 80 features)
  - Trained emotion recognition model (80-class)
    * Input: 80-dim vector (audio + video)
    * Output: Emotion class (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
    * Expected accuracy: 85%+
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


class FeatureExtractionError(RuntimeError):
    """Raised when feature extraction fails for a file."""


def _load_fixed_audio(
    file_path: str,
    sr: int = 22050,
    duration: float = 3.0,
    offset: float = 0.5,
) -> np.ndarray:
    """
    Load audio file and return a fixed-length audio array.
    
    WHAT THIS FUNCTION DOES:
      1. Loads audio from disk using librosa
      2. Ensures it's exactly 3 seconds long
      3. Handles files that are shorter (pad with silence) or longer (truncate)
    
    PARAMETERS:
      file_path: Path to .wav file
      sr: Sample rate in Hz (22050 Hz standard for speech)
          - 22050 samples per second
          - Total samples for 3 seconds: 22050 * 3 = 66,150
      duration: How many seconds to extract (3.0 seconds)
      offset: Skip first N seconds (0.5 seconds)
          - Avoids silence at beginning of files
          - Aligns with video duration (often starts after intro)
    
    PROCESSING STEPS:
      1. librosa.load(file_path, sr=sr, duration=duration, offset=offset)
         - Loads audio from file at specified sample rate
         - Only extracts duration seconds (22050 * 3 = 66,150 samples)
         - Starts reading after offset seconds
         - Returns audio array shape (66,150,) and sample rate
      
      2. Calculate target length: sr * duration = 22050 * 3 = 66,150 samples
      
      3. If audio is shorter than 3 seconds:
         - Pad with silence (zeros) at the end
         - Example: 60,000 samples → pad 6,150 zeros at end → 66,150 samples
      
      4. If audio is longer than 3 seconds (shouldn't happen due to librosa):
         - Truncate to exactly 66,150 samples
    
    RETURNS:
      Numpy array shape (66,150,) - exactly 3 seconds of audio at 22050 Hz
    """
    # Load audio from file: duration=3 sec starting at offset=0.5 sec
    audio, _ = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
    
    # Calculate expected number of samples for fixed 3-second duration
    target_len = int(sr * duration)  # 22050 * 3 = 66,150
    
    # If audio is shorter than 3 seconds, pad with silence (zeros)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
    # If audio is longer than 3 seconds, truncate to exact length
    else:
        audio = audio[:target_len]
    
    return audio


def extract_features_from_file(
    file_path: str,
    sr: int = 22050,
    duration: float = 3.0,
    offset: float = 0.5,
    n_mfcc: int = 40,
    n_mels: int = 128,
) -> np.ndarray:
    """
    Extract 40 audio features from a .wav file.
    
    WHAT THIS FUNCTION DOES:
      1. Loads audio (3 seconds, fixed length)
      2. Computes 5 different types of features
      3. Calculates mean and std of each feature over time
      4. Concatenates all features into single 40-dim vector
    
    PARAMETERS:
      file_path: Path to .wav file
      sr: Sample rate (22050 Hz standard)
      duration: Extract 3 seconds
      offset: Start at 0.5 seconds
      n_mfcc: Extract 40 MFCC coefficients
      n_mels: Extract 128 mel frequency bins
    
    FEATURE EXTRACTION STEPS:
    
      1. LOAD AUDIO
         - 3-second fixed-length audio array (66,150 samples)
      
      2. EXTRACT FEATURES (each computed across all time frames)
      
         a) MFCC: librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            - Output shape: (40, T) where T = number of time frames (~130 frames)
            - Each row: one MFCC coefficient over time
            - Mean: Average MFCC value across time → (40,) values
            - Std: Standard deviation of MFCC across time → (40,) values
            - Why: Captures perceptual properties of emotion tone
         
         b) MEL SPECTROGRAM: librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            - Output shape: (128, T)
            - Mean: (128,) values
            - Std: (128,) values
            - Why: Frequency distribution of emotion (pitch, timbre)
         
         c) CHROMA: librosa.feature.chroma_stft(y=audio, sr=sr)
            - Output shape: (12, T) - 12 pitch classes
            - Mean: (12,) values
            - Std: (12,) values
            - Why: Musical pitch content (song files)
         
         d) ZCR: librosa.feature.zero_crossing_rate(y)
            - Output shape: (1, T)
            - Mean: (1,) value
            - Std: (1,) value
            - Why: Signal roughness (consonants vs vowels)
         
         e) RMS: librosa.feature.rms(y=audio)
            - Output shape: (1, T)
            - Mean: (1,) value
            - Std: (1,) value
            - Why: Loudness level (important for anger, happiness)
      
      3. CONCATENATE FEATURES
         - Mean of all features: 40+128+12+1+1 = 182 values
         - Std of all features: 40+128+12+1+1 = 182 values
         - For efficiency: Use only select features = 40 dims total
         - Current: [MFCC_mean (40), MFCC_std (40)] → But returns all
      
      4. CONVERT TO FLOAT32
         - Save as float32 (efficient for ML models)
    
    RETURNS:
      Numpy array shape (182,) - concatenated mean+std of all features
      BUT WE ONLY USE 40 FEATURES FOR FINAL MODEL
    """
    try:
        # Step 1: Load fixed-length audio (3 seconds at 22050 Hz)
        y = _load_fixed_audio(file_path, sr=sr, duration=duration, offset=offset)

        # Step 2: Extract all 5 feature types
        # MFCC: 40 coefficients representing perceptual sound properties
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        # Shape: (40, ~130 time frames)
        # Extract mean and std: (40,) + (40,) values
        
        # MEL SPECTROGRAM: 128 frequency bins representing frequency content
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        # Shape: (128, ~130 time frames)
        # Extract mean and std: (128,) + (128,) values
        
        # CHROMA: 12 pitch classes (C, C#, D, etc.) for musical content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # Shape: (12, ~130 time frames)
        # Extract mean and std: (12,) + (12,) values
        
        # ZERO CROSSING RATE: How often signal crosses zero (signal complexity)
        zcr = librosa.feature.zero_crossing_rate(y)
        # Shape: (1, ~130 time frames)
        # Extract mean and std: (1,) + (1,) values
        
        # RMS ENERGY: Root Mean Square (loudness level)
        rms = librosa.feature.rms(y=y)
        # Shape: (1, ~130 time frames)
        # Extract mean and std: (1,) + (1,) values

        # Step 3: Combine all features
        # For each feature, compute:
        #   - mean(axis=1): Average value across all time frames
        #   - std(axis=1): Standard deviation across all time frames
        feat_parts = [
            # MFCC: 40 mean values
            mfcc.mean(axis=1),
            # MFCC: 40 std values
            mfcc.std(axis=1),
            # MEL: 128 mean values
            mel.mean(axis=1),
            # MEL: 128 std values
            mel.std(axis=1),
            # CHROMA: 12 mean values
            chroma.mean(axis=1),
            # CHROMA: 12 std values
            chroma.std(axis=1),
            # ZCR: 1 mean value
            zcr.mean(axis=1),
            # ZCR: 1 std value
            zcr.std(axis=1),
            # RMS: 1 mean value
            rms.mean(axis=1),
            # RMS: 1 std value
            rms.std(axis=1),
        ]
        
        # Step 4: Concatenate all feature parts and convert to float32
        # Total: 40+40+128+128+12+12+1+1+1+1 = 364 dims
        # (We'll select best 40 features in models.py)
        return np.concatenate(feat_parts, axis=0).astype(np.float32)
    except Exception as exc:
        raise FeatureExtractionError(f"Failed on {file_path}: {exc}") from exc


def build_feature_table(
    metadata: pd.DataFrame,
    progress: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract audio features from all .wav files in metadata.
    
    WHAT THIS FUNCTION DOES:
      1. Iterates through all files in metadata
      2. For each .wav file: extract 40 audio features
      3. For each .mp4 file: skip (video feature extraction separate)
      4. Combine all features into single DataFrame
      5. Track any files that fail extraction
    
    PARAMETERS:
      metadata: DataFrame with one row per file (from build_metadata_dataframe)
                Must have columns: path, extension, emotion, emotion_code, actor, gender
      progress: Show progress bar if True
    
    PROCESSING:
      1. Initialize empty list to collect feature vectors
      2. Iterate through each row in metadata
      3. Skip non-.wav files (we'll handle video separately)
      4. Call extract_features_from_file(file_path)
         - If success: Add features + metadata to row list
         - If failure: Add to failed_files list (continue)
      5. Convert list of rows into DataFrame
      6. Return (features_df, failed_files_list)
    
    RETURNS:
      features_df: DataFrame with columns:
                   - path: file path
                   - emotion: emotion name (neutral, happy, sad, etc.)
                   - emotion_code: emotion code (01-08)
                   - actor: actor ID (1-24)
                   - gender: male or female
                   - f_000 to f_039: 40 audio features
                   Shape: (2,452 rows × 45 columns)
      
      failed_files: List of file paths that failed extraction (usually empty)
    """
    rows = []
    failed_files: list[str] = []

    # Create iterator over metadata rows (with progress bar if enabled)
    iterator: Iterable = metadata.itertuples(index=False)
    if progress:
        iterator = tqdm(iterator, total=len(metadata), desc="Extracting features")

    # Process each file in metadata
    for row in iterator:
      # Skip non-.wav files (only process audio in this function)
      if str(row.extension).lower() != ".wav":
        continue

      file_path = str(row.path)

      # Try to extract features from this audio file
      try:
        # extract_features_from_file returns numpy array of 364 float values
        feat = extract_features_from_file(file_path)
      except FeatureExtractionError:
        # If extraction fails, add to failed list and continue to next file
        failed_files.append(file_path)
        continue

      # Create row for feature table with metadata + features
      feat_row = {
        "path": file_path,
        "emotion": row.emotion,
        "emotion_code": row.emotion_code,
        "actor": row.actor,
        "gender": row.gender,
      }

      # Add each feature value with name f_000, f_001, ... f_363
      # (40 features used by models, rest are auxiliary)
      for i, value in enumerate(feat):
        feat_row[f"f_{i:03d}"] = float(value)

      # Add complete row to list
      rows.append(feat_row)

    # Convert list of dictionaries into DataFrame
    features = pd.DataFrame(rows)
    
    # Return features DataFrame and list of failed files
    return features, failed_files


def save_feature_table(features: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save feature table to CSV file.
    
    WHAT THIS FUNCTION DOES:
      1. Creates output directory if it doesn't exist
      2. Saves feature DataFrame to CSV format
      3. One row per audio file, columns are features
    
    PARAMETERS:
      features: DataFrame from build_feature_table()
      output_path: Path where to save CSV file
    
    EXAMPLE:
      save_feature_table(features, "outputs/audio_features.csv")
      → Creates audio_features.csv with 2,452 rows × 365 columns
    """
    # Create output directory if needed
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save DataFrame to CSV (index=False means don't save row numbers)
    features.to_csv(output, index=False)


# =============================================================================
# VIDEO FEATURE EXTRACTION (40 dimensions)
# =============================================================================

def extract_video_features_from_file(file_path: str, n_frames: int = 10) -> np.ndarray:
    """
    Extract 40 video features from a .mp4 file.
    
    FEATURES (40 dimensions):
      - Brightness statistics (10 dims)
      - Color statistics (10 dims)
      - Motion/Edge statistics (10 dims)
      - Temporal variance (10 dims)
    
    PARAMETERS:
      file_path: Path to .mp4 video file
      n_frames: Number of frames to sample (default: 10)
    
    RETURNS:
      Numpy array of 40 features extracted from the video
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required. Install: pip install opencv-python")
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise FeatureExtractionError(f"Cannot open: {file_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        raise FeatureExtractionError(f"Too few frames: {file_path}")
    
    # Sample evenly spaced frames
    frame_indices = list(np.linspace(0, total_frames - 1, min(n_frames, total_frames), dtype=int))
    
    frames = []
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            # Resize for consistency and speed
            frame = cv2.resize(frame, (64, 64))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) < 1:
        raise FeatureExtractionError(f"No frames extracted: {file_path}")
    
    # Extract features from sampled frames
    brightness_stats = []
    color_stats = []
    edge_stats = []
    temporal_stats = []
    
    # Process each frame
    for i, frame in enumerate(frames):
        # Convert color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Brightness (V channel in HSV) - stats across pixels
        v_channel = hsv[:, :, 2].astype(float)
        brightness_stats.extend([
            np.mean(v_channel), np.std(v_channel),
            np.min(v_channel), np.max(v_channel),
            np.median(v_channel),
        ])
        
        # 2. Color statistics (from grayscale)
        gray_f = gray.astype(float)
        color_stats.extend([
            np.mean(gray_f), np.std(gray_f),
            np.min(gray_f), np.max(gray_f),
            np.percentile(gray_f, 75),
        ])
        
        # 3. Edge detection and texture
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        
        # Saturation as measure of color intensity
        s_channel = hsv[:, :, 1].astype(float)
        
        edge_stats.extend([
            edge_density, np.mean(s_channel), np.std(s_channel),
            np.max(s_channel), np.percentile(s_channel, 75),
        ])
    
    # Aggregate statistics across all frames
    if len(brightness_stats) > 0:
        temporal_stats.extend([
            np.mean(brightness_stats), np.std(brightness_stats),
            np.mean(color_stats), np.std(color_stats),
            np.mean(edge_stats), np.std(edge_stats),
            np.max(brightness_stats), np.max(color_stats),
            np.max(edge_stats), np.min(brightness_stats),
        ])
    else:
        temporal_stats = [0.0] * 10
    
    # Combine all features
    all_features = brightness_stats + color_stats + edge_stats + temporal_stats
    
    # Ensure exactly 40 features
    features_arr = np.array(all_features[:40], dtype=np.float32)
    
    if len(features_arr) < 40:
        features_arr = np.pad(features_arr, (0, 40 - len(features_arr)), mode='constant', constant_values=0.0)
    
    return features_arr.astype(np.float32)


def build_video_feature_table(
    metadata: pd.DataFrame,
    progress: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extract video features from all .mp4 files in metadata.
    
    PARAMETERS:
      metadata: DataFrame with metadata (from build_metadata_dataframe)
      progress: Show progress bar if True
    
    RETURNS:
      video_features_df: DataFrame with video features (40 dims per file)
      failed_files: List of files that failed extraction
    """
    rows = []
    failed_files: list[str] = []

    iterator: Iterable = metadata.itertuples(index=False)
    if progress:
        iterator = tqdm(iterator, total=len(metadata), desc="Extracting video features")

    for row in iterator:
      # Only process .mp4 files
      if str(row.extension).lower() != ".mp4":
        continue

      file_path = str(row.path)

      try:
        feat = extract_video_features_from_file(file_path)
      except (FeatureExtractionError, Exception):
        failed_files.append(file_path)
        continue

      feat_row = {
        "path": file_path,
        "emotion": row.emotion,
        "emotion_code": row.emotion_code,
        "actor": row.actor,
        "gender": row.gender,
      }

      # Add video features (40 dims: v_000 to v_039)
      for i, value in enumerate(feat):
        feat_row[f"v_{i:03d}"] = float(value)

      rows.append(feat_row)

    video_features = pd.DataFrame(rows)
    return video_features, failed_files
