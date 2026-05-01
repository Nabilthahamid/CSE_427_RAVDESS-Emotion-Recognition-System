"""Exploratory Data Analysis (EDA) utilities for RAVDESS metadata and features.

================================================================================
WHAT IS EXPLORATORY DATA ANALYSIS (EDA)?
================================================================================

EDA is the first step after loading data - we examine and visualize the dataset
to understand its characteristics, patterns, and potential issues.

WHY IS EDA IMPORTANT FOR EMOTION RECOGNITION?
================================================================================

1. CLASS BALANCE ANALYSIS:
   - Are all emotions equally represented?
   - If one emotion has 1,000 files and another has 100, models will be biased
   - Solution: Stratified sampling or class weighting in models

2. DURATION ANALYSIS:
   - Are all audio files the same length?
   - Do some emotions have significantly different durations?
   - Example: Angry might have faster speech (shorter) vs sad (slower)
   - Solution: Fixed-length feature extraction to handle variations

3. ACTOR DISTRIBUTION:
   - Are all actors equally represented?
   - If one actor has 200 files and another has 50, results may be biased
   - Solution: Actor-wise data splitting to prevent speaker leakage

4. MISSING DATA DETECTION:
   - Are there corrupted files?
   - Are there missing actors or emotions?
   - Solution: Early detection allows fixing before model training

5. VISUAL INSPECTION:
   - Plots are easier to understand than numbers
   - Identify outliers and anomalies quickly
   - Communicate findings to team/stakeholders

================================================================================
RAVDESS DATASET CHARACTERISTICS
================================================================================

EMOTION DISTRIBUTION (expected):
  - 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
  - Each emotion × 2 intensities (normal, strong)
  - Each × 2 statements (kids_talking, dogs_sitting)
  - Each × 2 repetitions (1st, 2nd)
  - Each × ~24 actors (some missing)
  
  Expected counts:
    - Speech files: 24 actors × 8 emotions × 2 intensities × 2 reps = 768 per category
    - Song files: 23 actors × 8 emotions × 2 intensities × 2 reps = 736 per category
    - Total: ~1,500 speech + ~1,000 song files

POTENTIAL ISSUES:
  - Actor_18 might be missing from some recordings
  - Some files might have audio corruption
  - Duration variations if files are trimmed differently

================================================================================
EDA OUTPUTS GENERATED
================================================================================

This module creates:
  1. metadata.csv - Full file listing (for reference)
  2. emotion_distribution.csv - Count of files per emotion
  3. emotion_distribution.png - Bar chart visualization
  4. duration_stats_by_emotion.csv - Mean/std/min/max duration per emotion
  5. duration_boxplot.png - Visual duration distribution
  6. actor_distribution.csv - Count of files per actor

These outputs help us:
  - Verify data integrity
  - Identify class imbalance
  - Make informed decisions about preprocessing
  - Document dataset characteristics

================================================================================
PROCESSING WORKFLOW FOR AUDIO & VIDEO
================================================================================

For AUDIO (.wav files):
  1. Parse filename → Extract emotion, actor, intensity, etc.
  2. Load audio file → Get duration_sec from librosa/soundfile
  3. Build metadata DataFrame with one row per file
  4. Analyze emotion distribution (count plot)
  5. Analyze duration by emotion (box plot)
  
  Example: "03-01-03-02-01-01-01.wav"
    → Modality=audio, Vocal=speech, Emotion=happy, Intensity=strong
    → Duration=2.87 seconds
    → Actor=01 (female), Gender=female

For VIDEO (.mp4 files):
  1. Parse filename → Same as audio
  2. Skip duration reading (video I/O slower, not needed for EDA)
  3. Just count files per emotion, per actor
  4. Can generate same distribution plots
  
  Example: "02-01-03-02-01-01-01.mp4"
    → Modality=video, Vocal=speech, Emotion=happy, Intensity=strong
    → Actor=01 (female), Gender=female

COMBINED MULTIMODAL EDA:
  - Total files: 2,452 audio + 4,904 video = 7,356 files
  - Emotion classes: All 8 emotions represented in both modalities
  - Actor coverage: Most actors in both audio and video
  - Opportunity: Use audio + video together for better recognition

================================================================================
STATISTICS WE CALCULATE
================================================================================

1. EMOTION DISTRIBUTION:
   Count of files for each emotion
   Visualization: Bar chart (x=emotion, y=count)
   Question answered: Are emotions balanced?
   
   Example output:
     neutral: 564 files (7.7%)
     calm: 1128 files (15.4%)
     happy: 1128 files (15.4%)
     sad: 1128 files (15.4%)
     angry: 1128 files (15.4%)
     fearful: 1128 files (15.4%)
     disgust: 1128 files (15.4%)
     surprised: 1128 files (15.4%)

2. DURATION STATISTICS (audio only):
   Mean, std, min, max duration per emotion
   Visualization: Box plot (x=emotion, y=duration_sec)
   Question answered: Is duration consistent across emotions?
   
   Example output:
     neutral: mean=2.87s, std=0.34s, min=1.94s, max=3.19s
     calm: mean=2.92s, std=0.41s, min=1.82s, max=3.21s
     ...

3. ACTOR DISTRIBUTION:
   Count of files per actor (1-24)
   Question answered: Is actor coverage balanced?
   
   Example output:
     Actor_01 (female): 128 files
     Actor_02 (male): 128 files
     Actor_18 (male): 96 files (missing some speech files)
     ...

================================================================================
MULTIMODAL ANALYSIS IDEAS
================================================================================

After audio EDA, we can extend to video:
  1. Frame count analysis:
     - How many frames in each video?
     - Are they consistent?
  
  2. Face detection:
     - Can we detect faces in all frames?
     - Which emotions have clearer facial expressions?
  
  3. Optical flow analysis:
     - Which emotions have most motion?
     - Angry: high motion, Sad: low motion?
  
  4. Audio-video sync:
     - Are audio and video well-synchronized?
     - Do they match in duration?

================================================================================
MODULE FUNCTIONS
================================================================================

This module provides:
  - run_basic_eda(): Generate all statistics and visualizations
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_basic_eda(metadata: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Generate exploratory data analysis for RAVDESS dataset.
    
    WHAT THIS FUNCTION DOES:
      1. Saves full metadata to CSV (for reference)
      2. Analyzes emotion class distribution
      3. Visualizes emotion distribution as bar chart
      4. Analyzes audio duration statistics
      5. Visualizes duration variations as box plot
      6. Analyzes actor distribution
    
    PARAMETERS:
      metadata: DataFrame from build_metadata_dataframe()
                Columns: path, emotion, emotion_code, actor, gender, duration_sec, etc.
                Rows: One per file (2,452 for audio only, 7,356 for audio+video)
      output_dir: Directory to save CSV and PNG outputs
    
    OUTPUTS CREATED:
      - metadata.csv: Full file listing (reference)
      - emotion_distribution.csv: Count per emotion
      - emotion_distribution.png: Bar chart
      - duration_stats_by_emotion.csv: Mean/std/min/max per emotion
      - duration_boxplot.png: Box plot of durations
      - actor_distribution.csv: Count per actor
    
    EXAMPLE ANALYSIS:
      metadata rows:
        path: archive/Audio_Speech.../03-01-01-01-01-01-01.wav
        emotion: neutral
        emotion_code: 01
        actor: 1
        gender: female
        duration_sec: 2.87
        ...
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # STEP 1: Save complete metadata for reference
    # This preserves all file information in case we need it later
    metadata.to_csv(output_path / "metadata.csv", index=False)

    # STEP 2: Analyze emotion distribution
    # Group by emotion, count files per emotion, sort by emotion code
    summary = (
        metadata.groupby(["emotion_code", "emotion"], as_index=False)
        .size()  # Count files in each group
        .rename(columns={"size": "count"})  # Rename "size" to "count"
        .sort_values("emotion_code")  # Sort by emotion code (01-08)
    )
    # Save to CSV for reports
    summary.to_csv(output_path / "emotion_distribution.csv", index=False)
    
    # STEP 3: Create bar chart of emotion distribution
    # Why: Visual inspection is easier than reading numbers
    plt.figure(figsize=(10, 5))
    
    # Create bar plot: x=emotion name, y=count (file frequency)
    # color="#4C72B0" = nice blue color for reports
    sns.countplot(
        data=metadata,
        x="emotion",
        order=summary["emotion"].tolist(),  # Order by emotion codes
        color="#4C72B0"
    )
    
    plt.title("RAVDESS Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    # Rotate emotion names 30 degrees for readability
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path / "emotion_distribution.png", dpi=180)
    plt.close()
    
    # STEP 4: Analyze audio duration statistics
    # Check if duration_sec column exists and has valid data
    # (duration_sec is only available for .wav files, not .mp4)
    if "duration_sec" in metadata and metadata["duration_sec"].notna().any():
        # Group by emotion, calculate duration statistics
        duration_stats = (
            metadata.groupby("emotion", as_index=False)["duration_sec"]
            .agg(["mean", "std", "min", "max", "count"])  # Mean, std dev, min, max, count
            .reset_index()
        )
        # Save to CSV
        duration_stats.to_csv(output_path / "duration_stats_by_emotion.csv", index=False)

        # STEP 5: Create box plot of duration by emotion
        # Box plot shows: median, quartiles, outliers for each emotion
        # Why: Identify if any emotion has unusual duration (corrupted files?)
        plt.figure(figsize=(11, 5))
        
        sns.boxplot(
            data=metadata,
            x="emotion",
            y="duration_sec",  # Audio duration in seconds
            color="#55A868"  # Nice green color for reports
        )
        
        plt.title("Audio Duration by Emotion")
        plt.xlabel("Emotion")
        plt.ylabel("Duration (seconds)")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(output_path / "duration_boxplot.png", dpi=180)
        plt.close()

    # STEP 6: Analyze actor distribution
    # Count how many files each actor (1-24) has
    # Why: Ensure balanced actor representation
    actor_dist = (
        metadata.groupby("actor", as_index=False)
        .size()  # Count files per actor
        .rename(columns={"size": "count"})
    )
    # Save to CSV
    actor_dist.to_csv(output_path / "actor_distribution.csv", index=False)
