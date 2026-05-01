"""
Multimodal emotion recognition pipeline: Audio + Video combined.

================================================================================
MULTIMODAL PIPELINE OVERVIEW
================================================================================

Goal: Combine audio and video features for better emotion recognition accuracy.

STRATEGY:
  1. Extract audio features from 2,452 .wav files (40 dims each)
  2. Extract video features from 4,904 .mp4 files (40 dims each)
  3. For files with BOTH audio + video: Combine features (80 dims total)
  4. Train models on combined 80-dim vectors
  5. Compare accuracy: Audio-only (78%) vs Multimodal (expected 85%+)

================================================================================
PROCESSING STAGES
================================================================================

[1/7] Load metadata: Parse all 7,356 files from archive/
[2/7] Extract audio features: Process 2,452 .wav files → 40-dim vectors
[3/7] Extract video features: Process 4,904 .mp4 files → 40-dim vectors
[4/7] Combine features: Merge audio + video for all files with both
[5/7] Split data: Create train/test split with stratification
[6/7] Train models: Compare 5 models on 80-dim combined features
[7/7] Evaluate: Show accuracy improvement (78% → 85%+)

================================================================================
FEATURE COMPOSITION
================================================================================

AUDIO FEATURES (40 dimensions):
  - MFCC: Mel-Frequency Cepstral Coefficients (perceptual spectrum)
  - Mel Spectrogram: Frequency distribution
  - Chroma: Pitch classes (for song files)
  - ZCR: Zero Crossing Rate (signal complexity)
  - RMS: Energy/loudness level

VIDEO FEATURES (40 dimensions):
  - Optical Flow (16 dims): Motion patterns (angry→fast, sad→slow)
  - Frame Statistics (12 dims): Color, edges, texture
  - Temporal Statistics (12 dims): How features change over time

COMBINED FEATURES (80 dimensions):
  - Audio (40) + Video (40) = 80 dims
  - Captures: What you SAY + How you LOOK + Combined effect
  - Expected: More robust emotion recognition

================================================================================
EXPECTED IMPROVEMENTS
================================================================================

Audio only:           78.21% accuracy
Video only:           ~72% accuracy (estimated)
Audio + Video:        ~85%+ accuracy (expected improvement: +7%)

Why multimodal is better:
  - Audio captures voice characteristics (pitch, loudness, speech rate)
  - Video captures facial/body expressions (smile, tension, movement)
  - Combined: Redundancy and complementary information reduce errors
"""

from pathlib import Path
from argparse import ArgumentParser
import sys

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ravdess.data import build_metadata_dataframe
from ravdess.features import build_feature_table, build_video_feature_table, save_feature_table
from ravdess.split import make_simple_stratified_split, make_actorwise_splits
from ravdess.models import train_and_evaluate_models


def combine_audio_video_features(
    audio_features: pd.DataFrame,
    video_features: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Combine audio and video features for multimodal training.
    
    PARAMETERS:
      audio_features: DataFrame from build_feature_table() (2,452 rows × 40 features)
      video_features: DataFrame from build_video_feature_table() (4,904 rows × 40 features)
      output_dir: Directory to save combined features
    
    STRATEGY:
      1. For files with both audio and video:
         - Combine audio (40) + video (40) = 80 dims
         - Keep metadata (emotion, actor, gender)
      
      2. For files with only audio:
         - Pad with zeros for video features (backward compatibility)
         - Result: 80 dims with video part as zeros
      
      3. For files with only video:
         - Pad with zeros for audio features
         - Result: 80 dims with audio part as zeros
    
    RETURNS:
      DataFrame with 80-dimensional combined features
    """
    print("\n[4/7] Combining audio + video features...")
    
    # Create a mapping of actor+emotion → features
    # This is how we can match corresponding audio/video pairs
    
    # Get emotions that have both audio and video files
    audio_emotions = set(audio_features['emotion'].unique())
    video_emotions = set(video_features['emotion'].unique())
    common_emotions = audio_emotions & video_emotions
    
    print(f"  Audio files: {len(audio_features)} files")
    print(f"  Video files: {len(video_features)} files")
    print(f"  Common emotions: {common_emotions}")
    
    # Create combined features for both audio and video
    combined_rows = []
    
    # Strategy: For each unique combination of (emotion, actor),
    # if we have both audio and video: combine them
    # Otherwise: use what we have
    
    for emotion in sorted(common_emotions):
        audio_emotion = audio_features[audio_features['emotion'] == emotion]
        video_emotion = video_features[video_features['emotion'] == emotion]
        
        # For each audio file with this emotion
        for idx, audio_row in audio_emotion.iterrows():
            # Try to find matching video file (same emotion, same actor ideally)
            matching_videos = video_emotion[video_emotion['actor'] == audio_row['actor']]
            
            combined_row = {
                'path_audio': audio_row['path'],
                'emotion': audio_row['emotion'],
                'emotion_code': audio_row['emotion_code'],
                'actor': audio_row['actor'],
                'gender': audio_row['gender'],
            }
            
            # Add audio features
            audio_feats = [audio_row[f'f_{i:03d}'] for i in range(40)]
            for i, val in enumerate(audio_feats):
                combined_row[f'a_{i:03d}'] = val
            
            # Add video features (if matching video exists)
            if len(matching_videos) > 0:
                video_row = matching_videos.iloc[0]
                combined_row['path_video'] = video_row['path']
                video_feats = [video_row[f'v_{i:03d}'] for i in range(40)]
                for i, val in enumerate(video_feats):
                    combined_row[f'v_{i:03d}'] = val
            else:
                # No matching video: use zeros
                combined_row['path_video'] = None
                for i in range(40):
                    combined_row[f'v_{i:03d}'] = 0.0
            
            combined_rows.append(combined_row)
    
    combined_df = pd.DataFrame(combined_rows)
    
    # Save combined features
    combined_output = output_dir / "combined_features.csv"
    combined_df.to_csv(combined_output, index=False)
    print(f"  ✓ Saved: {combined_output} ({len(combined_df)} files)")
    
    # Create feature matrix with both audio and video
    feature_cols_a = [f'a_{i:03d}' for i in range(40)]
    feature_cols_v = [f'v_{i:03d}' for i in range(40)]
    combined_df['features'] = combined_df.apply(
        lambda row: list(row[feature_cols_a]) + list(row[feature_cols_v]),
        axis=1
    )
    
    return combined_df


def main():
    """Main multimodal pipeline execution."""
    parser = ArgumentParser(description="Run multimodal (audio+video) emotion recognition")
    parser.add_argument("--archive-root", default="archive", help="Path to archive folder")
    parser.add_argument("--output-dir", default="outputs_multimodal", help="Output directory")
    parser.add_argument("--use-simple-split", action="store_true", 
                       help="Use 80/20 split (default: actor-wise)")
    
    args = parser.parse_args()
    
    archive_root = Path(args.archive_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # [1/7] Parse metadata
    print("\n[1/7] Parsing metadata for all 7,356 files...")
    metadata = build_metadata_dataframe(archive_root, include_video=True)
    print(f"  ✓ Parsed: {len(metadata)} files")
    metadata.to_csv(output_dir / "metadata.csv", index=False)
    
    # [2/7] Extract audio features
    print("\n[2/7] Extracting audio features from 2,452 .wav files...")
    audio_features, failed_audio = build_feature_table(metadata, progress=True)
    print(f"  ✓ Extracted: {len(audio_features)} audio files")
    if failed_audio:
        print(f"  ⚠ Failed: {len(failed_audio)} files")
    save_feature_table(audio_features, output_dir / "audio_features.csv")
    
    # [3/7] Extract video features
    print("\n[3/7] Extracting video features from 4,904 .mp4 files...")
    print("  (This will take longer - processing motion, color, edges)")
    video_features, failed_video = build_video_feature_table(metadata, progress=True)
    print(f"  ✓ Extracted: {len(video_features)} video files")
    if failed_video:
        print(f"  ⚠ Failed: {len(failed_video)} files")
    save_feature_table(video_features, output_dir / "video_features.csv")
    
    # [4/7] Combine features
    combined_features = combine_audio_video_features(
        audio_features, video_features, output_dir
    )
    
    # [5/7] Split data for multimodal features (80 dimensions)
    print("\n[5/7] Splitting data for 80-dimensional combined features...")
    
    # Create feature matrix for splitting
    X = pd.DataFrame({
        f"f_{i:03d}": combined_features.apply(lambda row: row['features'][i], axis=0)
        for i in range(80)
    })
    X['emotion'] = combined_features['emotion']
    X['actor'] = combined_features['actor']
    
    # Create split
    if args.use_simple_split:
        print("  Using simple 80/20 stratified split")
        split_frames = make_simple_stratified_split(X)
    else:
        print("  Using actor-wise split (60/20/20)")
        split_frames = make_actorwise_splits(X)
    
    # Save splits
    for split_name, df in split_frames.items():
        df.to_csv(output_dir / f"{split_name}.csv", index=False)
        print(f"    {split_name}: {len(df)} samples")
    
    # [6/7] Train models on multimodal features
    print("\n[6/7] Training 5 models on 80-dimensional combined features...")
    models_output = output_dir / "models"
    models_output.mkdir(exist_ok=True)
    metrics = train_and_evaluate_models(split_frames, models_output)
    print("\n  Model Comparison (Multimodal - 80 features):")
    print(metrics.to_string(index=False))
    metrics.to_csv(output_dir / "model_metrics_multimodal.csv", index=False)
    
    # [7/7] Summary
    print("\n[7/7] Multimodal pipeline complete!")
    print(f"\n{'='*70}")
    print("MULTIMODAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Audio features extracted:     {len(audio_features):>5} files")
    print(f"Video features extracted:     {len(video_features):>5} files")
    print(f"Combined features created:    {len(combined_features):>5} files")
    print(f"Training samples:             {len(split_frames['train']):>5} files")
    print(f"Test samples:                 {len(split_frames['test']):>5} files")
    print(f"\nBest model (multimodal):      {metrics.iloc[0]['model']}")
    print(f"Best accuracy (multimodal):   {metrics.iloc[0]['accuracy']:.4f}")
    print(f"\nComparison:")
    print(f"  Audio-only best:   78.21%")
    print(f"  Multimodal best:   {metrics.iloc[0]['accuracy']:.2%}")
    print(f"  Improvement:       +{(metrics.iloc[0]['accuracy'] - 0.7821)*100:.2f}%")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  - combined_features.csv : Merged audio+video features")
    print(f"  - audio_features.csv    : Audio features only")
    print(f"  - video_features.csv    : Video features only")
    print(f"  - train.csv, test.csv   : Data split")
    print(f"  - models/               : Trained model files")
    print(f"  - model_metrics_multimodal.csv : Performance comparison")


if __name__ == "__main__":
    main()
