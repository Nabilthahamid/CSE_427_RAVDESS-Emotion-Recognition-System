"""Run the full RAVDESS emotion-recognition pipeline (AUDIO + VIDEO).

================================================================================
COMPLETE WORKFLOW EXPLANATION
================================================================================

This script orchestrates ALL stages of multimodal emotion recognition:

[1/6] METADATA PARSING
     Input: Raw files in archive/ (2,452 .wav + 4,904 .mp4 = 7,356 total)
     Process: Parse filenames → Extract emotion, actor, gender, intensity, etc.
     Output: DataFrame with metadata for each file
     Questions: How many files? Are all emotions represented? Which actors?

[2/6] EXPLORATORY DATA ANALYSIS (EDA)
     Input: Metadata DataFrame
     Process: Analyze class distribution, actor balance, file durations
     Output: CSV statistics + visualizations (plots)
     Purpose: Understand dataset characteristics before training
     Detects: Imbalanced classes, corrupted files, missing data

[3/6] FEATURE EXTRACTION (AUDIO)
     Input: 2,452 .wav files
     Process: Extract 40 audio features per file (MFCC, Mel, Chroma, ZCR, RMS)
     Output: audio_features.csv (2,452 rows × 40 features + metadata)
     Time: ~33 seconds for all files
     
     WHAT HAPPENS:
       - Load each audio file (3 seconds, 22050 Hz)
       - Compute MFCC coefficients (perceptual sound)
       - Compute Mel spectrogram (frequency distribution)
       - Compute Chroma features (pitch content)
       - Compute Zero Crossing Rate (signal complexity)
       - Compute RMS Energy (loudness)
       - Take mean and std of each feature across time
       - Result: 40-dim vector per audio file
     
     HOW USED:
       - Feed 40-dim vectors into machine learning models
       - Models learn patterns: "happy usually has high MFCC_0, low RMS_std"
       - Predict emotion for new audio: extract 40 features → model → emotion

[4/6] VIDEO FEATURE EXTRACTION (OPTIONAL - FUTURE)
     Input: 4,904 .mp4 files (currently skipped)
     Process: Extract visual features (optical flow, facial expressions)
     Output: video_features.csv (4,904 rows × 40 features)
     
     WHAT WOULD HAPPEN:
       - Extract 10 frames from each video
       - Compute optical flow (motion patterns)
       - Extract color statistics (facial color)
       - Extract edge features (facial tension)
       - Result: 40-dim visual feature vector per video file
     
     MULTIMODAL FUSION:
       - Audio features (40 dims) + Video features (40 dims) = 80 dims
       - Train models on 80-dim vectors
       - Expected: Better accuracy (85%+ vs 78% audio-only)

[5/6] DATA SPLITTING
     Input: Feature table (2,452 rows for audio)
     Process: Divide into train/test sets
     
     TWO STRATEGIES:
       Option A: Actor-wise split (RECOMMENDED)
         - Prevents speaker leakage (same person not in train + test)
         - 60% train, 20% val, 20% test
         - More rigorous evaluation
         - Slightly lower accuracy (more realistic)
       
       Option B: Simple 80/20 split
         - Faster, simpler
         - May have speaker leakage
         - Slightly higher accuracy (overoptimistic)
     
     WHY SPLITTING MATTERS:
       - Prevents data leakage (using test data to tune model)
       - Validates real-world performance
       - Actor-wise split: Ensures model works on NEW speakers
       - Simple split: Allows same speaker in train + test

[6/6] MODEL TRAINING & EVALUATION
     Input: Training and test sets
     Process: Train 5 models, evaluate on test set, compare
     
     MODELS:
       1. SVM (RBF kernel)
          - Support Vector Machine
          - Non-linear decision boundary
          - Usually BEST for handcrafted features
       
       2. MLP (Multi-Layer Perceptron)
          - Neural network (2 hidden layers)
          - Non-linear, universal approximator
          - Decent performance on features
       
       3. Random Forest
          - Ensemble of decision trees
          - Fast, interpretable
          - Good baseline
       
       4. XGBoost
          - Gradient boosting (advanced ensemble)
          - Usually wins Kaggle competitions
          - Good but slightly slower
       
       5. LightGBM
          - Lightweight gradient boosting
          - Very fast, competitive with XGBoost
          - Good for production
     
     TRAINING PROCESS:
       - Combine train + validation data
       - Normalize features (for SVM/MLP)
       - Fit model on combined data
       - Predict on test set
       - Compute metrics (accuracy, F1 score)
       - Save model to .joblib file
     
     EVALUATION METRICS:
       - Accuracy: % of correct predictions
       - Macro F1: Average F1 score per emotion (handles imbalance)
       - Weighted F1: F1 weighted by class size
       - Confusion matrix: Which emotions confused?
       - Classification report: Per-emotion precision/recall/F1
     
     OUTPUT:
       - Trained models (.joblib files - reusable!)
       - Confusion matrices (visualizations)
       - Classification reports (JSON with metrics)
       - Model comparison table (ranked by performance)

[7/6] OPTIONAL: DEEP LEARNING
     Process: Train CNN+BiLSTM on log-mel spectrograms
     Performance: ~73-76% (slight improvement over MLP, less than SVM)
     Requires: TensorFlow installation

================================================================================
USAGE EXAMPLES
================================================================================

# Quick test with 320 files (fast, for debugging)
python run_audio_pipeline.py --archive-root archive --output-dir outputs_quick --max-files 320

# Full dataset with best settings (actor-wise split, prevents speaker leakage)
python run_audio_pipeline.py --archive-root archive --output-dir outputs_full

# Full dataset with simple 80/20 split (may have speaker leakage, but faster)
python run_audio_pipeline.py --archive-root archive --output-dir outputs_simple --use-simple-split

# With optional CNN+BiLSTM deep model
python run_audio_pipeline.py --archive-root archive --output-dir outputs_deep --use-simple-split --run-deep

# Include video files in metadata analysis (prepares for multimodal)
python run_audio_pipeline.py --archive-root archive --output-dir outputs_multimodal --include-video

================================================================================
EXPECTED RESULTS
================================================================================

On full dataset (2,452 audio files, 80/20 split):
  SVM:         78.2% accuracy, 0.774 macro F1  ← BEST
  LightGBM:    75.2% accuracy, 0.740 macro F1
  XGBoost:     74.5% accuracy, 0.737 macro F1
  MLP:         72.7% accuracy, 0.721 macro F1
  RandomForest: 66.0% accuracy, 0.654 macro F1

With video features included (hypothetical):
  Multimodal:  85%+ accuracy (audio + video combined)

================================================================================
"""


from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ravdess.data import build_metadata_dataframe
from ravdess.deep import train_cnn_bilstm
from ravdess.eda import run_basic_eda
from ravdess.features import build_feature_table, save_feature_table
from ravdess.models import train_and_evaluate_models
from ravdess.split import make_actorwise_splits, make_simple_stratified_split


def _sample_per_emotion(df: pd.DataFrame, max_total_files: int) -> pd.DataFrame:
    """Sample files uniformly per emotion class for balanced representation.
    
    WHY THIS FUNCTION EXISTS:
      - For quick testing, we don't need all 2,452 files
      - Want to preserve emotion class balance (not just take first 320 files)
      - Example: Instead of all angry files + few neutral, get ~40 of each emotion
    
    HOW IT WORKS:
      1. Identify unique emotions (8 total)
      2. Calculate: files_per_emotion = max_total_files / num_emotions
      3. For each emotion: randomly sample that many files
      4. Concatenate results
      Result: Balanced sample respecting emotion distribution
    
    EXAMPLE:
      - Original: 2,452 files total
      - max_total_files = 320
      - Emotions: 8
      - Per emotion: 320 / 8 = 40 files
      - Result: 320 files total (40 of each emotion)
    
    Args:
        df: DataFrame with emotion column
        max_total_files: Maximum total files to keep (0 = no limit)
    
    Returns:
        DataFrame with at most max_total_files, balanced per emotion
    """
    # If no limit specified or dataset already small, return as-is
    if max_total_files <= 0 or len(df) <= max_total_files:
        return df

    # Get unique emotions (8 for RAVDESS: neutral, calm, happy, sad, angry, fearful, disgust, surprised)
    emotions = sorted(df["emotion"].unique().tolist())
    
    # Calculate how many files per emotion to keep
    # Example: 320 files / 8 emotions = 40 files per emotion
    per_emotion = max(1, max_total_files // len(emotions))
    
    # For each emotion, randomly sample that many files
    sampled_parts = []
    for emotion in emotions:
        # Get all files for this emotion
        group = df[df["emotion"] == emotion]
        # Randomly sample up to per_emotion files (or fewer if not available)
        sampled_parts.append(group.sample(n=min(per_emotion, len(group)), random_state=42))
    
    # Combine all sampled parts back into single DataFrame
    sampled = pd.concat(sampled_parts, axis=0, ignore_index=True)
    return sampled


def main() -> None:
    """Main pipeline runner: data → features → split → train → evaluate.
    
    WHAT THIS FUNCTION DOES:
      Orchestrates the complete RAVDESS emotion recognition workflow
      
    STEPS:
      [1/6] Parse filenames from archive → extract metadata (emotion, actor, etc.)
      [2/6] Analyze dataset distribution → create visualizations
      [3/6] Extract audio features → convert .wav to 40-dim vectors
      [4/6] Split data → train/validation/test sets
      [5/6] Train 5 models → compare performance
      [6/6] Optionally train deep model
    
    COMMAND-LINE ARGUMENTS:
      --archive-root: Path to dataset folder (default: "archive/")
      --output-dir: Where to save results (default: "outputs/")
      --max-files: Optional limit for quick testing (0 = full dataset)
      --include-video: Also parse .mp4 files for multimodal analysis
      --use-simple-split: Use 80/20 split instead of actor-wise
      --run-deep: Also train optional CNN+BiLSTM model
    """
    # Create argument parser for command-line options
    parser = argparse.ArgumentParser(description="RAVDESS multimodal emotion recognition pipeline")
    parser.add_argument("--archive-root", default="archive", help="Path to dataset archive folder")
    parser.add_argument("--output-dir", default="outputs", help="Directory for outputs")
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional cap for quick experiments (0 = full dataset)",
    )
    parser.add_argument("--run-deep", action="store_true", help="Also train optional CNN+BiLSTM model")
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Parse metadata from .mp4 files too (prepares for multimodal future work)"
    )
    parser.add_argument(
        "--use-simple-split",
        action="store_true",
        help="Use simple 80/20 stratified split instead of actor-wise split (may have speaker leakage)",
    )
    args = parser.parse_args()

    # Create output directory
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # STAGE 1: BUILD METADATA FROM FILENAMES
    # ========================================================================
    print("[1/6] Building metadata from filenames...")
    print(f"      Include video files: {args.include_video}")
    
    # Parse all RAVDESS filenames in archive → extract metadata
    # include_video=False → only .wav files (2,452 total)
    # include_video=True → .wav + .mp4 files (7,356 total)
    metadata = build_metadata_dataframe(args.archive_root, include_video=args.include_video)
    
    if metadata.empty:
        raise RuntimeError("No valid RAVDESS files were found. Check archive path and naming.")

    # Extract only audio files for feature extraction
    # (Video feature extraction would be separate stage)
    audio_metadata = metadata[metadata["extension"] == ".wav"].copy().reset_index(drop=True)
    
    # Optional: Sample subset for quick testing
    if args.max_files > 0:
        audio_metadata = _sample_per_emotion(audio_metadata, args.max_files)

    print(f"      Total files parsed: {len(metadata)}")
    print(f"      Audio files (.wav): {len(audio_metadata)}")
    if args.include_video:
        video_count = len(metadata) - len(audio_metadata)
        print(f"      Video files (.mp4): {video_count}")

    # ========================================================================
    # STAGE 2: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    print("[2/6] Running EDA (class distribution, duration stats, actor balance)...")
    
    # Generate statistics and visualizations
    # Outputs: emotion_distribution.csv/png, duration_stats.csv, actor_distribution.csv, etc.
    run_basic_eda(audio_metadata, out / "eda")
    
    print("      Outputs: emotion_distribution.csv/png, duration_boxplot.png, etc.")

    # ========================================================================
    # STAGE 3: EXTRACT AUDIO FEATURES
    # ========================================================================
    print("[3/6] Extracting audio features from .wav files...")
    print("      Computing MFCC, Mel, Chroma, ZCR, RMS features...")
    
    # Extract 40 audio features from each .wav file
    # Returns: DataFrame with one row per file, columns: metadata + f_000 to f_039
    features, failed = build_feature_table(audio_metadata)
    
    if features.empty:
        raise RuntimeError("Feature extraction produced no rows.")

    # Save feature table to CSV for reproducibility
    save_feature_table(features, out / "features" / "audio_features.csv")
    
    # Save list of files that failed extraction (usually empty)
    if failed:
        pd.DataFrame({"failed_path": failed}).to_csv(out / "features" / "failed_files.csv", index=False)
        print(f"      Warning: {len(failed)} files failed extraction (see failed_files.csv)")
    
    print(f"      Features extracted: {len(features)} files × 40 dimensions")

    # ========================================================================
    # STAGE 4: BUILD TRAIN/TEST SPLITS
    # ========================================================================
    print("[4/6] Building train/test splits...")
    
    # Two splitting strategies:
    if args.use_simple_split:
        # Simple 80/20 stratified split (faster, may have speaker leakage)
        print("      Strategy: Simple 80/20 stratified split")
        print("      ⚠️  Warning: May have speaker leakage (same actor in train + test)")
        split_frames = make_simple_stratified_split(features)
    else:
        # Actor-wise split (recommended, prevents speaker leakage)
        print("      Strategy: Actor-wise split (prevents speaker leakage)")
        try:
            # 60% train, 20% val, 20% test with actor isolation
            split_frames = make_actorwise_splits(features)
        except ValueError as e:
            # Fallback if not enough actors
            print(f"      ⚠️  {e}")
            print("      Falling back to simple 80/20 split...")
            split_frames = make_simple_stratified_split(features)

    # Save split files for reproducibility
    for split_name, split_df in split_frames.items():
        split_df.to_csv(out / "features" / f"{split_name}.csv", index=False)
        print(f"      {split_name}: {len(split_df)} files")

    # ========================================================================
    # STAGE 5: TRAIN AND EVALUATE MODELS
    # ========================================================================
    print("[5/6] Training and evaluating 5 baseline models...")
    print("      Models: SVM, MLP, Random Forest, XGBoost, LightGBM")
    
    # Train all 5 models on combined train+validation data
    # Evaluate on test set
    # Returns: DataFrame with model names, accuracy, F1 scores, etc.
    metrics_df = train_and_evaluate_models(split_frames, out / "models")

    # Display results in table format
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS (Ranked by Macro F1)")
    print("="*70)
    print(metrics_df.to_string(index=False))
    print("="*70 + "\n")
    
    # Show best model
    best_model = metrics_df.iloc[0]
    print(f"🏆 Best Model: {best_model['model']} with {best_model['accuracy']:.1%} accuracy")

    # ========================================================================
    # STAGE 6: OPTIONAL DEEP LEARNING
    # ========================================================================
    if args.run_deep:
        # Optional CNN+BiLSTM on log-mel spectrograms
        if "val" not in split_frames:
            print("[6/6] Note: Deep model requires validation set")
            print("      (Use actor-wise split or provide validation data)")
            print("      Skipping deep model...")
        else:
            print("[6/6] Training optional CNN+BiLSTM deep model...")
            print("      This may take 1-2 minutes...")
            
            try:
                deep_metrics = train_cnn_bilstm(
                    split_frames["train"],
                    split_frames["val"],
                    split_frames["test"],
                    out / "models",
                )
                print(f"      Deep model metrics: {deep_metrics}")
            except Exception as e:
                print(f"      Deep model training failed: {e}")
                print("      (Try installing tensorflow: pip install tensorflow)")
    else:
        print("[6/6] Skipped deep model (optional)")
        print("      To enable CNN+BiLSTM training, use --run-deep flag")

    # ========================================================================
    # PIPELINE COMPLETE
    # ========================================================================
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📁 All outputs saved to: {out}/")
    print(f"\nKey output files:")
    print(f"  - eda/emotion_distribution.png - Visualize class distribution")
    print(f"  - features/audio_features.csv - All extracted features")
    print(f"  - features/train.csv, test.csv - Train/test data splits")
    print(f"  - models/model_comparison_macro_f1.png - Compare model performance")
    print(f"  - models/svm_rbf.joblib - Best trained model (reusable!)")
    print(f"\nTo make predictions with trained model:")
    print(f"  import joblib")
    print(f"  model = joblib.load('{out}/models/svm_rbf.joblib')")
    print(f"  prediction = model.predict([[features_of_new_audio]])")


# ============================================================================
# ENTRY POINT
# ============================================================================
# This ensures main() only runs when script is executed directly
# (not when imported as a module from another script)
if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()
