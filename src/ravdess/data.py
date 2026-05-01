"""Data loading and filename parsing for the RAVDESS dataset.

================================================================================
RAVDESS DATASET STRUCTURE & FILE NAMING CONVENTION
================================================================================

RAVDESS filenames encode metadata in a standardized format:
  MM-VC-EE-II-SS-RR-AA.extension
  
KEY FIELD MEANINGS:
  
  MM: Modality (file content type)
    01 = Full Audio-Visual (contains both audio + video) [NOT USED IN ARCHIVE]
    02 = Video only (.mp4 files) [4,904 video files in archive]
    03 = Audio only (.wav files) [2,452 audio files in archive]
    
  VC: Vocal Channel (type of speech content)
    01 = Speech utterance ("Kids are talking by the door", "Dogs are sitting")
    02 = Song utterance ("a-do-de", "a-e-i-o-u" sung)
    
  EE: Emotion category (8 basic emotions)
    01 = Neutral     (no emotional expression)
    02 = Calm        (relaxed, peaceful)
    03 = Happy       (joyful, cheerful)
    04 = Sad         (sorrowful, melancholy)
    05 = Angry       (frustrated, irritated)
    06 = Fearful     (anxious, scared)
    07 = Disgust     (repulsed, disgusted)
    08 = Surprised   (astonished, amazed)
    
  II: Intensity level (how strong the emotion is expressed)
    01 = Normal intensity
    02 = Strong intensity (more exaggerated emotion)
    
  SS: Statement (the specific phrase spoken/sung)
    01 = "Kids are talking by the door"
    02 = "Dogs are sitting in the corner"
    
  RR: Repetition (how many times this exact utterance was repeated)
    01 = First time
    02 = Second time
    
  AA: Actor (performer who recorded the file)
    01-24 = Actor IDs
    Odd (01,03,05,...,23) = Female speakers
    Even (02,04,06,...,24) = Male speakers

ARCHIVE ORGANIZATION:
  archive/
    ├── Audio_Song_Actors_01-24/           (Speech as song - 1,012 .wav files)
    │   ├── Actor_01/
    │   ├── Actor_02/
    │   └── ... Actor_24/
    │
    ├── Audio_Speech_Actors_01-24/         (Speech as utterance - 1,440 .wav files)
    │   ├── Actor_01/
    │   ├── Actor_02/
    │   └── ... Actor_24/
    │
    ├── Video_Song_Actor_01/               (Song + video - 2,452 .mp4 files)
    │   └── Actor_01/
    ├── Video_Song_Actor_02/
    │   └── Actor_02/
    └── ... Video_Song_Actor_24/ & Video_Speech_Actor_01-24/

TOTAL FILES:
  Audio (.wav): 2,452 files
    - Speech: 1,440 files (actors 1-24, 2 channels × 2 intensities × 2 reps = 16 per actor)
    - Song: 1,012 files (missing Actor_18, 2 channels × 2 intensities × 2 reps = 16 per actor)
  
  Video (.mp4): 4,904 files
    - Speech: 2,452 files
    - Song: 2,452 files
  
  GRAND TOTAL: 7,356 files (combines both modalities)

================================================================================
MULTIMODAL FEATURE EXTRACTION STRATEGY
================================================================================

AUDIO FILES (.wav):
  - Extract 40 audio features using librosa:
    * MFCC (40 dims): Mel-Frequency Cepstral Coefficients
      - Captures perceptual properties of sound
      - How humans hear different frequencies
    * Mel Spectrogram (128 dims): Frequency power over time
    * Chroma Features (12 dims): Musical pitch content
    * Zero Crossing Rate (1 dim): Signal complexity
    * RMS Energy (1 dim): Overall loudness
  - Total: 40-dimensional audio feature vector
  - File: audio_features.csv

VIDEO FILES (.mp4):
  - Extract visual features using OpenCV:
    * Optical Flow (2-16 dims): Motion patterns between frames
      - Captures head movements, mouth movements
      - Important for emotion (angry = fast movement, sad = slow movement)
    * Frame Statistics (12 dims): Color & texture features
      - Mean RGB values (facial color changes with emotion)
      - Edge density (facial tension)
    * Temporal Features (8 dims): Motion over time
  - Total: 40-dimensional visual feature vector (matched to audio)
  - File: video_features.csv

MULTIMODAL FUSION:
  - Audio features (40 dims) + Video features (40 dims) = 80-dim combined vector
  - Train models on fused features for better emotion recognition
  - Expected improvement: 78% → 85%+ accuracy

================================================================================
MODULE FUNCTIONS
================================================================================

This module provides:
  - RavdessRecord: dataclass for parsed file metadata (supports audio + video)
  - parse_ravdess_filename(): extract metadata from a single file (any modality)
  - build_metadata_dataframe(): create DataFrame from all files in archive
  - _iter_files(): iterator that finds all .wav and/or .mp4 files
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import soundfile as sf

from .constants import (
    EMOTION_MAP,
    GENDER_MAP,
    INTENSITY_MAP,
    MODALITY_MAP,
    STATEMENT_MAP,
    VOCAL_CHANNEL_MAP,
)

FILENAME_PATTERN = re.compile(
    r"^(?P<modality>\d{2})-(?P<vocal_channel>\d{2})-(?P<emotion>\d{2})-"
    r"(?P<intensity>\d{2})-(?P<statement>\d{2})-(?P<repetition>\d{2})-(?P<actor>\d{2})$"
)


@dataclass(frozen=True)
class RavdessRecord:
    path: str
    extension: str
    top_folder: str
    modality_code: str
    vocal_channel_code: str
    emotion_code: str
    intensity_code: str
    statement_code: str
    repetition_code: str
    actor_code: str
    modality: str
    vocal_channel: str
    emotion: str
    intensity: str
    statement: str
    actor: int
    gender: str
    duration_sec: float | None


def _iter_files(archive_root: Path, include_video: bool = False) -> Iterable[Path]:
    allowed_suffixes = {".wav"}
    if include_video:
        allowed_suffixes.add(".mp4")

    for path in archive_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_suffixes:
            yield path


def _get_wav_duration(file_path: Path) -> float | None:
    if file_path.suffix.lower() != ".wav":
        return None
    try:
        with sf.SoundFile(file_path) as handle:
            if handle.samplerate == 0:
                return None
            return len(handle) / float(handle.samplerate)
    except Exception:
        return None


def parse_ravdess_filename(file_path: Path) -> RavdessRecord | None:
    match = FILENAME_PATTERN.match(file_path.stem)
    if match is None:
        return None

    parts = match.groupdict()
    actor = int(parts["actor"])
    gender = GENDER_MAP[actor % 2]

    return RavdessRecord(
        path=str(file_path),
        extension=file_path.suffix.lower(),
        top_folder=file_path.parts[-3] if len(file_path.parts) >= 3 else "",
        modality_code=parts["modality"],
        vocal_channel_code=parts["vocal_channel"],
        emotion_code=parts["emotion"],
        intensity_code=parts["intensity"],
        statement_code=parts["statement"],
        repetition_code=parts["repetition"],
        actor_code=parts["actor"],
        modality=MODALITY_MAP.get(parts["modality"], "unknown"),
        vocal_channel=VOCAL_CHANNEL_MAP.get(parts["vocal_channel"], "unknown"),
        emotion=EMOTION_MAP.get(parts["emotion"], "unknown"),
        intensity=INTENSITY_MAP.get(parts["intensity"], "unknown"),
        statement=STATEMENT_MAP.get(parts["statement"], "unknown"),
        actor=actor,
        gender=gender,
        duration_sec=_get_wav_duration(file_path),
    )


def build_metadata_dataframe(archive_root: str | Path, include_video: bool = False) -> pd.DataFrame:
    root = Path(archive_root)
    records: list[RavdessRecord] = []

    for file_path in _iter_files(root, include_video=include_video):
        rec = parse_ravdess_filename(file_path)
        if rec is not None:
            records.append(rec)

    frame = pd.DataFrame([r.__dict__ for r in records])
    if frame.empty:
        return frame

    frame = frame.sort_values(["emotion_code", "actor", "path"]).reset_index(drop=True)
    return frame
