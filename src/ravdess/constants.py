"""Shared constants for RAVDESS parsing and labeling."""

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

MODALITY_MAP = {
    "01": "full_av",
    "02": "video_only",
    "03": "audio_only",
}

VOCAL_CHANNEL_MAP = {
    "01": "speech",
    "02": "song",
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong",
}

STATEMENT_MAP = {
    "01": "kids_talking_door",
    "02": "dogs_sitting_door",
}

GENDER_MAP = {
    0: "female",
    1: "male",
}
