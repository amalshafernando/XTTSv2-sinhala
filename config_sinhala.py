"""
Centralized configuration for Sinhala language support in XTTS-v2.
This file contains all Sinhala-specific parameters for training and inference.
"""

import os


# Language Configuration
LANGUAGE_CODE = "si"  # ISO 639-1 code for Sinhala
EXTENDED_VOCAB_SIZE = 2000  # Vocabulary size for BPE tokenizer

# Kaggle Paths
KAGGLE_DATASET_PATH = "/kaggle/input/sinhala-tts-dataset/sinhala-tts-dataset"
KAGGLE_WORKING_PATH = "/kaggle/working"

# Training Parameters
BATCH_SIZE = 8  # Reduce to 4 if CUDA OOM
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 5
SAVE_STEP = 50000

# GPT Configuration
MAX_TEXT_LENGTH = 400  # Maximum text length in tokens
MAX_AUDIO_LENGTH = 330750  # Maximum audio length in samples (~13.6 seconds at 24kHz)

# Text Processing Configuration
USE_PHONEMES = False  # Sinhala uses grapheme-based approach (no phoneme conversion)
PHONEME_LANGUAGE = None  # No phoneme converter needed for Sinhala
TEXT_CLEANER = "english_cleaners"  # Works for all scripts

# Dataset Configuration
DATASET_FORMAT = "coqui"  # XTTS format: pipe-separated values
METADATA_TRAIN_FILE = "metadata_train.csv"
METADATA_EVAL_FILE = "metadata_eval.csv"

# Model Download Links
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"

# Special Tokens for BPE Tokenizer
SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<pad>"]

# Sample Sinhala Test Texts for Inference Testing
SINHALA_TEST_TEXTS = [
    "නමස්කාරය",
    "සිංහල භාෂාව",
    "කෘතිම බුද්ධිය",
    "මෙය පරීක්ෂණ පෙළකි",
    "ඔබට සුභ දිනයක් වේවා"
]


def get_kaggle_paths():
    """Get Kaggle-specific paths for dataset and working directory."""
    return {
        "dataset_path": KAGGLE_DATASET_PATH,
        "working_path": KAGGLE_WORKING_PATH,
        "output_path": os.path.join(KAGGLE_WORKING_PATH, "checkpoints"),
        "model_files_path": os.path.join(KAGGLE_WORKING_PATH, "checkpoints", "XTTS_v2.0_original_model_files")
    }


def get_local_paths(base_path="."):
    """Get local paths for dataset and working directory."""
    return {
        "dataset_path": os.path.join(base_path, "dataset"),
        "working_path": os.path.join(base_path, "output"),
        "output_path": os.path.join(base_path, "output", "checkpoints"),
        "model_files_path": os.path.join(base_path, "output", "checkpoints", "XTTS_v2.0_original_model_files")
    }


def print_config():
    """Print all configuration settings."""
    print("=" * 60)
    print("SINHALA XTTS-v2 CONFIGURATION")
    print("=" * 60)
    print(f"Language Code: {LANGUAGE_CODE}")
    print(f"Extended Vocabulary Size: {EXTENDED_VOCAB_SIZE}")
    print(f"\nTraining Parameters:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Number of Epochs: {NUM_EPOCHS}")
    print(f"  Save Step: {SAVE_STEP}")
    print(f"\nGPT Configuration:")
    print(f"  Max Text Length: {MAX_TEXT_LENGTH}")
    print(f"  Max Audio Length: {MAX_AUDIO_LENGTH}")
    print(f"\nText Processing:")
    print(f"  Use Phonemes: {USE_PHONEMES}")
    print(f"  Phoneme Language: {PHONEME_LANGUAGE}")
    print(f"  Text Cleaner: {TEXT_CLEANER}")
    print(f"\nDataset Format: {DATASET_FORMAT}")
    print(f"  Train Metadata: {METADATA_TRAIN_FILE}")
    print(f"  Eval Metadata: {METADATA_EVAL_FILE}")
    print(f"\nSpecial Tokens: {SPECIAL_TOKENS}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()

