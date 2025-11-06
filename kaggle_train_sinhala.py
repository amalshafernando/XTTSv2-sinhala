"""
Complete integrated training pipeline for XTTS-v2 Sinhala fine-tuning on Kaggle.
This script orchestrates all phases: dataset preparation → model download → vocabulary extension → GPT fine-tuning.
"""

import os
import sys
import subprocess
import gc

# Set required environment variables before any imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
from TTS.utils.manage import ModelManager

# Import configuration
try:
    from config_sinhala import (
        LANGUAGE_CODE,
        EXTENDED_VOCAB_SIZE,
        KAGGLE_DATASET_PATH,
        KAGGLE_WORKING_PATH,
        BATCH_SIZE,
        GRADIENT_ACCUMULATION,
        LEARNING_RATE,
        WEIGHT_DECAY,
        NUM_EPOCHS,
        SAVE_STEP,
        MAX_TEXT_LENGTH,
        MAX_AUDIO_LENGTH,
        print_config,
        get_kaggle_paths
    )
except ImportError:
    print("ERROR: Could not import config_sinhala. Make sure config_sinhala.py is in the same directory.")
    sys.exit(1)


def phase_1_setup_verification():
    """Phase 1: Setup verification (PyTorch, CUDA, GPU info)."""
    print("\n" + "=" * 60)
    print("PHASE 1: SETUP VERIFICATION")
    print("=" * 60)
    
    # Print configuration
    print_config()
    
    # Check PyTorch
    print(f"\n[1/4] PyTorch Version: {torch.__version__}")
    
    # Check CUDA
    print(f"[2/4] CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA Version: {torch.version.cuda}")
        print(f"    GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("    ⚠ WARNING: CUDA not available. Training will be very slow.")
    
    # Check paths
    print(f"\n[3/4] Checking paths")
    paths = get_kaggle_paths()
    for key, path in paths.items():
        print(f"    {key}: {path}")
        if key != "dataset_path":  # Dataset path might not exist yet
            os.makedirs(path, exist_ok=True)
            print(f"      ✓ Directory exists/created")
    
    # Check Python version
    print(f"\n[4/4] Python Version: {sys.version}")
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETED")
    print("=" * 60)
    
    return paths


def phase_2_prepare_dataset(kaggle_path, output_path):
    """Phase 2: Dataset preparation using prepare_dataset_sinhala.py."""
    print("\n" + "=" * 60)
    print("PHASE 2: DATASET PREPARATION")
    print("=" * 60)
    
    print(f"\n[1/2] Preparing dataset from {kaggle_path}")
    
    # Check if dataset path exists
    if not os.path.exists(kaggle_path):
        raise FileNotFoundError(f"Kaggle dataset path does not exist: {kaggle_path}")
    
    # Run dataset preparation script
    print(f"[2/2] Running prepare_dataset_sinhala.py")
    cmd = [
        sys.executable,
        "prepare_dataset_sinhala.py",
        "--kaggle_path", kaggle_path,
        "--output_path", output_path
    ]
    
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ❌ Error occurred:")
        print(result.stderr)
        raise RuntimeError(f"Dataset preparation failed with return code {result.returncode}")
    
    print(result.stdout)
    
    # Verify output files
    train_metadata = os.path.join(output_path, "metadata_train.csv")
    eval_metadata = os.path.join(output_path, "metadata_eval.csv")
    
    if not os.path.exists(train_metadata):
        raise FileNotFoundError(f"Training metadata not created: {train_metadata}")
    if not os.path.exists(eval_metadata):
        raise FileNotFoundError(f"Evaluation metadata not created: {eval_metadata}")
    
    print(f"\n    ✓ Training metadata: {train_metadata}")
    print(f"    ✓ Evaluation metadata: {eval_metadata}")
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETED")
    print("=" * 60)
    
    return train_metadata, eval_metadata


def phase_3_download_model(model_files_path):
    """Phase 3: Download XTTS-v2 pre-trained model."""
    print("\n" + "=" * 60)
    print("PHASE 3: DOWNLOAD XTTS-v2 MODEL")
    print("=" * 60)
    
    print(f"\n[1/2] Downloading XTTS-v2 model files to {model_files_path}")
    
    # Use download_checkpoint.py if available, otherwise download directly
    if os.path.exists("download_checkpoint.py"):
        print(f"[2/2] Using download_checkpoint.py")
        cmd = [
            sys.executable,
            "download_checkpoint.py",
            "--output_path", os.path.dirname(model_files_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    ⚠ Error in download_checkpoint.py: {result.stderr}")
            print(f"    Attempting direct download...")
        else:
            print(result.stdout)
            print(f"    ✓ Download completed")
    else:
        print(f"[2/2] download_checkpoint.py not found, downloading directly...")
    
    # Download files directly using ModelManager
    os.makedirs(model_files_path, exist_ok=True)
    
    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    
    DVAE_CHECKPOINT = os.path.join(model_files_path, "dvae.pth")
    MEL_NORM_FILE = os.path.join(model_files_path, "mel_stats.pth")
    
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print("    Downloading DVAE files...")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], model_files_path, progress_bar=True)
    
    # XTTS files
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"
    
    TOKENIZER_FILE = os.path.join(model_files_path, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(model_files_path, "model.pth")
    XTTS_CONFIG_FILE = os.path.join(model_files_path, "config.json")
    
    if not os.path.isfile(TOKENIZER_FILE):
        print("    Downloading tokenizer...")
        ModelManager._download_model_files([TOKENIZER_FILE_LINK], model_files_path, progress_bar=True)
    
    if not os.path.isfile(XTTS_CHECKPOINT):
        print("    Downloading XTTS checkpoint...")
        ModelManager._download_model_files([XTTS_CHECKPOINT_LINK], model_files_path, progress_bar=True)
    
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print("    Downloading XTTS config...")
        ModelManager._download_model_files([XTTS_CONFIG_LINK], model_files_path, progress_bar=True)
    
    # Verify all files exist
    required_files = [DVAE_CHECKPOINT, MEL_NORM_FILE, TOKENIZER_FILE, XTTS_CHECKPOINT, XTTS_CONFIG_FILE]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        raise FileNotFoundError(f"Missing required model files: {missing_files}")
    
    print(f"\n    ✓ All model files downloaded successfully")
    for f in required_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"      - {os.path.basename(f)} ({size_mb:.1f} MB)")
    
    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETED")
    print("=" * 60)


def phase_4_extend_vocabulary(model_files_path, train_metadata):
    """Phase 4: Vocabulary extension using extend_vocab_sinhala.py."""
    print("\n" + "=" * 60)
    print("PHASE 4: VOCABULARY EXTENSION")
    print("=" * 60)
    
    print(f"\n[1/2] Extending vocabulary for Sinhala language")
    
    # Run vocabulary extension script
    print(f"[2/2] Running extend_vocab_sinhala.py")
    cmd = [
        sys.executable,
        "extend_vocab_sinhala.py",
        "--metadata_path", train_metadata,
       # "--output_path", os.path.dirname(model_files_path),
        "--output_path", os.path.dirname(model_files_path),
        "--language", LANGUAGE_CODE,
        "--vocab_size", str(EXTENDED_VOCAB_SIZE)
    ]
    
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"    ❌ Error occurred:")
        print(result.stderr)
        raise RuntimeError(f"Vocabulary extension failed with return code {result.returncode}")
    
    print(result.stdout)
    
    # Verify vocab.json was updated
    vocab_file = os.path.join(model_files_path, "vocab.json")
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(f"vocab.json not found: {vocab_file}")
    
    # Verify config.json was updated
    config_file = os.path.join(model_files_path, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"config.json not found: {config_file}")
    
    import json
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if LANGUAGE_CODE not in config.get("languages", []):
        raise ValueError(f"Language '{LANGUAGE_CODE}' not found in config.json after extension")
    
    print(f"\n    ✓ Vocabulary extended successfully")
    print(f"    ✓ Language '{LANGUAGE_CODE}' added to config.json")
    
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETED")
    print("=" * 60)


def phase_5_dvae_finetuning(model_files_path, dataset_path):
    """Phase 5: Optional DVAE fine-tuning (commented out by default)."""
    print("\n" + "=" * 60)
    print("PHASE 5: DVAE FINE-TUNING (OPTIONAL)")
    print("=" * 60)
    
    print("\n[INFO] DVAE fine-tuning is skipped by default.")
    print("       Uncomment this phase if you have >20 hours of training data.")
    print("       To enable, uncomment the code in phase_5_dvae_finetuning() function.")
    
    # Uncomment the following code if you want to fine-tune DVAE:
    # print(f"\n[1/2] Fine-tuning DVAE")
    # if os.path.exists("train_dvae_xtts.py"):
    #     cmd = [
    #         sys.executable,
    #         "train_dvae_xtts.py",
    #         "--dataset_path", dataset_path,
    #         "--output_path", os.path.dirname(model_files_path)
    #     ]
    #     result = subprocess.run(cmd, capture_output=True, text=True)
    #     if result.returncode != 0:
    #         print(f"    ⚠ DVAE fine-tuning had errors: {result.stderr}")
    #     else:
    #         print(f"    ✓ DVAE fine-tuning completed")
    # else:
    #     print(f"    ⚠ train_dvae_xtts.py not found, skipping DVAE fine-tuning")
    
    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETED (SKIPPED)")
    print("=" * 60)


def phase_6_gpt_finetuning(model_files_path, train_metadata, eval_metadata, output_path):
    """Phase 6: GPT fine-tuning with proper command construction."""
    print("\n" + "=" * 60)
    print("PHASE 6: GPT FINE-TUNING")
    print("=" * 60)
    
    print(f"\n[1/3] Preparing GPT fine-tuning")
    print(f"    Training metadata: {train_metadata}")
    print(f"    Evaluation metadata: {eval_metadata}")
    print(f"    Language: {LANGUAGE_CODE}")
    
    # Construct metadata string for train_gpt_xtts.py
    metadata_string = f"{train_metadata},{eval_metadata},{LANGUAGE_CODE}"
    
    print(f"\n[2/3] Running GPT fine-tuning")
    print(f"    Batch size: {BATCH_SIZE}")
    print(f"    Gradient accumulation: {GRADIENT_ACCUMULATION}")
    print(f"    Learning rate: {LEARNING_RATE}")
    print(f"    Epochs: {NUM_EPOCHS}")
    
    cmd = [
        sys.executable,
        "train_gpt_xtts.py",
        "--output_path", output_path,
        "--metadatas", metadata_string,
        "--num_epochs", str(NUM_EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--grad_acumm", str(GRADIENT_ACCUMULATION),
        "--max_audio_length", str(MAX_AUDIO_LENGTH),
        "--max_text_length", str(MAX_TEXT_LENGTH),
        "--lr", str(LEARNING_RATE),
        "--weight_decay", str(WEIGHT_DECAY),
        "--save_step", str(SAVE_STEP)
    ]
    
    print(f"\n[3/3] Command: {' '.join(cmd)}")
    print(f"    Starting training...")
    print(f"    (This may take several hours)")
    
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        print(f"\n    ❌ Training failed with return code {result.returncode}")
        raise RuntimeError("GPT fine-tuning failed")
    
    print(f"\n    ✓ GPT fine-tuning completed successfully")
    
    print("\n" + "=" * 60)
    print("PHASE 6 COMPLETED")
    print("=" * 60)


def main():
    """Main training pipeline."""
    print("\n" + "=" * 80)
    print("XTTS-v2 SINHALA FINE-TUNING PIPELINE")
    print("=" * 80)
    
    try:
        # Phase 1: Setup verification
        paths = phase_1_setup_verification()
        
        # Phase 2: Dataset preparation
        train_metadata, eval_metadata = phase_2_prepare_dataset(
            KAGGLE_DATASET_PATH,
            paths["output_path"]
        )
        
        # Phase 3: Download model
        phase_3_download_model(paths["model_files_path"])
        
        # Phase 4: Extend vocabulary
        phase_4_extend_vocabulary(paths["model_files_path"], train_metadata)
        
        # Phase 5: Optional DVAE fine-tuning
        phase_5_dvae_finetuning(paths["model_files_path"], paths["output_path"])
        
        # Phase 6: GPT fine-tuning
        phase_6_gpt_finetuning(
            paths["model_files_path"],
            train_metadata,
            eval_metadata,
            paths["output_path"]
        )
        
        print("\n" + "=" * 80)
        print("ALL PHASES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nTraining completed. Checkpoints saved in: {paths['output_path']}")
        print(f"\nTo test inference, use the trained model with Sinhala text.")
        print(f"Example test texts:")
        from config_sinhala import SINHALA_TEST_TEXTS
        for text in SINHALA_TEST_TEXTS[:3]:
            print(f"  - {text}")
        
        return 0
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"❌ PIPELINE FAILED: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    exit(main())

