"""
Complete integrated training pipeline for XTTS-v2 Sinhala fine-tuning on Kaggle.
This script orchestrates all phases: dataset preparation → model download → vocabulary extension → GPT fine-tuning.
"""

import os
import sys
import subprocess
import gc
import argparse
import math

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
    
    gpu_info = []

    # Check CUDA
    print(f"[2/4] CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    CUDA Version: {torch.version.cuda}")
        print(f"    GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem_gb = torch.cuda.get_device_properties(i).total_memory / 1024**3
            gpu_info.append({
                "index": i,
                "name": name,
                "memory_gb": mem_gb,
            })
            print(f"    GPU {i}: {name}")
            print(f"      Memory: {mem_gb:.2f} GB")
    else:
        print("    ⚠ WARNING: CUDA not available. Training will be very slow.")
    
    # Check paths
    print(f"\n[3/4] Checking paths")
    paths = get_kaggle_paths()
    for key, path in paths.items():
        print(f"    {key}: {path}")
        if key == "dataset_path":
            continue

        target_dir = path
        _, ext = os.path.splitext(path)
        if ext:
            target_dir = os.path.dirname(path)

        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
            print(f"      ✓ Directory ready: {target_dir}")
    
    # Check Python version
    print(f"\n[4/4] Python Version: {sys.version}")
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETED")
    print("=" * 60)
    
    return paths, gpu_info


def _count_samples(metadata_path):
    count = 0
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            for idx, _ in enumerate(f):
                if idx == 0:
                    continue  # skip header
                count += 1
    except FileNotFoundError:
        return 0
    return count


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
    
    dataset_stats = {
        "train_samples": _count_samples(train_metadata),
        "eval_samples": _count_samples(eval_metadata),
    }

    print(f"    ✓ Train samples: {dataset_stats['train_samples']}")
    print(f"    ✓ Eval samples : {dataset_stats['eval_samples']}")

    return train_metadata, eval_metadata, dataset_stats


def auto_tune_training_params(gpu_info, dataset_stats):
    """Derive batch size, accumulation, and scheduler settings for Kaggle GPU."""

    batch_size = BATCH_SIZE
    grad_accum = GRADIENT_ACCUMULATION
    learning_rate = LEARNING_RATE
    save_step = SAVE_STEP

    # Adjust batch size based on the largest available GPU
    max_mem = max((gpu.get("memory_gb", 0) for gpu in gpu_info), default=0)
    if max_mem and max_mem < 16:
        batch_size = max(2, batch_size // 2)
        grad_accum = max(grad_accum, math.ceil((BATCH_SIZE * GRADIENT_ACCUMULATION) / batch_size))
    elif 16 <= max_mem < 24:
        batch_size = min(batch_size, 6)
    elif max_mem >= 40:
        batch_size = min(max(batch_size, 10), 12)

    effective_original = BATCH_SIZE * GRADIENT_ACCUMULATION
    effective_new = batch_size * grad_accum

    if effective_new < effective_original and effective_new > 0:
        scale = effective_new / effective_original
        learning_rate = max(1e-6, LEARNING_RATE * scale)

    train_samples = dataset_stats.get("train_samples", 0)
    steps_per_epoch = max(1, math.ceil(train_samples / max(effective_new, 1)))
    save_step = max(steps_per_epoch * 2, min(SAVE_STEP, steps_per_epoch * 10))

    tuned = {
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": learning_rate,
        "save_step": save_step,
        "steps_per_epoch": steps_per_epoch,
        "effective_batch": effective_new,
    }

    print("\n[Auto-Tune] Training hyperparameters suggested:")
    print(f"    Batch size        : {batch_size}")
    print(f"    Grad accumulation : {grad_accum}")
    print(f"    Effective batch   : {effective_new}")
    print(f"    Learning rate     : {learning_rate:.2e}")
    print(f"    Save step         : {save_step}")
    print(f"    Steps per epoch   : {steps_per_epoch}")

    return tuned


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
    """Phase 5: DVAE fine-tuning (enabled)."""
    print("\n" + "=" * 60)
    print("PHASE 5: DVAE FINE-TUNING")
    print("=" * 60)

    train_csv = os.path.join(dataset_path, "metadata_train.csv")
    eval_csv = os.path.join(dataset_path, "metadata_eval.csv")
    out_path = os.path.dirname(model_files_path)

    if not (os.path.exists(train_csv) and os.path.exists(eval_csv)):
        raise FileNotFoundError("DVAE: train/eval metadata CSV not found.")

    print(f"\n[1/2] Fine-tuning DVAE")
    if os.path.exists("train_dvae_xtts.py"):
        cmd = [
            sys.executable,
            "train_dvae_xtts.py",
            f"--output_path={out_path}",
            f"--train_csv_path={train_csv}",
            f"--eval_csv_path={eval_csv}",
            '--language="si"',
            "--num_epochs=3",
            "--batch_size=256",
            "--lr=5e-6",
        ]
        print("    Command:", " ".join(cmd))
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError("DVAE fine-tuning failed")
        else:
            print(f"    ✓ DVAE fine-tuning completed")
    else:
        raise FileNotFoundError("train_dvae_xtts.py not found")

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETED")
    print("=" * 60)


def phase_6_gpt_finetuning(model_files_path, train_metadata, eval_metadata, output_path, training_params):
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
    batch_size = training_params.get("batch_size", BATCH_SIZE)
    grad_accum = training_params.get("grad_accum", GRADIENT_ACCUMULATION)
    learning_rate = training_params.get("learning_rate", LEARNING_RATE)
    save_step = training_params.get("save_step", SAVE_STEP)

    print(f"    Batch size: {batch_size}")
    print(f"    Gradient accumulation: {grad_accum}")
    print(f"    Learning rate: {learning_rate}")
    print(f"    Save step: {save_step}")
    print(f"    Epochs: {NUM_EPOCHS}")
    
    os.makedirs(output_path, exist_ok=True)  # ensure exists before run
    cmd = [
        sys.executable,
        "train_gpt_xtts.py",
        "--output_path", output_path,  # now a checkpoints/training dir
        "--metadatas", metadata_string,
        "--num_epochs", str(NUM_EPOCHS),
        "--batch_size", str(batch_size),
        "--grad_acumm", str(grad_accum),
        "--max_audio_length", str(MAX_AUDIO_LENGTH),
        "--max_text_length", str(MAX_TEXT_LENGTH),
        "--lr", str(learning_rate),
        "--weight_decay", str(WEIGHT_DECAY),
        "--save_step", str(save_step),
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
    # ------------------------------------------------------------------
    # CLI arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="XTTS-v2 Sinhala end-to-end pipeline")
    parser.add_argument(
        "--enable_dvae",
        action="store_true",
        help="Run DVAE fine-tuning (Phase 5). Optional; typically only beneficial with >20h audio.",
    )
    args, unknown = parser.parse_known_args()
    print("\n" + "=" * 80)
    print("XTTS-v2 SINHALA FINE-TUNING PIPELINE")
    print("=" * 80)
    
    try:
        # Phase 1: Setup verification
        paths, gpu_info = phase_1_setup_verification()
        
        # Phase 2: Dataset preparation
        train_metadata, eval_metadata, dataset_stats = phase_2_prepare_dataset(
            KAGGLE_DATASET_PATH,
            paths["output_path"]
        )

        training_params = auto_tune_training_params(gpu_info, dataset_stats)
        
        # Phase 3: Download model
        phase_3_download_model(paths["model_files_path"])
        
        # Phase 4: Extend vocabulary
        phase_4_extend_vocabulary(paths["model_files_path"], train_metadata)
        
        # Phase 5: DVAE fine-tuning (optional)
        if args.enable_dvae:
            phase_5_dvae_finetuning(paths["model_files_path"], paths["output_path"])
        else:
            print("\n" + "=" * 60)
            print("PHASE 5: DVAE FINE-TUNING (SKIPPED)")
            print("=" * 60)
            print("Info: Use --enable_dvae to run DVAE fine-tuning (recommended only for large datasets >20h)")
        
        # Phase 6: GPT fine-tuning
        phase_6_gpt_finetuning(
            paths["model_files_path"],
            train_metadata,
            eval_metadata,
            paths["training_output"],  # use a training output directory, not datasets
            training_params,
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

