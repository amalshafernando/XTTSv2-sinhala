#!/usr/bin/env python3
"""
GPT Fine-tuning Script for XTTS-v2 with Sinhala Language Support

This script fine-tunes the GPT component of XTTS-v2 model for Sinhala language.
It loads pre-trained XTTS-v2 model and trains on Sinhala speech datasets.

Key Features:
- ByteLevel tokenization for Sinhala Unicode
- Grapheme-based text processing (no phoneme conversion)
- Multi-dataset support
- Comprehensive error handling
- Progress logging with status indicators

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \
        --output_path checkpoints/ \
        --metadatas datasets/metadata_train.csv,datasets/metadata_eval.csv,si \
        --num_epochs 5 \
        --batch_size 8 \
        --grad_acumm 4 \
        --max_text_length 400 \
        --max_audio_length 330750 \
        --weight_decay 1e-2 \
        --lr 5e-6 \
        --save_step 50000
"""

import os
import gc
import sys
import argparse
from pathlib import Path

# CORRECTED IMPORTS - Import from TTS library, not local modules
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import (
    GPTArgs,
    GPTTrainer,
    GPTTrainerConfig,
    XttsAudioConfig
)
from TTS.utils.manage import ModelManager


def create_xtts_trainer_parser():
    """
    Create argument parser for XTTS GPT training.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="GPT Fine-tuning for XTTS-v2 with Sinhala Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic Sinhala training
  CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \\
    --output_path checkpoints/ \\
    --metadatas datasets/metadata_train.csv,datasets/metadata_eval.csv,si \\
    --num_epochs 5 \\
    --batch_size 8

  # Multiple datasets
  CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \\
    --output_path checkpoints/ \\
    --metadatas dataset1/train.csv,dataset1/eval.csv,si \\
              dataset2/train.csv,dataset2/eval.csv,si \\
    --num_epochs 10 \\
    --batch_size 8
        """
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where checkpoints will be saved"
    )
    parser.add_argument(
        "--metadatas",
        nargs='+',
        type=str,
        required=True,
        help="Dataset metadata as: train_csv,eval_csv,language [train_csv,eval_csv,language ...]"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Mini batch size (default: 8, reduce to 4 for low VRAM)"
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        default=330750,
        help="Max audio length in samples (~13.6s at 24kHz, default: 330750)"
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=400,
        help="Max text length in tokens (default: 400)"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for optimizer (default: 1e-2)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)"
    )
    parser.add_argument(
        "--save_step",
        type=int,
        default=50000,
        help="Save checkpoint every N steps (default: 50000)"
    )
    
    return parser


def train_gpt(
    metadatas,
    output_path,
    num_epochs=5,
    batch_size=8,
    grad_acumm=4,
    max_audio_length=330750,
    max_text_length=400,
    lr=5e-6,
    weight_decay=1e-2,
    save_step=50000
):
    """
    Train GPT component of XTTS-v2 for Sinhala language.
    
    Args:
        metadatas: List of metadata specs (train_csv,eval_csv,language)
        output_path: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        grad_acumm: Gradient accumulation steps
        max_audio_length: Maximum audio length in samples
        max_text_length: Maximum text length in tokens
        lr: Learning rate
        weight_decay: Weight decay parameter
        save_step: Save checkpoint every N steps
        
    Returns:
        str: Path to training output directory
        
    Raises:
        ValueError: If dataset configuration is invalid
        FileNotFoundError: If required files are not found
    """
    
    print(f"\n{'='*80}")
    print("XTTS-v2 GPT FINE-TUNING FOR SINHALA")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    print("📋 Configuration:")
    print(f"   - Output path: {output_path}")
    print(f"   - Number of epochs: {num_epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Gradient accumulation: {grad_acumm}")
    print(f"   - Learning rate: {lr}")
    print(f"   - Weight decay: {weight_decay}")
    print(f"   - Max audio length: {max_audio_length}")
    print(f"   - Max text length: {max_text_length}")
    
    # Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None
    
    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  # For multi-gpu training set to False
    START_WITH_EVAL = False
    
    # ========================================================================
    # DATASET CONFIGURATION
    # ========================================================================
    
    print(f"\n{'─'*80}")
    print("PROCESSING DATASETS")
    print(f"{'─'*80}\n")
    
    DATASETS_CONFIG_LIST = []
    
    for metadata_spec in metadatas:
        print(f"📂 Processing metadata spec: {metadata_spec}")
        
        # Parse metadata specification
        parts = metadata_spec.split(",")
        if len(parts) != 3:
            raise ValueError(
                f"❌ Invalid metadata format: {metadata_spec}\n"
                f"   Expected: train_csv,eval_csv,language\n"
                f"   Got: {metadata_spec}"
            )
        
        train_csv, eval_csv, language = parts
        train_csv = train_csv.strip()
        eval_csv = eval_csv.strip()
        language = language.strip()
        
        # Validate files exist
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"❌ Training CSV not found: {train_csv}")
        if not os.path.exists(eval_csv):
            raise FileNotFoundError(f"❌ Evaluation CSV not found: {eval_csv}")
        
        print(f"   ✅ Train CSV: {os.path.basename(train_csv)}")
        print(f"   ✅ Eval CSV: {os.path.basename(eval_csv)}")
        print(f"   ✅ Language: {language}")
        
        # Create dataset config
        config_dataset = BaseDatasetConfig(
            formatter="coqui",
            dataset_name="ft_dataset",
            path=os.path.dirname(train_csv),
            meta_file_train=os.path.basename(train_csv),
            meta_file_val=os.path.basename(eval_csv),
            language=language,
        )
        
        DATASETS_CONFIG_LIST.append(config_dataset)
        print(f"   ✅ Dataset config created\n")
    
    # ========================================================================
    # DOWNLOAD MODEL FILES
    # ========================================================================
    
    print(f"{'─'*80}")
    print("DOWNLOADING XTTS-v2 MODEL FILES")
    print(f"{'─'*80}\n")
    
    OUT_PATH = output_path
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
    
    print(f"📁 Checkpoint directory: {CHECKPOINTS_OUT_PATH}\n")
    
    # DVAE files (audio encoder)
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
    
    # Download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print("📥 Downloading DVAE files...")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True
        )
        print("✅ DVAE files downloaded\n")
    else:
        print("✅ DVAE files already present\n")
    
    # XTTS-v2 model files
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    XTTS_CONFIG_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/config.json"
    
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))
    
    # Download tokenizer if needed
    if not os.path.isfile(TOKENIZER_FILE):
        print("📥 Downloading XTTS-v2 tokenizer...")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True
        )
        print("✅ Tokenizer downloaded\n")
    else:
        print("✅ Tokenizer already present\n")
    
    # Download model checkpoint if needed
    if not os.path.isfile(XTTS_CHECKPOINT):
        print("📥 Downloading XTTS-v2 checkpoint...")
        ModelManager._download_model_files(
            [XTTS_CHECKPOINT_LINK],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True
        )
        print("✅ Model checkpoint downloaded\n")
    else:
        print("✅ Model checkpoint already present\n")
    
    # Download config if needed
    if not os.path.isfile(XTTS_CONFIG_FILE):
        print("📥 Downloading XTTS-v2 config...")
        ModelManager._download_model_files(
            [XTTS_CONFIG_LINK],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True
        )
        print("✅ Config file downloaded\n")
    else:
        print("✅ Config file already present\n")
    
    # ========================================================================
    # MODEL CONFIGURATION
    # ========================================================================
    
    print(f"{'─'*80}")
    print("CONFIGURING MODEL")
    print(f"{'─'*80}\n")
    
    # GPT Model Arguments
    model_args = GPTArgs(
        max_conditioning_length=132300,  # ~6 seconds
        min_conditioning_length=11025,   # ~0.5 seconds
        debug_loading_failures=False,
        max_wav_length=max_audio_length,
        max_text_length=max_text_length,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )
    print("✅ GPT model arguments configured")
    
    # Audio Configuration
    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000
    )
    print("✅ Audio configuration configured\n")
    
    # Training Configuration
    config = GPTTrainerConfig()
    config.load_json(XTTS_CONFIG_FILE)
    print("✅ Base configuration loaded from checkpoint")
    
    # ========================================================================
    # SINHALA-SPECIFIC CONFIGURATION (CRITICAL)
    # ========================================================================
    
    print(f"\n📝 Setting Sinhala-specific configuration:")
    
    # Sinhala uses grapheme-based text processing, not phoneme conversion
    config.use_phonemes = False  # CRITICAL: Set to False for Sinhala
    config.phoneme_language = None  # CRITICAL: Set to None for Sinhala
    config.text_cleaner = "english_cleaners"  # Works for all scripts
    
    print(f"   ✅ use_phonemes = False (grapheme-based)")
    print(f"   ✅ phoneme_language = None")
    print(f"   ✅ text_cleaner = english_cleaners\n")
    
    # ========================================================================
    # TRAINING PARAMETERS
    # ========================================================================
    
    config.epochs = num_epochs
    config.output_path = OUT_PATH
    config.model_args = model_args
    config.run_name = RUN_NAME
    config.project_name = PROJECT_NAME
    config.run_description = "GPT XTTS fine-tuning for Sinhala"
    config.dashboard_logger = DASHBOARD_LOGGER
    config.logger_uri = LOGGER_URI
    config.audio = audio_config
    config.batch_size = batch_size
    config.num_loader_workers = 8
    config.eval_split_max_size = 256
    config.print_step = 50
    config.plot_step = 100
    config.log_model_step = 100
    config.save_step = save_step
    config.save_n_checkpoints = 1
    config.save_checkpoints = True
    config.print_eval = False
    config.optimizer = "AdamW"
    config.optimizer_wd_only_on_weights = OPTIMIZER_WD_ONLY_ON_WEIGHTS
    config.optimizer_params = {
        "betas": [0.9, 0.96],
        "eps": 1e-8,
        "weight_decay": weight_decay
    }
    config.lr = lr
    config.lr_scheduler = "MultiStepLR"
    config.lr_scheduler_params = {
        "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
        "gamma": 0.5,
        "last_epoch": -1
    }
    config.test_sentences = []
    
    print("📊 Training configuration updated:")
    print(f"   - Optimizer: AdamW")
    print(f"   - Learning rate: {lr}")
    print(f"   - Weight decay: {weight_decay}")
    print(f"   - Save step: {save_step}\n")
    
    # ========================================================================
    # INITIALIZE MODEL
    # ========================================================================
    
    print(f"{'─'*80}")
    print("INITIALIZING MODEL")
    print(f"{'─'*80}\n")
    
    print("🔧 Initializing GPT model from config...")
    model = GPTTrainer.init_from_config(config)
    print("✅ Model initialized\n")
    
    # ========================================================================
    # LOAD TRAINING SAMPLES
    # ========================================================================
    
    print(f"{'─'*80}")
    print("LOADING TRAINING SAMPLES")
    print(f"{'─'*80}\n")
    
    print("📦 Loading training and evaluation samples...")
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    print(f"✅ Training samples: {len(train_samples)}")
    print(f"✅ Evaluation samples: {len(eval_samples)}\n")
    
    # ========================================================================
    # INITIALIZE TRAINER
    # ========================================================================
    
    print(f"{'─'*80}")
    print("INITIALIZING TRAINER")
    print(f"{'─'*80}\n")
    
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=grad_acumm
        ),
        config,
        output_path=os.path.join(output_path, "run", "training"),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    print("✅ Trainer initialized\n")
    
    # ========================================================================
    # START TRAINING
    # ========================================================================
    
    print(f"{'='*80}")
    print("🚀 STARTING TRAINING")
    print(f"{'='*80}\n")
    
    try:
        trainer.fit()
        print("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {str(e)}")
        raise
    
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    print(f"\n{'─'*80}")
    print("CLEANUP")
    print(f"{'─'*80}\n")
    
    # Get training output path
    trainer_out_path = trainer.output_path
    print(f"📁 Training output saved to: {trainer_out_path}")
    
    # Get longest text audio file for speaker reference
    print(f"\n🔍 Finding longest text for speaker reference...")
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]
    print(f"✅ Speaker reference: {speaker_ref}")
    
    # Deallocate VRAM and RAM
    print(f"\n🧹 Cleaning up memory...")
    del model, trainer, train_samples, eval_samples
    gc.collect()
    print("✅ Memory cleaned\n")
    
    return trainer_out_path


def main():
    """Main entry point for the training script."""
    
    try:
        parser = create_xtts_trainer_parser()
        args = parser.parse_args()
        
        trainer_out_path = train_gpt(
            metadatas=args.metadatas,
            output_path=args.output_path,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            grad_acumm=args.grad_acumm,
            max_audio_length=args.max_audio_length,
            max_text_length=args.max_text_length,
            weight_decay=args.weight_decay,
            lr=args.lr,
            save_step=args.save_step
        )
        
        print(f"\n{'='*80}")
        print("✅ TRAINING PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"\n🎉 Checkpoint saved to: {trainer_out_path}")
        print(f"\nNext steps:")
        print(f"1. Run inference with the fine-tuned model")
        print(f"2. Test with Sinhala text using the trained checkpoint")
        print(f"\n{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
