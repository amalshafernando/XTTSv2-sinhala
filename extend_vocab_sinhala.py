"""
Extend XTTS-v2 vocabulary and configuration for Sinhala script.
This script creates a BPE tokenizer trained specifically on Sinhala text
and updates the config.json to register 'si' as a supported language.
"""

import argparse
import json
import os
import shutil

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


def combine_tokenizers(old_tokenizer_dir, new_tokenizer_dir, save_dir):
    """
    Combine old and new tokenizer vocabularies.
    
    Args:
        old_tokenizer_dir: Path to old tokenizer directory
        new_tokenizer_dir: Path to new tokenizer directory
        save_dir: Path to save merged tokenizer
    """
    # Load both vocab.json files
    with open(os.path.join(old_tokenizer_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
        json1 = json.load(f)
    
    with open(os.path.join(new_tokenizer_dir, 'vocab.json'), 'r', encoding='utf-8') as f:
        json2 = json.load(f)
    
    # Create a new vocabulary with union of both
    new_vocab = {}
    idx = 0
    
    # Add words from old tokenizer first
    for word in json1.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1
    
    # Add words from new tokenizer
    for word in json2.keys():
        if word not in new_vocab.keys():
            new_vocab[word] = idx
            idx += 1
    
    # Make the directory if necessary
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the merged vocab
    with open(os.path.join(save_dir, 'vocab.json'), 'w', encoding='utf-8') as fp:
        json.dump(new_vocab, fp, ensure_ascii=False, indent=2)
    
    # Merge the two merges files
    # Concatenate them, but ignore the first line of the second file
    if os.path.exists(os.path.join(old_tokenizer_dir, 'merges.txt')):
        shutil.copy(os.path.join(old_tokenizer_dir, 'merges.txt'), os.path.join(save_dir, 'merges.txt'))
    
    # Append new merges (skip first line which is version info)
    if os.path.exists(os.path.join(new_tokenizer_dir, 'merges.txt')):
        with open(os.path.join(new_tokenizer_dir, 'merges.txt'), 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 1:
                with open(os.path.join(save_dir, 'merges.txt'), 'a', encoding='utf-8') as f:
                    f.writelines(lines[1:])  # Skip first line


def extend_tokenizer(args):
    """
    Extend the XTTS-v2 tokenizer with Sinhala-specific vocabulary.
    
    Args:
        args: Command line arguments
    """
    print("=" * 60)
    print("EXTENDING VOCABULARY FOR SINHALA")
    print("=" * 60)
    
    root = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/")
    
    # Check if required files exist
    vocab_json_path = os.path.join(root, "vocab.json")
    if not os.path.exists(vocab_json_path):
        raise FileNotFoundError(f"Original vocab.json not found at {vocab_json_path}. Please download XTTS-v2 model first.")
    
    print(f"\n[1/6] Backing up existing tokenizer files")
    old_tokenizer_path = os.path.join(root, "old_tokenizer/")
    os.makedirs(old_tokenizer_path, exist_ok=True)
    # Copy existing vocab/merges into backup folder
    shutil.copy(os.path.join(root, "vocab.json"), os.path.join(old_tokenizer_path, "vocab.json"))
    merges_src = os.path.join(root, "merges.txt")
    if os.path.exists(merges_src):
        shutil.copy(merges_src, os.path.join(old_tokenizer_path, "merges.txt"))
    print(f"    Saved existing tokenizer to {old_tokenizer_path}")
    
    # Load Sinhala text from metadata
    print(f"\n[2/6] Loading Sinhala text from {args.metadata_path}")
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    traindf = pd.read_csv(args.metadata_path, sep="|")
    if 'text' not in traindf.columns:
        raise ValueError("Metadata file must contain 'text' column")
    
    texts = traindf.text.tolist()
    print(f"    Loaded {len(texts)} text samples")
    
    # Create new BPE tokenizer with ByteLevel pre-tokenizer
    print(f"\n[3/6] Creating new BPE tokenizer with vocabulary size {args.vocab_size}")
    new_tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    new_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    
    # Set up trainer with special tokens
    special_tokens = args.special_tokens + [f"[{args.language}]"]
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        min_frequency=2
    )
    
    # Train tokenizer on Sinhala text
    print(f"    Training tokenizer on Sinhala text...")
    new_tokenizer.train_from_iterator(iter(texts), trainer=trainer)
    
    # Add special tokens
    new_tokenizer.add_special_tokens(special_tokens)
    
    # Set post-processor
    new_tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
    
    # Save new tokenizer
    new_tokenizer_path = os.path.join(root, "new_tokenizer/")
    os.makedirs(new_tokenizer_path, exist_ok=True)
    new_tokenizer.model.save(new_tokenizer_path)
    print(f"    Saved new tokenizer to {new_tokenizer_path}")
    
    # Merge tokenizers
    print(f"\n[4/6] Merging old and new tokenizers")
    merged_tokenizer_path = os.path.join(root, "merged_tokenizer/")
    combine_tokenizers(
        old_tokenizer_path,
        new_tokenizer_path,
        merged_tokenizer_path
    )
    print(f"    Merged tokenizer saved to {merged_tokenizer_path}")
    
    # Overwrite original vocab/merges with merged ones
    print(f"\n[5/6] Overwriting original vocab/merges with merged tokenizer")
    shutil.copy(os.path.join(merged_tokenizer_path, 'vocab.json'), os.path.join(root, 'vocab.json'))
    merged_merges = os.path.join(merged_tokenizer_path, 'merges.txt')
    if os.path.exists(merged_merges):
        shutil.copy(merged_merges, os.path.join(root, 'merges.txt'))
    print(f"    Replaced files in {root}")

    # Basic validation: ensure language token exists in final vocab
    with open(os.path.join(root, 'vocab.json'), 'r', encoding='utf-8') as f:
        final_vocab = json.load(f)
    lang_token = f"[{args.language}]"
    if lang_token in final_vocab:
        print(f"    ✓ Verified presence of language token '{lang_token}' in final vocab")
    else:
        raise ValueError(f"Language token '{lang_token}' not found in merged vocab.json")
    
    # Clean up temporary directories
    print(f"\n[6/6] Cleaning up temporary files")
    for temp_dir in [old_tokenizer_path, new_tokenizer_path, merged_tokenizer_path]:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"    Removed {temp_dir}")
    
    print("\n" + "=" * 60)
    print("VOCABULARY EXTENSION COMPLETED SUCCESSFULLY")
    print("=" * 60)


def adjust_config(args):
    """
    Update config.json to add Sinhala language support.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print("UPDATING CONFIG.JSON FOR SINHALA")
    print("=" * 60)
    
    config_path = os.path.join(args.output_path, "XTTS_v2.0_original_model_files/config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}. Please download XTTS-v2 model first.")
    
    print(f"\n[1/3] Loading config.json from {config_path}")
    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    
    # Add language to supported languages if not already present
    if "languages" not in config:
        config["languages"] = []
    
    if args.language not in config["languages"]:
        config["languages"].append(args.language)
        print(f"    Added '{args.language}' to supported languages")
    else:
        print(f"    Language '{args.language}' already in supported languages")
    
    # Add language_settings for Sinhala
    if "language_settings" not in config:
        config["language_settings"] = {}
    
    if args.language not in config["language_settings"]:
        config["language_settings"][args.language] = {
            "use_phonemes": False,
            "phoneme_language": None
        }
        print(f"    Added language settings for '{args.language}'")
    else:
        print(f"    Language settings for '{args.language}' already exist")
        # Update existing settings
        config["language_settings"][args.language]["use_phonemes"] = False
        config["language_settings"][args.language]["phoneme_language"] = None
        print(f"    Updated language settings for '{args.language}'")
    
    # Save updated config
    print(f"\n[2/3] Saving updated config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"    Config saved to {config_path}")
    
    # Verify update
    print(f"\n[3/3] Verifying configuration")
    with open(config_path, "r", encoding='utf-8') as f:
        verify_config = json.load(f)
    
    if args.language in verify_config.get("languages", []):
        print(f"    ✓ Language '{args.language}' confirmed in supported languages")
    else:
        raise ValueError(f"Language '{args.language}' not found in config after update")
    
    if args.language in verify_config.get("language_settings", {}):
        print(f"    ✓ Language settings for '{args.language}' confirmed")
    else:
        raise ValueError(f"Language settings for '{args.language}' not found in config after update")
    
    print("\n" + "=" * 60)
    print("CONFIG UPDATE COMPLETED SUCCESSFULLY")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extend XTTS-v2 vocabulary and configuration for Sinhala language"
    )
    
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to training metadata CSV file (pipe-separated format)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory where XTTS-v2 model files are stored"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="si",
        help="Language code (default: 'si' for Sinhala)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=2000,
        help="Extended vocabulary size (default: 2000)"
    )
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs='+',
        default=["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<pad>"],
        help="Special tokens for tokenizer (default: ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<pad>'])"
    )
    
    args = parser.parse_args()
    
    try:
        # Extend vocabulary
        extend_tokenizer(args)
        
        # Update config
        adjust_config(args)
        
        print("\n" + "=" * 60)
        print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

