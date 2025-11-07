#!/usr/bin/env python3
"""
Extended vocabulary and configuration for Sinhala language in XTTS-v2

This script:
1. Loads Sinhala texts from metadata CSV
2. Creates BPE tokenizer using ByteLevel pre-tokenization
3. Saves extended vocabulary (15,000 tokens)
4. Updates config.json for Sinhala language support

Key improvements over original:
- ByteLevel pre-tokenizer (perfect for Sinhala Unicode)
- No subdirectory creation (saves directly to output_path)
- Robust error handling with helpful messages
- Sinhala-specific configuration
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import ByteLevel as ByteLevelProcessor
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import sys

from sinhala_text_normalizer import normalize_sinhala_text


class SinhalaBPETokenizer:
    """Handles Sinhala tokenization with ByteLevel BPE."""
    
    SINHALA_UNICODE_START = 0x0D80
    SINHALA_UNICODE_END = 0x0DFF
    
    def __init__(self, vocab_size=15000):
        """
        Initialize Sinhala tokenizer.
        
        Args:
            vocab_size: Size of BPE vocabulary (default: 15000)
        """
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.vocab = None
    
    def load_sinhala_texts(self, metadata_path):
        """
        Load Sinhala texts from metadata CSV.
        
        Expected format: audio_file|text|speaker_name
        Text is expected in column 1 (0-indexed)
        
        Args:
            metadata_path: Path to metadata_train.csv
            
        Returns:
            List of Sinhala text strings
            
        Raises:
            FileNotFoundError: If metadata file not found
            ValueError: If CSV format is incorrect
        """
        print(f"\n📖 Loading Sinhala texts from: {metadata_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"❌ Metadata file not found: {metadata_path}")

        path = Path(metadata_path)
        texts: List[str]

        if path.suffix.lower() == ".jsonl":
            texts = self._load_from_jsonl(path)
        else:
            texts = self._load_from_csv(path)
        
        original_count = len(texts)
        texts = list(dict.fromkeys(texts))  # Preserve order while deduplicating
        deduped = original_count - len(texts)

        print(f"✅ Loaded {len(texts)} Sinhala text samples")
        if deduped:
            print(f"   - Removed {deduped} duplicate entries during normalization")
        
        # Show sample texts
        print(f"\n📝 Sample Sinhala texts:")
        for i, text in enumerate(texts[:3]):
            print(f"  {i+1}. {text[:60]}{'...' if len(text) > 60 else ''}")
        
        # Statistics
        total_chars = sum(len(text) for text in texts)
        sinhala_chars = sum(
            sum(1 for char in text 
                if self.SINHALA_UNICODE_START <= ord(char) <= self.SINHALA_UNICODE_END)
            for text in texts
        )
        
        print(f"\n📊 Text Statistics:")
        print(f"   - Total characters: {total_chars:,}")
        print(f"   - Sinhala characters: {sinhala_chars:,} ({100*sinhala_chars/max(total_chars,1):.1f}%)")
        
        return texts

    @staticmethod
    def _load_from_csv(path: Path) -> List[str]:
        """Load texts from CSV, handling both headered and headerless formats."""

        read_kwargs = {
            "sep": "|",
            "encoding": "utf-8",
            "dtype": str,
        }

        try:
            df = pd.read_csv(path, **read_kwargs)
        except Exception as exc:
            raise ValueError(f"❌ Error reading CSV '{path}': {exc}")

        if df.shape[1] == 0:
            raise ValueError(f"❌ CSV '{path}' contains no columns")

        # Attempt to locate the text column by name; fallback to column index 1.
        text_column = None
        for candidate in ("text", "transcript", 1):
            if isinstance(candidate, str) and candidate in df.columns:
                text_column = candidate
                break
            if candidate == 1 and df.shape[1] > 1:
                text_column = df.columns[1]

        if text_column is None:
            raise ValueError(
                "❌ Could not identify text column. Expected column named 'text' or second column in CSV."
            )

        texts = df[text_column].fillna("").astype(str).tolist()
        texts = [normalize_sinhala_text(text) for text in texts if text.strip()]

        if not texts:
            raise ValueError(f"❌ No Sinhala texts found in CSV '{path}'")

        return texts

    @staticmethod
    def _load_from_jsonl(path: Path) -> List[str]:
        """Load texts from JSONL manifest."""

        texts: List[str] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    text = normalize_sinhala_text(str(record.get("text", "")))
                    if text:
                        texts.append(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"❌ Invalid JSON in '{path}': {exc}")

        if not texts:
            raise ValueError(f"❌ No Sinhala texts found in JSONL '{path}'")

        return texts
    
    def train_tokenizer(self, texts):
        """
        Train BPE tokenizer on Sinhala texts.
        
        Uses ByteLevel pre-tokenizer which is optimal for Unicode scripts
        including Sinhala (abugida script).
        
        Args:
            texts: List of Sinhala text strings
            
        Raises:
            ValueError: If training fails
        """
        print(f"\n{'='*80}")
        print(f"TRAINING SINHALA BPE TOKENIZER")
        print(f"Target Vocabulary Size: {self.vocab_size}")
        print(f"{'='*80}")
        
        try:
            # Initialize tokenizer with BPE model
            self.tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
            
            # Set pre-tokenizer to ByteLevel (crucial for Sinhala Unicode)
            self.tokenizer.pre_tokenizer = ByteLevel()
            self.tokenizer.post_processor = ByteLevelProcessor()
            self.tokenizer.decoder = ByteLevelDecoder()
            
            print(f"\n✅ Tokenizer initialized with ByteLevel pre-tokenizer")
            
            # Create BPE trainer
            # Use min_frequency=1 for small datasets to reach target vocab size
            # For larger datasets, min_frequency=2 is better
            min_freq = 1 if len(texts) < 5000 else 2
            trainer = BpeTrainer(
                vocab_size=self.vocab_size,
                min_frequency=min_freq,  # Lower for small datasets
                special_tokens=[
                    "<|endoftext|>",
                    "<|im_start|>",
                    "<|im_end|>",
                    "<pad>",
                    "<unk>",
                    "[SPACE]",
                    "[STOP]",
                    "[si]",
                ],
                show_progress=True
            )
            
            print(f"✅ BpeTrainer configured")
            print(f"   - Min frequency: {min_freq}")
            print(f"   - Special tokens: 5")
            
            # Train on Sinhala corpus
            print(f"\n🔧 Training BPE tokenizer on {len(texts)} Sinhala texts...")
            self.tokenizer.train_from_iterator(
                iterator=texts,
                trainer=trainer,
                length=len(texts)
            )
            
            # Ensure Sinhala language and XTTS control tokens exist
            mandatory_tokens = ["[SPACE]", "[STOP]", "[si]"]
            missing_tokens = [tok for tok in mandatory_tokens if self.tokenizer.token_to_id(tok) is None]
            if missing_tokens:
                print(f"   - Adding mandatory tokens: {missing_tokens}")
                self.tokenizer.add_special_tokens(missing_tokens)

            self.vocab = self.tokenizer.get_vocab()
            print(f"\n✅ Training completed!")
            print(f"   - Actual vocabulary size: {len(self.vocab):,} tokens")
            
            # Test tokenization
            self._test_tokenization(texts[:3])
            
        except Exception as e:
            raise ValueError(f"❌ Tokenizer training failed: {str(e)}")
    
    def _test_tokenization(self, texts):
        """Test tokenization on sample texts."""
        print(f"\n✅ Testing tokenization on samples:")
        
        unk_tokens_found = False
        for text in texts[:3]:
            encoding = self.tokenizer.encode(text)
            
            # Check for UNK tokens (typically ID 0 or 4)
            unk_count = encoding.ids.count(0) + encoding.ids.count(4)
            
            if unk_count > 0:
                unk_tokens_found = True
            
            status = "✅ No UNK" if unk_count == 0 else f"⚠️ {unk_count} UNK"
            print(f"   Text: {text[:50]:50} → {len(encoding.tokens):3} tokens [{status}]")
        
        if not unk_tokens_found:
            print(f"\n   ✅ EXCELLENT: No UNK tokens in test samples!")
        
        return unk_tokens_found
    
    def save_vocabulary(self, output_path):
        """
        Save tokenizer and vocabulary to files.
        
        CRITICAL: Saves directly to output_path, NOT a subdirectory
        This ensures train_gpt_xtts.py finds vocab.json at the expected location
        
        Args:
            output_path: Directory to save files
            
        Returns:
            Tuple of (vocab_file_path, tokenizer_file_path)
            
        Raises:
            ValueError: If saving fails
        """
        print(f"\n{'='*80}")
        print(f"SAVING VOCABULARY")
        print(f"{'='*80}")
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        print(f"\n✅ Output directory created: {output_path}")
        
        try:
            # Save vocabulary as JSON (CRITICAL: no subdirectory)
            vocab_file = os.path.join(output_path, "vocab.json")
            with open(vocab_file, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
            file_size = os.path.getsize(vocab_file) / 1024
            print(f"✅ Vocabulary saved: {vocab_file}")
            print(f"   - File size: {file_size:.1f} KB")
            print(f"   - Tokens: {len(self.vocab):,}")
            
            # Save tokenizer in native format
            tokenizer_file = os.path.join(output_path, "tokenizer.json")
            self.tokenizer.save(tokenizer_file)
            
            tokenizer_size = os.path.getsize(tokenizer_file) / 1024
            print(f"✅ Tokenizer saved: {tokenizer_file}")
            print(f"   - File size: {tokenizer_size:.1f} KB")

            merges_file = os.path.join(output_path, "merges.txt")
            merges = getattr(self.tokenizer.model, "get_merges", lambda: [])()
            with open(merges_file, "w", encoding="utf-8") as f:
                for pair in merges:
                    # Merges are tuples of (token_a, token_b)
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        f.write(f"{pair[0]} {pair[1]}\n")
            print(f"✅ Merges saved: {merges_file}")

            return vocab_file, tokenizer_file, merges_file
            
        except Exception as e:
            raise ValueError(f"❌ Error saving vocabulary: {str(e)}")


class ConfigUpdater:
    """Updates XTTS config.json for Sinhala language support."""
    
    @staticmethod
    def find_config_file(search_paths):
        """
        Find config.json in common locations.
        
        Args:
            search_paths: List of paths to search
            
        Returns:
            Path to config.json or None if not found
        """
        print(f"\n🔍 Searching for config.json...")
        
        for path in search_paths:
            if os.path.exists(path):
                print(f"✅ Found: {path}")
                return path
            else:
                print(f"   ℹ️ Not found: {path}")
        
        print(f"❌ config.json not found in any searched locations")
        return None
    
    @staticmethod
    def update_config_for_sinhala(config_path, language_code="si"):
        """
        Update config.json to support Sinhala language.
        
        Args:
            config_path: Path to config.json
            language_code: Language code (default: 'si' for Sinhala)
            
        Raises:
            FileNotFoundError: If config not found
            ValueError: If update fails
        """
        print(f"\n{'='*80}")
        print(f"UPDATING CONFIG.JSON FOR SINHALA")
        print(f"{'='*80}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config file not found: {config_path}")
        
        try:
            print(f"\n📖 Loading config from: {config_path}")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Config loaded successfully")
            
        except Exception as e:
            raise ValueError(f"❌ Error reading config: {str(e)}")
        
        # Add language_ids mapping
        if 'language_ids' not in config:
            config['language_ids'] = {}
        
        existing_ids = list(config['language_ids'].values()) if config['language_ids'] else []
        language_id = max(existing_ids) + 1 if existing_ids else 21
        
        config['language_ids'][language_code] = language_id
        print(f"\n✅ Added language_ids:")
        print(f"   - Language: {language_code}")
        print(f"   - ID: {language_id}")
        
        # Add language-specific settings
        # Handle both dict and list formats for languages
        if 'languages' not in config:
            config['languages'] = {}
        elif isinstance(config['languages'], list):
            config['languages'] = {item.get('language', str(idx)): item for idx, item in enumerate(config['languages']) if isinstance(item, dict)}

        config['languages'][language_code] = {
            'phoneme_language': None,  # Sinhala uses grapheme-based approach
            'use_phonemes': False,     # No phoneme conversion needed
            'name': 'Sinhala'
        }

        # Update tokenizer references to point to regenerated files
        tokenizer_dir = os.path.dirname(config_path)
        tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
        vocab_json = os.path.join(tokenizer_dir, "vocab.json")
        merges_txt = os.path.join(tokenizer_dir, "merges.txt")

        tokenizer_section = config.get('tokenizer')
        if isinstance(tokenizer_section, dict):
            for key in ("config_path", "path", "tokenizer_path", "tokenizer_file"):
                if key in tokenizer_section:
                    tokenizer_section[key] = tokenizer_json
            if 'vocab_path' in tokenizer_section:
                tokenizer_section['vocab_path'] = vocab_json
            if 'merges_path' in tokenizer_section:
                tokenizer_section['merges_path'] = merges_txt
        else:
            config['tokenizer'] = {
                'path': tokenizer_json,
                'vocab_path': vocab_json,
                'merges_path': merges_txt,
            }

        model_args = config.get('model_args')
        if isinstance(model_args, dict) and 'tokenizer_file' in model_args:
            model_args['tokenizer_file'] = tokenizer_json
        
        print(f"✅ Added language settings:")
        print(f"   - Phoneme language: None (grapheme-based)")
        print(f"   - Use phonemes: False")
        print(f"   - Language name: Sinhala")
        
        # Save updated config
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Config saved successfully: {config_path}")
            print(f"\n📊 Languages in config: {list(config.get('language_ids', {}).keys())}")
            
        except Exception as e:
            raise ValueError(f"❌ Error saving config: {str(e)}")


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(
        description="Extend XTTS-v2 vocabulary and config for Sinhala language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python extend_vocab_sinhala.py \\
    --metadata_path datasets/metadata_train.csv \\
    --output_path checkpoints/XTTS_v2.0_original_model_files/

  # With custom vocabulary size
  python extend_vocab_sinhala.py \\
    --metadata_path datasets/metadata_train.csv \\
    --output_path checkpoints/XTTS_v2.0_original_model_files/ \\
    --vocab_size 20000

  # With explicit config path
  python extend_vocab_sinhala.py \\
    --metadata_path datasets/metadata_train.csv \\
    --output_path checkpoints/XTTS_v2.0_original_model_files/ \\
    --xtts_config checkpoints/XTTS_v2.0_original_model_files/config.json
        """
    )
    
    parser.add_argument(
        "--metadata_path",
        required=True,
        help="Path to metadata_train.csv with Sinhala texts"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output directory for vocabulary files (e.g., checkpoints/XTTS_v2.0_original_model_files/)"
    )
    parser.add_argument(
        "--language",
        default="si",
        help="Language code (default: si for Sinhala)"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=15000,
        help="Vocabulary size for BPE tokenizer (default: 15000)"
    )
    parser.add_argument(
        "--xtts_config",
        default=None,
        help="Path to XTTS config.json (auto-detect if not provided)"
    )
    
    args = parser.parse_args()
    
    # Print header
    print(f"\n{'='*80}")
    print(f"SINHALA VOCABULARY EXTENSION FOR XTTS-v2")
    print(f"{'='*80}")
    print(f"\n📋 Configuration:")
    print(f"   - Language: {args.language} (Sinhala)")
    print(f"   - Vocabulary size: {args.vocab_size:,} tokens")
    print(f"   - Metadata: {args.metadata_path}")
    print(f"   - Output: {args.output_path}")
    
    try:
        # Step 1: Load Sinhala texts
        print(f"\n{'─'*80}")
        print(f"STEP 1: LOADING SINHALA TEXTS")
        print(f"{'─'*80}")
        
        tokenizer = SinhalaBPETokenizer(vocab_size=args.vocab_size)
        texts = tokenizer.load_sinhala_texts(args.metadata_path)
        
        # Step 2: Train BPE tokenizer
        print(f"\n{'─'*80}")
        print(f"STEP 2: TRAINING BPE TOKENIZER")
        print(f"{'─'*80}")
        
        tokenizer.train_tokenizer(texts)
        
        # Step 3: Save vocabulary (CRITICAL: no subdirectory)
        print(f"\n{'─'*80}")
        print(f"STEP 3: SAVING VOCABULARY")
        print(f"{'─'*80}")
        
        vocab_file, tokenizer_file, merges_file = tokenizer.save_vocabulary(args.output_path)
        
        # Step 4: Update config
        print(f"\n{'─'*80}")
        print(f"STEP 4: UPDATING CONFIG")
        print(f"{'─'*80}")
        
        config_path = args.xtts_config
        
        if not config_path:
            # Auto-detect config path
            search_paths = [
                os.path.join(args.output_path, "config.json"),
                os.path.join(os.path.dirname(args.output_path), "config.json"),
                os.path.join(args.output_path, "XTTS_v2.0_original_model_files", "config.json"),
            ]
            config_path = ConfigUpdater.find_config_file(search_paths)
        
        if config_path and os.path.exists(config_path):
            ConfigUpdater.update_config_for_sinhala(config_path, language_code=args.language)
        else:
            print(f"\n⚠️ WARNING: Config file not found")
            print(f"   You can manually add language settings to config.json:")
            print(f"""
   {{
       "language_ids": {{"si": 21}},
       "languages": {{
           "si": {{
               "phoneme_language": null,
               "use_phonemes": false,
               "name": "Sinhala"
           }}
       }}
   }}
            """)
        
        # Success summary
        print(f"\n{'='*80}")
        print(f"✅ VOCABULARY EXTENSION COMPLETE!")
        print(f"{'='*80}")
        
        print(f"\n📍 Output files created:")
        print(f"   ✅ {vocab_file}")
        print(f"   ✅ {tokenizer_file}")
        print(f"   ✅ {merges_file}")
        if config_path and os.path.exists(config_path):
            print(f"   ✅ {config_path} (updated)")
        
        print(f"\n🚀 Ready for GPT fine-tuning!")
        print(f"   Use command:")
        print(f"   CUDA_VISIBLE_DEVICES=0 python train_gpt_xtts.py \\")
        print(f"     --output_path {args.output_path} \\")
        print(f"     --metadatas <path_to_train>,<path_to_eval>,{args.language} \\")
        print(f"     --num_epochs 5 \\")
        print(f"     --batch_size 8")
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"\n{'='*80}")
        sys.exit(1)


if __name__ == "__main__":
    main()
