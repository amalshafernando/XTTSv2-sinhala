"""
Prepare Sinhala dataset from Kaggle format to XTTS-v2 format.
This script converts the Kaggle dataset structure to the format expected by XTTS-v2.
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def copy_audio_files(source_dir, dest_dir, audio_files):
    """
    Copy audio files from source to destination directory.
    
    Args:
        source_dir: Source directory containing audio files
        dest_dir: Destination directory (wavs/)
        audio_files: List of audio file paths relative to source_dir
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    copied = 0
    skipped = 0
    errors = []
    
    print(f"\n[1/3] Copying audio files from {source_dir} to {dest_dir}")
    
    for audio_file in tqdm(audio_files, desc="Copying audio files"):
        source_path = os.path.join(source_dir, audio_file)
        # Get just the filename
        filename = os.path.basename(audio_file)
        dest_path = os.path.join(dest_dir, filename)
        
        # Update path in metadata to use wavs/ directory
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                copied += 1
            except Exception as e:
                errors.append(f"Error copying {source_path}: {str(e)}")
                skipped += 1
        else:
            errors.append(f"File not found: {source_path}")
            skipped += 1
    
    print(f"    ✓ Copied {copied} files")
    if skipped > 0:
        print(f"    ⚠ Skipped {skipped} files")
    if errors:
        print(f"    ⚠ {len(errors)} errors occurred")
        if len(errors) <= 10:
            for error in errors:
                print(f"      - {error}")
        else:
            print(f"      (Showing first 10 errors)")
            for error in errors[:10]:
                print(f"      - {error}")
    
    return copied, skipped, errors


def convert_metadata(kaggle_path, output_path):
    """
    Convert Kaggle metadata CSV files to XTTS-v2 format.
    
    Args:
        kaggle_path: Path to Kaggle dataset directory
        output_path: Path to output directory
    """
    print("\n" + "=" * 60)
    print("CONVERTING DATASET TO XTTS-v2 FORMAT")
    print("=" * 60)
    
    # Paths
    metadata_train_path = os.path.join(kaggle_path, "metadata_train.csv")
    metadata_eval_path = os.path.join(kaggle_path, "metadata_eval.csv")
    
    # Check if metadata files exist
    if not os.path.exists(metadata_train_path):
        raise FileNotFoundError(f"Training metadata not found: {metadata_train_path}")
    if not os.path.exists(metadata_eval_path):
        raise FileNotFoundError(f"Evaluation metadata not found: {metadata_eval_path}")
    
    print(f"\n[2/3] Reading metadata files")
    print(f"    Train: {metadata_train_path}")
    print(f"    Eval: {metadata_eval_path}")
    
    # Try to read CSV files - first attempt with default separator (comma)
    # If that results in a single column with pipe-separated values, we'll parse it
    try:
        train_df = pd.read_csv(metadata_train_path)
        eval_df = pd.read_csv(metadata_eval_path)
    except Exception as e:
        print(f"    ⚠ Error reading CSV with default separator: {e}")
        # Try with pipe separator
        try:
            train_df = pd.read_csv(metadata_train_path, sep='|')
            eval_df = pd.read_csv(metadata_eval_path, sep='|')
            print(f"    ✓ Successfully read with pipe separator")
        except Exception as e2:
            raise ValueError(f"Failed to read CSV files: {e2}")
    
    print(f"    Train samples: {len(train_df)}")
    print(f"    Eval samples: {len(eval_df)}")
    
    # Check if CSV has pipe-separated format (single column with pipe-separated values)
    # This handles CSVs with format: "audio_file_path|transcript|speaker_id"
    def parse_pipe_separated_csv(df, df_name):
        """Parse pipe-separated CSV format into separate columns."""
        # Check if we have a single column that looks like pipe-separated format
        if len(df.columns) == 1:
            col_name = df.columns[0]
            # Check if column name or first row contains pipe separator
            has_pipe_in_header = '|' in col_name
            has_pipe_in_data = len(df) > 0 and '|' in str(df.iloc[0, 0])
            
            if has_pipe_in_header or has_pipe_in_data:
                print(f"    Detected pipe-separated format in {df_name} CSV")
                # Split the single column into multiple columns
                # First, check if header has pipe separator
                if has_pipe_in_header:
                    # Column name is the header: "audio_file_path|transcript|speaker_id"
                    header_parts = [part.strip() for part in col_name.split('|')]
                    if len(header_parts) == 3:
                        # Split each row by pipe
                        df[header_parts] = df[col_name].str.split('|', expand=True, n=2)
                        df = df.drop(columns=[col_name])
                    else:
                        raise ValueError(f"Cannot parse {df_name} CSV header: expected 3 columns, got {len(header_parts)}")
                else:
                    # No header with pipes, split data rows
                    # Try to infer column names from first row or use defaults
                    df_split = df[col_name].str.split('|', expand=True)
                    if len(df_split.columns) == 3:
                        df_split.columns = ['audio_file_path', 'transcript', 'speaker_id']
                        df = df_split
                    else:
                        raise ValueError(f"Cannot parse {df_name} CSV: expected 3 pipe-separated columns, got {len(df_split.columns)}")
                
                # Clean up any extra whitespace
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.strip()
        
        return df
    
    # Parse pipe-separated format if needed
    train_df = parse_pipe_separated_csv(train_df, "train")
    eval_df = parse_pipe_separated_csv(eval_df, "eval")
    
    # Verify required columns exist
    required_columns = {"audio_file_path", "transcript", "speaker_id"}
    for df_name, df in [("train", train_df), ("eval", eval_df)]:
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"{df_name} metadata missing required columns: {missing_cols}. Found columns: {list(df.columns)}")
    
    # Convert column names and format
    def convert_dataframe(df, split_name):
        """Convert a dataframe to XTTS format."""
        print(f"\n[3/3] Converting {split_name} metadata")
        
        # Rename columns
        df_xtts = df.copy()
        df_xtts = df_xtts.rename(columns={
            "audio_file_path": "audio_file",
            "transcript": "text",
            "speaker_id": "speaker_name"
        })
        
        # Update path format: change 'wav/' to 'wavs/'
        df_xtts["audio_file"] = df_xtts["audio_file"].str.replace("wav/", "wavs/", regex=False)
        df_xtts["audio_file"] = df_xtts["audio_file"].str.replace("\\", "/", regex=False)  # Normalize path separators
        
        # Remove any rows with missing values
        initial_count = len(df_xtts)
        df_xtts = df_xtts.dropna(subset=["audio_file", "text", "speaker_name"])
        if len(df_xtts) < initial_count:
            print(f"    ⚠ Removed {initial_count - len(df_xtts)} rows with missing values")
        
        # Ensure audio_file paths don't start with '/'
        df_xtts["audio_file"] = df_xtts["audio_file"].str.lstrip("/")
        
        # Extract audio files for copying
        audio_files = df_xtts["audio_file"].tolist()
        
        # Create output directory
        wavs_dir = os.path.join(output_path, "wavs")
        
        # Copy audio files
        # Try to find source audio directory
        source_audio_dir = None
        possible_dirs = [
            os.path.join(kaggle_path, "wavs"),  # Check wavs first (most common)
            os.path.join(kaggle_path, "wav"),
            os.path.join(kaggle_path, "audio"),
            os.path.join(kaggle_path, "audio_files"),
            kaggle_path  # Audio files might be in root
        ]
        
        for possible_dir in possible_dirs:
            if os.path.exists(possible_dir):
                # Check if it contains audio files
                if any(os.path.isfile(os.path.join(possible_dir, f)) for f in os.listdir(possible_dir) if f.endswith(('.wav', '.mp3', '.flac'))):
                    source_audio_dir = possible_dir
                    break
        
        if source_audio_dir:
            # Get unique audio files (basenames for copying)
            # The metadata may have paths like "wavs/filename.wav" but we need to find the source file
            unique_filenames = set()
            source_files_to_copy = []
            
            for audio_file in audio_files:
                filename = os.path.basename(audio_file)
                if filename not in unique_filenames:
                    unique_filenames.add(filename)
                    # Try to find the source file
                    source_path = os.path.join(source_audio_dir, filename)
                    if os.path.exists(source_path):
                        source_files_to_copy.append(filename)
                    else:
                        # Try with wav/ prefix removed
                        alt_source = os.path.join(kaggle_path, filename)
                        if os.path.exists(alt_source):
                            source_files_to_copy.append(filename)
            
            if source_files_to_copy:
                copy_audio_files(source_audio_dir, wavs_dir, source_files_to_copy)
            else:
                print(f"    ⚠ Warning: Could not locate source audio files. Proceeding with metadata only.")
        else:
            print(f"    ⚠ Warning: Could not find audio directory. Expected: {os.path.join(kaggle_path, 'wav')}")
            print(f"    Will proceed with metadata conversion only.")
        
        # Save in XTTS format: pipe-separated values without headers
        output_file = os.path.join(output_path, f"metadata_{split_name}.csv")
        df_xtts[["audio_file", "text", "speaker_name"]].to_csv(
            output_file,
            sep="|",
            index=False,
            header=False,
            encoding='utf-8'
        )
        
        print(f"    ✓ Saved {len(df_xtts)} samples to {output_file}")
        print(f"    Format: audio_file|text|speaker_name")
        
        return df_xtts, output_file
    
    # Convert both splits
    train_xtts, train_output = convert_dataframe(train_df, "train")
    eval_xtts, eval_output = convert_dataframe(eval_df, "eval")
    
    # Verify output files
    print(f"\n[Verification] Checking output files")
    for output_file in [train_output, eval_output]:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                parts = first_line.split("|")
                if len(parts) == 3:
                    print(f"    ✓ {os.path.basename(output_file)}: Format OK")
                    print(f"      Sample: {first_line[:80]}...")
                else:
                    print(f"    ⚠ {os.path.basename(output_file)}: Unexpected format ({len(parts)} parts)")
        else:
            print(f"    ❌ {os.path.basename(output_file)}: File not found")
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nOutput directory: {output_path}")
    print(f"  - {os.path.basename(train_output)} ({len(train_xtts)} samples)")
    print(f"  - {os.path.basename(eval_output)} ({len(eval_xtts)} samples)")
    print(f"  - wavs/ (audio files)")
    
    return train_output, eval_output


def main():
    parser = argparse.ArgumentParser(
        description="Convert Kaggle Sinhala dataset to XTTS-v2 format"
    )
    
    parser.add_argument(
        "--kaggle_path",
        type=str,
        required=True,
        help="Path to Kaggle dataset directory (contains metadata_train.csv, metadata_eval.csv, wav/)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output directory where XTTS-format dataset will be saved"
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.kaggle_path):
        print(f"❌ ERROR: Kaggle dataset path does not exist: {args.kaggle_path}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    try:
        convert_metadata(args.kaggle_path, args.output_path)
        return 0
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

