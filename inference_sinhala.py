#!/usr/bin/env python3
"""
Sinhala Text-to-Speech Inference Script for Fine-tuned XTTS-v2

This script loads a fine-tuned XTTS-v2 model and generates Sinhala speech
from Sinhala text using voice cloning from a reference audio file.

Features:
- Voice cloning from reference audio
- Batch inference support
- Audio quality control
- Comprehensive error handling
- Professional logging

Usage:
    python inference_sinhala.py \
        --checkpoint_path checkpoints/GPT_XTTS_FT-*/best_model.pth \
        --config_path checkpoints/GPT_XTTS_FT-*/config.json \
        --vocab_path checkpoints/XTTS_v2.0_original_model_files/vocab.json \
        --text "නිරන්තරයි ඉතා වැදගත්" \
        --reference_audio reference_speaker.wav \
        --output_path output_speech.wav
"""

import os
import sys
import argparse
import torch
import torchaudio
from pathlib import Path
from typing import Optional, Union, List
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
except ImportError as e:
    print(f"❌ Error: TTS library not found")
    print(f"   Install with: pip install TTS")
    sys.exit(1)


class SinhalaTTSInference:
    """
    Sinhala Text-to-Speech Inference using Fine-tuned XTTS-v2
    
    Attributes:
        model: Loaded XTTS-v2 model
        device: GPU or CPU device
        config: XTTS configuration
        language: Language code (always 'si' for Sinhala)
    """
    
    LANGUAGE_CODE = "si"  # ISO 639-1 code for Sinhala
    
    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        vocab_path: str,
        device: Optional[str] = None,
        use_deepspeed: bool = False
    ):
        """
        Initialize the Sinhala TTS model.
        
        Args:
            checkpoint_path: Path to the fine-tuned model checkpoint
            config_path: Path to the model configuration JSON
            vocab_path: Path to the vocabulary/tokenizer file
            device: Device to load model on ('cuda' or 'cpu'). Auto-detects if None.
            use_deepspeed: Whether to use DeepSpeed optimization
            
        Raises:
            FileNotFoundError: If required files are not found
            RuntimeError: If model initialization fails
            
        Example:
            >>> tts = SinhalaTTSInference(
            ...     checkpoint_path="checkpoints/GPT_XTTS_FT-*/best_model.pth",
            ...     config_path="checkpoints/GPT_XTTS_FT-*/config.json",
            ...     vocab_path="checkpoints/XTTS_v2.0_original_model_files/vocab.json"
            ... )
        """
        
        print("\n" + "=" * 80)
        print("SINHALA XTTS-v2 INFERENCE - MODEL LOADING")
        print("=" * 80)
        
        # Set device
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"\n📍 Device: {self.device}")
        
        # Verify files exist
        self._verify_files(checkpoint_path, config_path, vocab_path)
        
        # Load configuration
        print(f"\n📖 Loading configuration from: {config_path}")
        self.config = XttsConfig()
        self.config.load_json(config_path)
        print(f"✅ Configuration loaded")
        
        # Initialize model
        print(f"\n🔧 Initializing XTTS-v2 model...")
        try:
            self.model = Xtts.init_from_config(self.config)
            print(f"✅ Model initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
        
        # Load checkpoint
        print(f"\n📥 Loading checkpoint from: {checkpoint_path}")
        try:
            self.model.load_checkpoint(
                self.config,
                checkpoint_path=checkpoint_path,
                vocab_path=vocab_path,
                use_deepspeed=use_deepspeed
            )
            print(f"✅ Checkpoint loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
        
        # Move to device
        self.model.to(self.device)
        print(f"✅ Model moved to {self.device}")
        
        print("\n" + "=" * 80)
        print("✅ MODEL READY FOR INFERENCE")
        print("=" * 80 + "\n")
    
    @staticmethod
    def _verify_files(*file_paths):
        """
        Verify that all required files exist.
        
        Args:
            *file_paths: Paths to verify
            
        Raises:
            FileNotFoundError: If any file is missing
        """
        for file_path in file_paths:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ File not found: {file_path}")
    
    def get_conditioning_latents(
        self,
        reference_audio_path: str
    ) -> tuple:
        """
        Extract voice characteristics from reference audio.
        
        Args:
            reference_audio_path: Path to reference audio file for voice cloning
            
        Returns:
            tuple: (gpt_cond_latent, speaker_embedding)
            
        Raises:
            FileNotFoundError: If reference audio not found
            RuntimeError: If latent extraction fails
        """
        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(f"❌ Reference audio not found: {reference_audio_path}")
        
        try:
            print(f"🎤 Extracting voice characteristics from: {os.path.basename(reference_audio_path)}")
            
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=reference_audio_path,
                gpt_cond_len=self.model.config.gpt_cond_len,
                max_ref_length=self.model.config.max_ref_len,
                sound_norm_refs=self.model.config.sound_norm_refs,
            )
            
            print(f"✅ Voice characteristics extracted successfully")
            return gpt_cond_latent, speaker_embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract conditioning latents: {str(e)}")
    
    def synthesize(
        self,
        text: str,
        gpt_cond_latent,
        speaker_embedding,
        temperature: float = 0.1,
        length_penalty: float = 1.0,
        repetition_penalty: float = 10.0,
        top_k: int = 10,
        top_p: float = 0.3
    ) -> torch.Tensor:
        """
        Synthesize Sinhala speech from text.
        
        Args:
            text: Sinhala text to synthesize
            gpt_cond_latent: Voice conditioning from reference audio
            speaker_embedding: Speaker embedding from reference audio
            temperature: Controls randomness (lower = more deterministic)
            length_penalty: Penalty for long sequences
            repetition_penalty: Penalty for repetitive tokens
            top_k: Number of top tokens to consider
            top_p: Cumulative probability threshold
            
        Returns:
            torch.Tensor: Generated audio waveform (1, num_samples)
            
        Raises:
            RuntimeError: If synthesis fails
            
        Example:
            >>> wav = tts.synthesize(
            ...     text="නිරන්තරයි ඉතා වැදගත්",
            ...     gpt_cond_latent=latent,
            ...     speaker_embedding=embedding
            ... )
        """
        
        print(f"\n📝 Sinhala text: {text}")
        print(f"🎵 Generating speech...")
        
        try:
            wav_output = self.model.inference(
                text=text,
                language=self.LANGUAGE_CODE,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Convert to tensor if needed
            if isinstance(wav_output, dict) and "wav" in wav_output:
                wav_tensor = torch.tensor(wav_output["wav"]).unsqueeze(0)
            else:
                wav_tensor = torch.tensor(wav_output).unsqueeze(0)
            
            duration_sec = wav_tensor.shape[1] / 24000
            print(f"✅ Speech generated successfully!")
            print(f"   - Duration: {duration_sec:.2f} seconds")
            print(f"   - Audio shape: {wav_tensor.shape}")
            
            return wav_tensor
            
        except Exception as e:
            raise RuntimeError(f"Speech synthesis failed: {str(e)}")
    
    def save_audio(
        self,
        wav_tensor: torch.Tensor,
        output_path: str,
        sample_rate: int = 24000
    ) -> str:
        """
        Save generated audio to a WAV file.
        
        Args:
            wav_tensor: Audio waveform tensor
            output_path: Path where to save the WAV file
            sample_rate: Sample rate of the audio (default: 24000 Hz for XTTS)
            
        Returns:
            str: Path to saved audio file
            
        Raises:
            RuntimeError: If saving fails
            
        Example:
            >>> output_file = tts.save_audio(wav, "output.wav")
        """
        
        print(f"\n💾 Saving audio to: {output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Ensure tensor is on CPU
        if wav_tensor.device.type != 'cpu':
            wav_tensor = wav_tensor.cpu()
        
        try:
            torchaudio.save(output_path, wav_tensor, sample_rate)
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"✅ Audio saved successfully!")
            print(f"   - File: {os.path.basename(output_path)}")
            print(f"   - Size: {file_size:.2f} MB")
            return output_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to save audio: {str(e)}")
    
    def synthesize_and_save(
        self,
        text: str,
        reference_audio_path: str,
        output_path: str,
        **synthesis_kwargs
    ) -> str:
        """
        Synthesize speech and save to file in one call.
        
        Args:
            text: Sinhala text to synthesize
            reference_audio_path: Path to reference audio for voice cloning
            output_path: Path to save output WAV file
            **synthesis_kwargs: Additional arguments for synthesize()
            
        Returns:
            str: Path to saved audio file
            
        Raises:
            FileNotFoundError: If required files not found
            RuntimeError: If synthesis or saving fails
            
        Example:
            >>> output_file = tts.synthesize_and_save(
            ...     text="නිරන්තරයි ඉතා වැදගත්",
            ...     reference_audio_path="reference.wav",
            ...     output_path="output.wav"
            ... )
        """
        
        # Extract voice characteristics
        gpt_cond_latent, speaker_embedding = self.get_conditioning_latents(
            reference_audio_path
        )
        
        # Synthesize speech
        wav = self.synthesize(
            text,
            gpt_cond_latent,
            speaker_embedding,
            **synthesis_kwargs
        )
        
        # Save audio
        return self.save_audio(wav, output_path)
    
    def batch_synthesize_and_save(
        self,
        texts: List[str],
        reference_audio_path: str,
        output_dir: str,
        prefix: str = "sinhala",
        **synthesis_kwargs
    ) -> List[str]:
        """
        Synthesize multiple texts and save them.
        
        Args:
            texts: List of Sinhala texts to synthesize
            reference_audio_path: Path to reference audio
            output_dir: Directory to save output files
            prefix: Prefix for output files
            **synthesis_kwargs: Additional arguments for synthesize()
            
        Returns:
            list: Paths to saved audio files
            
        Example:
            >>> texts = ["නිරන්තරයි", "ශ්‍රී ලංකා"]
            >>> files = tts.batch_synthesize_and_save(
            ...     texts=texts,
            ...     reference_audio_path="reference.wav",
            ...     output_dir="output_audio"
            ... )
        """
        
        print(f"\n" + "=" * 80)
        print(f"BATCH SYNTHESIS ({len(texts)} texts)")
        print(f"=" * 80)
        
        # Extract voice characteristics once
        gpt_cond_latent, speaker_embedding = self.get_conditioning_latents(
            reference_audio_path
        )
        
        output_files = []
        
        for idx, text in enumerate(texts, 1):
            print(f"\n[{idx}/{len(texts)}] Processing: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            try:
                # Synthesize
                wav = self.synthesize(
                    text,
                    gpt_cond_latent,
                    speaker_embedding,
                    **synthesis_kwargs
                )
                
                # Save
                output_path = os.path.join(output_dir, f"{prefix}_{idx:03d}.wav")
                self.save_audio(wav, output_path)
                
                output_files.append(output_path)
                
            except Exception as e:
                print(f"   ⚠️ Error processing text {idx}: {str(e)}")
                continue
        
        print(f"\n" + "=" * 80)
        print(f"✅ BATCH SYNTHESIS COMPLETE")
        print(f"   - Processed: {len(output_files)}/{len(texts)}")
        print(f"   - Output directory: {output_dir}")
        print(f"=" * 80 + "\n")
        
        return output_files


def create_parser():
    """Create argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Sinhala Text-to-Speech Inference using Fine-tuned XTTS-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single text synthesis
  python inference_sinhala.py \\
    --checkpoint_path checkpoints/GPT_XTTS_FT-*/best_model.pth \\
    --config_path checkpoints/GPT_XTTS_FT-*/config.json \\
    --vocab_path checkpoints/XTTS_v2.0_original_model_files/vocab.json \\
    --text "නිරන්තරයි ඉතා වැදගත්" \\
    --reference_audio reference.wav \\
    --output_path output.wav

  # Batch synthesis
  python inference_sinhala.py \\
    --checkpoint_path checkpoints/GPT_XTTS_FT-*/best_model.pth \\
    --config_path checkpoints/GPT_XTTS_FT-*/config.json \\
    --vocab_path checkpoints/XTTS_v2.0_original_model_files/vocab.json \\
    --texts_file texts.txt \\
    --reference_audio reference.wav \\
    --output_dir output_audio
        """
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to fine-tuned model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to model configuration (config.json)"
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to vocabulary file (vocab.json)"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Sinhala text to synthesize"
    )
    parser.add_argument(
        "--texts_file",
        type=str,
        default=None,
        help="File with Sinhala texts (one per line) for batch synthesis"
    )
    parser.add_argument(
        "--reference_audio",
        type=str,
        required=True,
        help="Path to reference audio file for voice cloning"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_speech.wav",
        help="Output path for single synthesis (default: output_speech.wav)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_audio",
        help="Output directory for batch synthesis (default: output_audio)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu'). Auto-detects if not specified."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (default: 0.1, lower = more deterministic)"
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Length penalty (default: 1.0)"
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=10.0,
        help="Repetition penalty (default: 10.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K sampling (default: 10)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.3,
        help="Top-P (nucleus) sampling (default: 0.3)"
    )
    
    return parser


def main():
    """Main entry point for inference."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and not args.texts_file:
        print("❌ Error: Either --text or --texts_file must be provided")
        sys.exit(1)
    
    if args.text and args.texts_file:
        print("⚠️ Warning: Both --text and --texts_file provided, using --text")
        args.texts_file = None
    
    try:
        # Initialize model
        tts = SinhalaTTSInference(
            checkpoint_path=args.checkpoint_path,
            config_path=args.config_path,
            vocab_path=args.vocab_path,
            device=args.device
        )
        
        # Prepare synthesis parameters
        synthesis_kwargs = {
            "temperature": args.temperature,
            "length_penalty": args.length_penalty,
            "repetition_penalty": args.repetition_penalty,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
        
        # Synthesis mode
        if args.text:
            # Single text synthesis
            print("\n" + "=" * 80)
            print("SINGLE TEXT SYNTHESIS")
            print("=" * 80)
            
            output_file = tts.synthesize_and_save(
                text=args.text,
                reference_audio_path=args.reference_audio,
                output_path=args.output_path,
                **synthesis_kwargs
            )
            
            print(f"\n✅ Synthesis complete!")
            print(f"   Output: {output_file}")
            
        elif args.texts_file:
            # Batch synthesis
            print(f"\n📄 Reading texts from: {args.texts_file}")
            
            with open(args.texts_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"✅ Loaded {len(texts)} texts for synthesis")
            
            output_files = tts.batch_synthesize_and_save(
                texts=texts,
                reference_audio_path=args.reference_audio,
                output_dir=args.output_dir,
                **synthesis_kwargs
            )
            
            print(f"✅ Batch synthesis complete!")
            print(f"   Output directory: {args.output_dir}")
            print(f"   Generated files: {len(output_files)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
