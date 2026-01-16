#!/usr/bin/env python3
"""
Local TTS Pipeline using Pocket TTS (Kyutai)

Converts text files to audio using the pocket-tts model.
Runs entirely on CPU without external API dependencies.

Usage:
    python test.py input.txt output_dir/
    python test.py input_dir/ output_dir/

Requirements:
    pip install pocket-tts
"""

import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging
import json
import hashlib
from datetime import datetime

import scipy.io.wavfile
from pocket_tts import TTSModel

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Configure logging output."""
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(str(log_file)))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Built-in voices that work without HuggingFace authentication
AVAILABLE_VOICES = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma']


@dataclass
class TTSConfig:
    """TTS processing parameters for pocket-tts."""
    voice: str = "alba"
    chunk_size: int = 500
    retries: int = 3

    def __post_init__(self):
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if self.retries < 1:
            raise ValueError("retries must be at least 1")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def setup_output_structure(output_base: Path) -> tuple[Path, Path, Path]:
    """Create output directory structure: Full_Audio, Single_Files, Zips."""
    full_audio_dir = output_base / "Full_Audio"
    single_files_dir = output_base / "Single_Files"
    zips_dir = single_files_dir / "Zips"
    for d in [full_audio_dir, single_files_dir, zips_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return full_audio_dir, single_files_dir, zips_dir


def create_run_log(full_audio_dir: Path, run_info: dict) -> Path:
    """Save run metadata to a timestamped JSON log."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = full_audio_dir / f"run_{timestamp}.json"
    with open(log_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    return log_file


def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into word-based chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# ─────────────────────────────────────────────────────────────────────────────
# TTS Model Management
# ─────────────────────────────────────────────────────────────────────────────
class TTSEngine:
    """
    Wraps pocket-tts model loading and inference.
    
    Keeps model and voice state in memory for efficiency since both
    load_model() and get_state_for_audio_prompt() are slow operations.
    """
    
    def __init__(self, voice: str):
        self.model = TTSModel.load_model()
        self.voice_state = self.model.get_state_for_audio_prompt(voice)
        self.sample_rate = self.model.sample_rate
    
    def generate(self, text: str) -> "torch.Tensor":
        """Generate audio tensor from text."""
        return self.model.generate_audio(self.voice_state, text)


# ─────────────────────────────────────────────────────────────────────────────
# Audio Processing
# ─────────────────────────────────────────────────────────────────────────────
def save_chunk(
    engine: TTSEngine,
    text_chunk: str,
    file_path: Path,
    index: int,
    total: int,
    config: TTSConfig
) -> Optional[Exception]:
    """
    Convert a single text chunk to audio with retry logic.
    
    Returns None on success, or the final Exception on failure.
    """
    print(f"Starting chunk {index + 1}/{total} -> {file_path.name}")
    
    for attempt in range(config.retries):
        try:
            audio = engine.generate(text_chunk)
            scipy.io.wavfile.write(str(file_path), engine.sample_rate, audio.numpy())
            print(f"Finished chunk {index + 1}/{total} -> {file_path.name}")
            return None
        except Exception as e:
            logger.error(f"Chunk {index + 1} failed attempt {attempt + 1}: {type(e).__name__}: {str(e)}")
            print(f"--- WARNING: Chunk {index + 1} failed attempt {attempt + 1} ---")
            if attempt >= config.retries - 1:
                return e
    
    return None


def process_text_to_audio_chunks(
    engine: TTSEngine,
    text_chunks: list[str],
    input_file: Path,
    config: TTSConfig,
    output_dir: Path
) -> tuple[list[int], Path, Path]:
    """Process all text chunks to audio files sequentially."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "processing.log"
    setup_logging(log_file)

    failed_chunks = []
    for i, chunk in enumerate(text_chunks):
        file_path = output_dir / f"{i}.wav"
        error = save_chunk(engine, chunk, file_path, i, len(text_chunks), config)
        if error is not None:
            failed_chunks.append(i)
            logger.error(f"Final result for chunk {i+1}: {type(error).__name__}: {str(error)}")

    failed_json = output_dir / "failed_chunks.json"
    with open(failed_json, 'w') as f:
        json.dump({
            "input_file": str(input_file),
            "input_hash": get_file_hash(input_file),
            "failed_chunks": failed_chunks
        }, f)

    return failed_chunks, log_file, failed_json


def concatenate_audio_files(input_dir: Path, output_file: Path) -> bool:
    """Concatenate WAV chunks into a single audio file using ffmpeg."""
    files = sorted(input_dir.glob("*.wav"), key=lambda f: int(f.stem))
    if not files:
        print("No audio files to concatenate.")
        return False

    # Single file: just copy it
    if len(files) == 1:
        shutil.copy(files[0], output_file)
        print(f"Copied single chunk to {output_file}")
        return True

    try:
        import ffmpeg
    except ImportError:
        print("WARNING: ffmpeg-python not installed. Skipping concatenation.")
        print("Run: pip install ffmpeg-python")
        return False

    try:
        inputs = [ffmpeg.input(str(f)) for f in files]
        ffmpeg.concat(*inputs, v=0, a=1).output(str(output_file)).run(quiet=True, overwrite_output=True)
        print(f"Concatenated to {output_file}")
        return True
    except FileNotFoundError:
        print("WARNING: ffmpeg binary not found. Skipping concatenation.")
        print("Install ffmpeg system package to enable audio concatenation.")
        return False
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error during concatenation: {e.stderr.decode() if e.stderr else str(e)}")
        print(f"ERROR: Failed to concatenate audio files: {e}")
        return False


def create_zip(source_dir: Path, zip_path: Path) -> bool:
    """Create a zip archive of audio chunks."""
    try:
        zip_base = zip_path.with_suffix('')
        shutil.make_archive(str(zip_base), 'zip', source_dir)
        print(f"Created zip: {zip_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create zip: {e}")
        print(f"ERROR: Failed to create zip archive: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main Processing Functions
# ─────────────────────────────────────────────────────────────────────────────
def process_single_text_file(
    input_file: Path,
    output_base: Path,
    config: Optional[TTSConfig] = None,
    engine: Optional[TTSEngine] = None
) -> dict:
    """Process a single text file to audio."""
    if config is None:
        config = TTSConfig()
    
    # Reuse engine if provided, otherwise create one
    if engine is None:
        engine = TTSEngine(config.voice)

    full_audio_dir, single_files_dir, zips_dir = setup_output_structure(output_base)
    stem = input_file.stem
    single_dir = single_files_dir / stem
    single_dir.mkdir(exist_ok=True)

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text_chunks = chunk_text(text, config.chunk_size)
    failed_chunks, log_file, failed_json = process_text_to_audio_chunks(
        engine, text_chunks, input_file, config, single_dir
    )

    success = len(failed_chunks) == 0
    full_file = None
    zip_file = None

    if success:
        full_file = full_audio_dir / f"{stem}.wav"
        concatenate_audio_files(single_dir, full_file)
        zip_file = zips_dir / f"{stem}.zip"
        create_zip(single_dir, zip_file)

    return {
        "input_file": str(input_file),
        "input_hash": get_file_hash(input_file),
        "output_base": str(output_base),
        "success": success,
        "failed_chunks": failed_chunks,
        "log_file": str(log_file),
        "failed_json": str(failed_json),
        "full_file": str(full_file) if full_file else None,
        "zip_file": str(zip_file) if zip_file else None
    }


def process_directory(
    input_dir: Path,
    output_base: Path,
    config: Optional[TTSConfig] = None
) -> tuple[dict, Path]:
    """Process all .txt files in a directory."""
    if config is None:
        config = TTSConfig()

    full_audio_dir, single_files_dir, zips_dir = setup_output_structure(output_base)
    txt_files = list(input_dir.glob("*.txt"))

    # Load model once, reuse for all files
    engine = TTSEngine(config.voice)

    run_info = {
        "run_timestamp": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "output_base": str(output_base),
        "config": asdict(config),
        "files_processed": []
    }

    for txt_file in txt_files:
        result = process_single_text_file(txt_file, output_base, config, engine)
        run_info["files_processed"].append(result)

    run_log = create_run_log(full_audio_dir, run_info)
    return run_info, run_log


def retry_failed_chunks_from_json(
    failed_json: Path,
    config: Optional[TTSConfig] = None
) -> list[int]:
    """Retry processing failed chunks from a failed_chunks.json file."""
    if config is None:
        config = TTSConfig()

    with open(failed_json, 'r') as f:
        data = json.load(f)

    input_file = Path(data["input_file"])
    failed = data["failed_chunks"]

    if not failed:
        print("No failed chunks to retry.")
        return []

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text_chunks = chunk_text(text, config.chunk_size)
    output_dir = failed_json.parent

    engine = TTSEngine(config.voice)
    
    new_failed = []
    for i in failed:
        if i >= len(text_chunks):
            print(f"Invalid chunk index {i}")
            new_failed.append(i)
            continue
        chunk = text_chunks[i]
        file_path = output_dir / f"{i}.wav"
        error = save_chunk(engine, chunk, file_path, i, len(text_chunks), config)
        if error is not None:
            new_failed.append(i)

    with open(failed_json, 'w') as f:
        json.dump({
            "input_file": str(input_file),
            "input_hash": data["input_hash"],
            "failed_chunks": new_failed
        }, f)

    if not new_failed:
        full_audio_dir = output_dir.parent.parent / "Full_Audio"
        full_file = full_audio_dir / f"{input_file.stem}.wav"
        concatenate_audio_files(output_dir, full_file)
        zips_dir = output_dir.parent / "Zips"
        zip_file = zips_dir / f"{input_file.stem}.zip"
        create_zip(output_dir, zip_file)
        print("All chunks succeeded, created full audio and zip.")

    return new_failed


def retry_failures(output_base: Path, config: Optional[TTSConfig] = None) -> None:
    """Find and retry all failed chunks in an output directory."""
    if config is None:
        config = TTSConfig()

    single_files_dir = output_base / "Single_Files"
    failed_jsons = list(single_files_dir.rglob("failed_chunks.json"))

    if not failed_jsons:
        print("No failed chunks found!")
        return

    print(f"Found {len(failed_jsons)} file(s) with failures. Retrying...")

    for failed_json in failed_jsons:
        print(f"\nRetrying: {failed_json.parent.name}")
        remaining = retry_failed_chunks_from_json(failed_json, config)
        if remaining:
            print(f"  Still have {len(remaining)} failed chunks")
        else:
            print(f"  ✓ All chunks succeeded!")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience API
# ─────────────────────────────────────────────────────────────────────────────
def process_file(
    input_file: str,
    output_dir: str,
    voice: str = "alba",
    chunk_size: int = 500
) -> dict:
    """
    Process a single text file to audio using pocket-tts.

    Args:
        input_file: Path to input .txt file
        output_dir: Output directory path
        voice: Voice name from AVAILABLE_VOICES (default: alba)
        chunk_size: Words per chunk (default: 500)

    Returns:
        Dict with processing results

    Example:
        >>> result = process_file("book.txt", "./output")
        >>> print(result["full_file"])
    """
    config = TTSConfig(
        voice=voice,
        chunk_size=chunk_size
    )
    return process_single_text_file(Path(input_file), Path(output_dir), config)


def process_dir(
    input_dir: str,
    output_dir: str,
    voice: str = "alba",
    chunk_size: int = 500
) -> tuple[dict, Path]:
    """
    Process all .txt files in a directory.

    Args:
        input_dir: Directory containing .txt files
        output_dir: Output directory path
        voice: Voice name from AVAILABLE_VOICES (default: alba)
        chunk_size: Words per chunk (default: 500)

    Returns:
        Tuple of (run_info dict, run_log path)

    Example:
        >>> info, log = process_dir("./texts", "./output")
    """
    config = TTSConfig(
        voice=voice,
        chunk_size=chunk_size
    )
    return process_directory(Path(input_dir), Path(output_dir), config)


def retry(output_dir: str) -> None:
    """
    Retry all failed chunks in an output directory.

    Args:
        output_dir: Output directory containing Single_Files/

    Example:
        >>> retry("./output")
    """
    config = TTSConfig()
    retry_failures(Path(output_dir), config)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert text files to audio using pocket-tts")
    parser.add_argument("input", nargs="?", help="Input text file or directory")
    parser.add_argument("output", nargs="?", help="Output directory")
    parser.add_argument("--retry", action="store_true", help="Retry failed chunks in output dir")
    parser.add_argument("--voice", default="alba", choices=AVAILABLE_VOICES,
                        help="Voice name (default: alba)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Words per chunk")

    args = parser.parse_args()

    if args.retry:
        if not args.input:
            parser.error("--retry requires an output directory")
        retry(args.input)
    elif args.input and args.output:
        input_path = Path(args.input)
        if input_path.is_dir():
            process_dir(args.input, args.output, args.voice, args.chunk_size)
        else:
            process_file(args.input, args.output, args.voice, args.chunk_size)
    else:
        parser.print_help()
