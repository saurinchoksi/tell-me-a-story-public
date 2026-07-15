"""Speaker diarization using pyannote.audio.

Provides the heavy ML pipeline: model loading, audio preparation, and
diarization inference. Speaker enrichment of transcripts lives in speaker.py.
"""

import os

# Let any op unimplemented on Apple's Metal GPU (MPS) fall back to CPU instead of
# crashing the run — must be set before torch is imported (TMAS-95: diarization on GPU).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import subprocess
import tempfile
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# PyTorch 2.6+ changed weights_only default to True for security.
# pyannote.audio models need these classes allowlisted to load.
# This is safe because we're loading official pyannote models from Hugging Face.
from pyannote.audio.core.task import Specifications, Problem, Resolution, Scope
torch.serialization.add_safe_globals([Specifications, Problem, Resolution, Scope])

_GENERATOR_VERSION = "pyannote-speaker-diarization-community-1"
MODEL = "pyannote/speaker-diarization-community-1"


def load_diarization_model() -> Pipeline:
    """Load the speaker diarization model.
    
    Requires HF_TOKEN environment variable to be set.
    First run will download the model (~1GB).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    print("Loading diarization model (this may take a minute)...")
    model = Pipeline.from_pretrained(
        MODEL,
        token=token
    )
    # pyannote does NOT auto-use the GPU (unlike most libraries), so without this the
    # whole pipeline ran on CPU at ~real-time — ~52% of the pipeline's wall-clock. Moving
    # it to Apple's Metal GPU (MPS) is ~10-15x faster and, validated bit-identical to the
    # CPU output across 4 sessions spanning 2-4 speakers (TMAS-95), changes nothing else:
    # same speaker count, same segments, same per-speaker timings. Same weights and memory
    # footprint, so it fits the 8GB product target too. The device is freed in diarize()'s
    # finally block (MPS only reclaims on empty_cache / process exit).
    if torch.backends.mps.is_available():
        model.to(torch.device("mps"))
    return model


def prepare_audio_for_diarization(audio_path: str) -> str:
    """Convert audio file to 16kHz mono WAV for pyannote compatibility.
    
    Returns path to temporary WAV file. Caller is responsible for cleanup.
    """
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",      # mono
        "-loglevel", "error",  # suppress ffmpeg output
        temp_wav.name
    ], check=True)
    
    return temp_wav.name


def diarize(audio_path: str, model: Pipeline = None, num_speakers: int = None) -> dict:
    """Run speaker diarization on an audio file.

    Args:
        audio_path: Path to audio file
        model: Optional pre-loaded diarization model (loads one if not provided).
            If not provided, a model is loaded and explicitly freed before returning.
        num_speakers: Optional hint for exact number of speakers (improves accuracy)

    Returns:
        Dict with '_generator_version' and 'segments' keys.
        Segments is a list of dicts with 'start', 'end', 'speaker' keys.
    """
    model_loaded_here = model is None
    if model_loaded_here:
        model = load_diarization_model()

    # Convert to 16kHz mono WAV for compatibility with pyannote
    wav_path = prepare_audio_for_diarization(audio_path)

    try:
        # Run diarization with progress feedback
        with ProgressHook() as hook:
            output = model(wav_path, hook=hook, num_speakers=num_speakers)

        # Extract segments using exclusive mode (one speaker at a time)
        # This simplifies alignment with transcription timestamps
        segments = []
        for turn, speaker in output.exclusive_speaker_diarization:
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        return {
            "_generator_version": _GENERATOR_VERSION,
            "segments": segments
        }
    finally:
        # Clean up temp file
        os.unlink(wav_path)
        # Free model memory if we loaded it (MPS doesn't release GPU memory on GC alone)
        if model_loaded_here:
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
