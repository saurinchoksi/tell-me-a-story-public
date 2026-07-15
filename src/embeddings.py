"""Speaker embedding extraction using pyannote's wespeaker model.

Extracts per-speaker 256-dimensional embeddings from diarized audio.
Each speaker's segments are embedded independently, L2-normalized,
averaged, and re-normalized to produce a single identity vector.

These embeddings feed into profiles.py for cross-session speaker
identification.
"""

import json
import logging
import os

import numpy as np
import torch
import torchaudio

# PyTorch 2.6+ safe globals for pyannote model loading
from pyannote.audio.core.task import Specifications, Problem, Resolution, Scope
torch.serialization.add_safe_globals([Specifications, Problem, Resolution, Scope])

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

from diarize import prepare_audio_for_diarization

logger = logging.getLogger(__name__)

MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"
MIN_SEGMENT_DURATION = 0.5  # seconds — skip very short segments


def load_embedding_model() -> PretrainedSpeakerEmbedding:
    """Load the speaker embedding model.

    Uses wespeaker-voxceleb-resnet34-LM (public, no HF_TOKEN needed).
    First run downloads the model weights.

    Returns:
        PretrainedSpeakerEmbedding model instance.
    """
    logger.info(f"Loading embedding model: {MODEL}")
    model = PretrainedSpeakerEmbedding(MODEL)
    return model


def _model_label() -> str:
    """Derive a short label from the MODEL constant (e.g. 'wespeaker-voxceleb-resnet34-LM')."""
    return MODEL.split("/")[-1]


def extract_speaker_embeddings(
    model: PretrainedSpeakerEmbedding,
    audio_path: str,
    diarization_data: dict,
) -> dict:
    """Extract per-speaker embeddings from diarized audio.

    Converts audio to 16kHz mono WAV (reusing diarize.py's helper),
    groups diarization segments by speaker, embeds each usable segment,
    and produces one averaged embedding vector per speaker.

    Args:
        model: Loaded embedding model from load_embedding_model().
        audio_path: Path to the original audio file.
        diarization_data: Dict with "segments" list from diarize().

    Returns:
        Dict with schema:
        {
            "_generator_version": "wespeaker-voxceleb-resnet34-LM",
            "_dimension": 256,
            "speakers": {
                "SPEAKER_00": {
                    "vector": [float, ...],
                    "num_segments": int,
                    "total_duration_s": float,
                },
                ...
            }
        }
    """
    wav_path = prepare_audio_for_diarization(audio_path)

    try:
        waveform, sr = torchaudio.load(wav_path)

        # Group segments by speaker
        speaker_segments = {}
        for seg in diarization_data.get("segments", []):
            speaker = seg["speaker"]
            speaker_segments.setdefault(speaker, []).append(seg)

        speakers = {}
        for speaker, segments in sorted(speaker_segments.items()):
            embedding = _embed_speaker(model, waveform, sr, segments)
            if embedding is not None:
                speakers[speaker] = embedding
            else:
                logger.warning(f"No usable segments for {speaker}, skipping")

        return {
            "_generator_version": _model_label(),
            "_dimension": model.dimension,
            "speakers": speakers,
        }
    finally:
        os.unlink(wav_path)


def _embed_speaker(
    model: PretrainedSpeakerEmbedding,
    waveform: torch.Tensor,
    sr: int,
    segments: list[dict],
) -> dict | None:
    """Embed a single speaker from their diarization segments.

    Strategy: embed each segment independently, L2-normalize each,
    average all, re-normalize. This dilutes outliers from noisy segments.

    Returns:
        Dict with "vector", "num_segments", "total_duration_s",
        or None if no usable segments.
    """
    embeddings = []
    total_duration = 0.0
    num_segments = 0

    for seg in segments:
        duration = seg["end"] - seg["start"]
        total_duration += duration

        if duration < MIN_SEGMENT_DURATION:
            continue

        start_sample = max(0, int(seg["start"] * sr))
        end_sample = min(waveform.shape[1], int(seg["end"] * sr))

        if end_sample <= start_sample:
            continue

        chunk = waveform[:, start_sample:end_sample]

        # Check minimum samples if model specifies it
        if hasattr(model, "min_num_samples") and chunk.shape[-1] < model.min_num_samples:
            continue

        # Model expects (batch, channel, samples)
        batch = chunk.unsqueeze(0)  # (1, 1, samples)

        with torch.no_grad():
            emb = model(batch)

        # Model may return torch tensor or numpy array depending on version
        if hasattr(emb, "numpy"):
            emb = emb.squeeze().numpy()
        else:
            emb = np.squeeze(emb)

        # Skip if NaN in output
        if not np.all(np.isfinite(emb)):
            logger.warning(f"Non-finite embedding for segment {seg['start']:.2f}-{seg['end']:.2f}, skipping")
            continue

        # L2-normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        embeddings.append(emb)
        num_segments += 1

    if not embeddings:
        return None

    # Average then re-normalize
    avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm

    # Validate final vector
    if not np.all(np.isfinite(avg)):
        logger.warning("Averaged embedding contains non-finite values")
        return None

    return {
        "vector": avg.tolist(),
        "num_segments": num_segments,
        "total_duration_s": round(total_duration, 2),
    }


def save_embeddings(embeddings_result: dict, output_path: str) -> None:
    """Write embeddings to JSON.

    Args:
        embeddings_result: Dict from extract_speaker_embeddings().
        output_path: Destination file path.
    """
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(embeddings_result, f, indent=2)
