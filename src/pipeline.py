"""Run the full transcription pipeline and save computed artifacts."""

import copy
import glob
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from transcribe import transcribe, clean_transcript, MODEL as TRANSCRIPTION_MODEL, make_processing_entry as make_transcription_entry
from diarize import diarize
from embeddings import load_embedding_model, extract_speaker_embeddings, save_embeddings, MODEL as EMBEDDING_MODEL
from speaker import enrich_with_diarization, detect_unintelligible_gaps, DIARIZATION_MODEL
from mutagen import File as MutagenFile
from dictionary import load_library, build_variant_map, normalize_variants, make_processing_entry as make_dictionary_entry
from corrections import extract_text, apply_corrections
# Story segmentation runs on Qwen3.5 (5/5 story counts, 0.86->0.94 region overlap vs Gemma;
# see emp/src/seg_qwen35.py). The Gemma backend (story_segment.py) stays as the baseline.
from story_segment_qwen35 import segment_transcript, MODEL_ID as SEGMENT_MODEL_ID, SEGMENT_CONFIG_VERSION
from stories import enrich_with_stories
from model_cache import cached_or_run, fingerprint

_DEFAULT_LIBRARY_PATH = None

logger = logging.getLogger(__name__)


def _record_fixer_outcome(cache_dir, fixer: str, version: str, transcript: dict,
                          reason: str, entry: dict) -> None:
    """Leave a fixer's didn't-run trace — a skip or a failed run — in the session's
    shared queue metadata, where /name-review looks (TMAS-72). Recording must never
    sink the pipeline, but a recording failure may not vanish into a log line either:
    it lands on the stage's processing entry as `queue_record_error`."""
    if cache_dir is None:
        return
    try:
        import namefix as _namefix
        _namefix.write_skip(Path(cache_dir), fixer, version, transcript, reason)
    except Exception as e:
        logger.warning(f"{fixer} outcome not recorded in queue: {e}")
        entry["queue_record_error"] = str(e)


def enrich_transcript(
    transcript: dict,
    diarization: dict,
    library_path: str = None,
    verbose: bool = True,
    cache_dir=None,
    audio_path: str = None,
    family_roster_path=None,
    run_name_fixers: bool = True,
) -> tuple[dict, list[dict], dict]:
    """Run all enrichment stages on a transcript.

    Runs diarization enrichment, gap detection, story segmentation, name
    fixing (namefix — needs the audio), and dictionary normalization in
    sequence. Each stage is wrapped in try/except so failures are recorded
    but don't block subsequent stages.

    Does NOT set _processing on the transcript — the caller assembles that.

    Args:
        transcript: Whisper transcript dict with segments containing words.
        diarization: Diarization result dict with a 'segments' list.
        library_path: Path to dictionary library JSON. When None (default),
            dictionary normalization is skipped.
        verbose: Log progress messages.
        cache_dir: Session dir for the segmentation/namefix caches and the
            pending-name-corrections file. None skips both stages (tests).
        audio_path: The session audio — namefix re-decodes short clips from
            it. None skips namefix with a "skipped" processing entry.
        family_roster_path: Override for the family roster (tests). None uses
            the default gitignored data/name_roster.json; an absent file skips
            the familyfix pass with a "skipped" processing entry.
        run_name_fixers: When False, skip all three name fixers (namefix,
            familyfix, spellingfix) — each records a "skipped"/"deferred"
            processing entry instead of running. The transcripts-first churn
            (load_oldest --skip-name-fixers) sets this so the expensive per-name
            re-listens defer to a later fix_session batch (TMAS-95). Default True.

    Returns:
        Tuple of (enriched_transcript, processing_entries, counts_dict).
        processing_entries: list of stage dicts for _processing.
        counts_dict: {"dict_count": M, "namefix_groups": N} — namefix runs in SHADOW
        (calibration), so there is no applied count; N is labeled questions written.
    """
    processing = []
    dict_count = 0
    namefix_groups = 0
    family_groups = 0
    spelling_groups = 0
    story_worlds: dict = {}   # story idx -> recognized world (namefix's report, Pass 4)
    nf_groups: list = []      # namefix's groups — Pass 4c's defer-to-namefix input

    # Word realignment (transcript repair) now runs in run_pipeline BEFORE the raw
    # snapshot — see realign_words() — so enrich_transcript holds only interpretive
    # passes. On --re-enrich, transcript-raw.json already carries corrected timings,
    # so the passes below key off them without re-running the deterministic aligner.

    # Pass 1: Diarization enrichment
    try:
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        transcript, diar_entry = enrich_with_diarization(transcript, diarization)
        diar_entry["started_at"] = started_at
        diar_entry["duration_seconds"] = round(time.monotonic() - t0, 2)
        processing.append(diar_entry)
    except Exception as e:
        logger.warning(f"Diarization enrichment failed: {e}")
        processing.append({
            "stage": "diarization_enrichment",
            "model": DIARIZATION_MODEL,
            "status": "error",
            "error": str(e),
            "started_at": started_at,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Pass 2: Unintelligible gap detection
    try:
        started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        transcript, gap_entry = detect_unintelligible_gaps(transcript, diarization)
        gap_entry["started_at"] = started_at
        gap_entry["duration_seconds"] = round(time.monotonic() - t0, 2)
        processing.append(gap_entry)
        if verbose:
            logger.info(f"Gap detection: {gap_entry['gaps_found']} unintelligible gaps injected")
    except Exception as e:
        logger.warning(f"Gap detection failed: {e}")
        processing.append({
            "stage": "gap_detection",
            "status": "error",
            "error": str(e),
            "started_at": started_at,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Pass 3: Story segmentation (cached — re-enrich reuses stories when the segment
    # structure/text and the segmenter config are unchanged). Moved BEFORE name fixing
    # (2026-07-01): namefix corrects per story, so it needs _stories first — and the
    # segmenter reads boundary cues, not name spellings, so running it on un-corrected
    # text is safe (verified by the tmas-segment-truth ground-truth check design).
    # Runs only when a session dir to cache into is given. Degrade-safe.
    if cache_dir is not None:
        try:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            seg_input = "".join(
                f"{s['id']}|{s.get('start')}|{s.get('end')}|{s.get('text', '')}\n"
                for s in transcript["segments"])
            stories, was_cached = cached_or_run(
                cache_dir, "story_segmentation",
                fingerprint(seg_input), fingerprint(SEGMENT_CONFIG_VERSION),
                lambda: segment_transcript(transcript))
            transcript = enrich_with_stories(transcript, stories)
            processing.append({
                "stage": "story_segmentation",
                "model": SEGMENT_MODEL_ID,
                "status": "success",
                "stories_found": len(stories),
                "from_cache": was_cached,
                "started_at": started_at,
                "duration_seconds": round(time.monotonic() - t0, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if verbose:
                logger.info(f"Story segmentation: {len(stories)} stories"
                            + (" (cached)" if was_cached else ""))
        except Exception as e:
            logger.warning(f"Story segmentation failed: {e}")
            processing.append({
                "stage": "story_segmentation",
                "model": SEGMENT_MODEL_ID,
                "status": "error",
                "error": str(e),
                "started_at": started_at,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    # Pass 4: Name fixing (namefix) — running in SHADOW during the calibration phase
    # (2026-07-02, docs/name-architecture.md). The stage re-decodes the AUDIO around each
    # suspected name with the story's world+cast in Whisper's ear, but APPLIES NOTHING:
    # every opinion becomes a labeled question in pending-name-corrections.json, sorted by
    # why the machine believed it (sounds-like / audio-only / ordinary-word). Transcripts
    # stay honest/blind until the deliberate apply pass after classes graduate. The
    # calibration rule: this stage may only add labeled questions and read the Name Book.
    # Needs _stories + the audio; skips cleanly without them. Cached like segmentation.
    import namefix as _namefix
    if run_name_fixers and cache_dir is not None and audio_path is not None and transcript.get("_stories"):
        try:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            nf_input = _namefix.transcript_fingerprint(transcript)
            result, was_cached = cached_or_run(
                cache_dir, "namefix",
                fingerprint(nf_input), fingerprint(_namefix.config_fingerprint()),
                lambda: _namefix.run_namefix(transcript, str(audio_path)))
            namefix_groups = len(result["groups"])
            nf_groups = result["groups"]
            story_worlds = {w["story_id"]: w["recognized_world"]
                            for w in result["worlds"]}
            by_label: dict = {}
            for g in result["groups"]:
                by_label[g["label"]] = by_label.get(g["label"], 0) + 1
            _namefix.write_pending(Path(cache_dir), result, transcript)
            processing.append({
                "stage": "namefix",
                "model": _namefix.MODEL,
                "status": "success",
                "namefix_version": _namefix.VERSION,
                "shadow": True,
                "groups": namefix_groups,
                "by_label": by_label,
                "worlds": [w["recognized_world"] for w in result["worlds"]],
                "from_cache": was_cached,
                "started_at": started_at,
                "duration_seconds": round(time.monotonic() - t0, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if verbose:
                logger.info(f"Namefix (shadow): {namefix_groups} labeled questions "
                            f"{by_label}, nothing applied"
                            + (" (cached)" if was_cached else ""))
        except Exception as e:
            logger.warning(f"Namefix failed: {e}")
            entry = {
                "stage": "namefix",
                "status": "error",
                "error": str(e),
                "started_at": started_at,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _record_fixer_outcome(cache_dir, _namefix.FIXER, _namefix.VERSION,
                                  transcript, "error", entry)
            processing.append(entry)
    else:
        reason = ("deferred" if not run_name_fixers else
                  "no_audio_path" if audio_path is None else
                  "no_stories" if cache_dir is not None else "no_cache_dir")
        if verbose:
            logger.info(f"Namefix: skipped ({reason})")
        entry = {
            "stage": "namefix",
            "status": "skipped",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _record_fixer_outcome(cache_dir, _namefix.FIXER, _namefix.VERSION,
                              transcript, reason, entry)
        processing.append(entry)

    # Pass 4b: Family-name fixing (familyfix) — SHADOW, same calibration rule as
    # namefix. The frozen m9a detector's matching finds the suspect spots; one Whisper
    # subprocess re-listens with the family roster as the cast; coverage-first, every
    # detector-known spot becomes a labeled question (fixer="family") in the shared
    # queue. Needs the audio + the gitignored roster; skips cleanly without them.
    # No stories needed — family names apply to every recording. Cached like namefix.
    import familyfix as _familyfix
    roster_path = Path(family_roster_path) if family_roster_path \
        else _familyfix.DEFAULT_ROSTER_PATH
    if run_name_fixers and cache_dir is not None and audio_path is not None and roster_path.exists():
        try:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            ff_input = _namefix.transcript_fingerprint(transcript)
            ff_config = _familyfix.config_fingerprint(roster_path)
            ff_result, was_cached = cached_or_run(
                cache_dir, "family_namefix",
                fingerprint(ff_input), fingerprint(ff_config),
                lambda: _familyfix.run_familyfix(transcript, str(audio_path),
                                                 roster_path=roster_path))
            family_groups = len(ff_result["groups"])
            _namefix.write_queue(Path(cache_dir), _familyfix.FIXER, ff_result["groups"],
                                 transcript, _familyfix.VERSION, ff_config)
            processing.append({
                "stage": "family_namefix",
                "model": _namefix.MODEL,
                "status": "success",
                "familyfix_version": _familyfix.VERSION,
                "shadow": True,
                "groups": family_groups,
                "from_cache": was_cached,
                "started_at": started_at,
                "duration_seconds": round(time.monotonic() - t0, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if verbose:
                logger.info(f"Familyfix (shadow): {family_groups} labeled questions, "
                            f"nothing applied" + (" (cached)" if was_cached else ""))
        except Exception as e:
            logger.warning(f"Familyfix failed: {e}")
            entry = {
                "stage": "family_namefix",
                "status": "error",
                "error": str(e),
                "started_at": started_at,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _record_fixer_outcome(cache_dir, _familyfix.FIXER, _familyfix.VERSION,
                                  transcript, "error", entry)
            processing.append(entry)
    else:
        reason = ("deferred" if not run_name_fixers else
                  "no_cache_dir" if cache_dir is None else
                  "no_audio_path" if audio_path is None else "no_roster")
        if verbose:
            logger.info(f"Familyfix: skipped ({reason})")
        entry = {
            "stage": "family_namefix",
            "status": "skipped",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _record_fixer_outcome(cache_dir, _familyfix.FIXER, _familyfix.VERSION,
                              transcript, reason, entry)
        processing.append(entry)

    # Pass 4c: Invented-name spelling proposals (spellingfix) — SHADOW, pure code (no
    # model, milliseconds): cluster capitalized tokens by sound (the m9b semantics),
    # propose the majority spelling for each inconsistently-spelled invented name, and
    # ask one labeled question per deviant spelling (fixer="spelling") in the shared
    # queue. Worlds ride in from namefix's Pass-4 report; without one the questions
    # carry world "" and verdicts record audit-only. Needs only a session dir.
    if run_name_fixers and cache_dir is not None:
        try:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            # imported inside the try (unlike namefix, which familyfix's unconditional
            # import already pulls): a broken shadow-stage module must not sink the
            # minutes of transcription work behind it
            import spellingfix as _spellingfix
            sf_result = _spellingfix.run_spellingfix(
                transcript, story_worlds, roster_path=roster_path,
                defer_words=_spellingfix.defer_words_from(nf_groups))
            spelling_groups = len(sf_result["groups"])
            _namefix.write_queue(Path(cache_dir), _spellingfix.FIXER,
                                 sf_result["groups"], transcript, _spellingfix.VERSION,
                                 _spellingfix.config_fingerprint(roster_path=roster_path))
            processing.append({
                "stage": "spellingfix",
                "status": "success",
                "spellingfix_version": _spellingfix.VERSION,
                "shadow": True,
                "groups": spelling_groups,
                "started_at": started_at,
                "duration_seconds": round(time.monotonic() - t0, 2),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if verbose:
                logger.info(f"Spellingfix (shadow): {spelling_groups} labeled questions, "
                            "nothing applied")
        except Exception as e:
            logger.warning(f"Spellingfix failed: {e}")
            entry = {
                "stage": "spellingfix",
                "status": "error",
                "error": str(e),
                "started_at": started_at,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # literal tag/version: _spellingfix may be exactly what failed to import
            _record_fixer_outcome(cache_dir, "spelling", "unknown",
                                  transcript, "error", entry)
            processing.append(entry)
    else:
        reason = "deferred" if not run_name_fixers else "no_cache_dir"
        if verbose:
            logger.info(f"Spellingfix: skipped ({reason})")
        entry = {
            "stage": "spellingfix",
            "status": "skipped",
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        # Record the skip in the shared queue like namefix/familyfix do (above), so all
        # three fixers stay in parity for /name-review + the TMAS-72 audit: a deferred
        # churn must show "spelling: deferred", not a missing key. _record_fixer_outcome
        # no-ops when cache_dir is None, so the no_cache_dir case is unchanged.
        try:
            import spellingfix as _spellingfix
            _record_fixer_outcome(cache_dir, _spellingfix.FIXER, _spellingfix.VERSION,
                                  transcript, reason, entry)
        except Exception:
            _record_fixer_outcome(cache_dir, "spelling", "unknown", transcript, reason, entry)
        processing.append(entry)

    # Pass 5: Dictionary normalization
    lib_path = library_path or _DEFAULT_LIBRARY_PATH
    if lib_path is None:
        if verbose:
            logger.info("Dictionary normalization: skipped (no library path)")
        processing.append({
            "stage": "dictionary_normalization",
            "status": "skipped",
            "reason": "no_library_path",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    else:
        try:
            started_at = datetime.now(timezone.utc).isoformat()
            t0 = time.monotonic()
            library = load_library(lib_path)
            variant_map = build_variant_map(library)
            text = extract_text(transcript)
            dict_corrections = normalize_variants(text, variant_map)
            transcript, dict_count = apply_corrections(transcript, dict_corrections, "dictionary")
            dict_entry = make_dictionary_entry(lib_path, dict_count)
            dict_entry["started_at"] = started_at
            dict_entry["duration_seconds"] = round(time.monotonic() - t0, 2)
            processing.append(dict_entry)
            if verbose:
                logger.info(f"Dictionary normalization: {dict_count} corrections applied")
        except Exception as e:
            logger.warning(f"Dictionary normalization failed: {e}")
            processing.append({
                "stage": "dictionary_normalization",
                "library": lib_path,
                "status": "error",
                "error": str(e),
                "started_at": started_at,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return transcript, processing, {"dict_count": dict_count,
                                    "namefix_groups": namefix_groups,
                                    "family_groups": family_groups,
                                    "spelling_groups": spelling_groups}


def get_audio_info(filepath: str) -> dict | None:
    """Return audio properties enriched with ffprobe metadata.

    Reads Mutagen for codec info and ffprobe for recording date,
    Voice Memos UUID, and encoder tags directly from the audio file.
    """
    path = Path(filepath)

    if not path.exists():
        return None

    audio = MutagenFile(path)

    if audio is None:
        return None

    info = {
        "filename": path.name,
        "format": type(audio).__name__,
        "duration_seconds": audio.info.length,
        "sample_rate": audio.info.sample_rate,
        "channels": audio.info.channels,
    }

    # Enrich from ffprobe tags embedded in the audio file
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
             str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            tags = json.loads(result.stdout).get("format", {}).get("tags", {})
            if tags.get("creation_time"):
                info["recording_date_utc"] = tags["creation_time"]
            if tags.get("voice-memo-uuid"):
                info["voice_memo_uuid"] = tags["voice-memo-uuid"]
            if tags.get("encoder"):
                info["encoder"] = tags["encoder"]
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return info


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def guard_re_enrich(session_dir: str, force: bool = False) -> None:
    """Refuse a full --re-enrich on a coded session unless forced (TMAS-105).

    EMP axial codes (axial-labels.json) join to transcript segments by their
    `id` field; a full re-enrich rebuilds transcript-rich.json and can orphan
    every code. A coded session wants surgical in-place edits, not a rebuild.
    """
    labels_path = os.path.join(session_dir, "axial-labels.json")
    if force or not os.path.exists(labels_path):
        return
    raise SystemExit(
        f"REFUSED: {session_dir} is a coded session — {labels_path} exists.\n"
        "A full --re-enrich rebuilds transcript-rich.json and can orphan the "
        "EMP axial codes, which join on segment ids.\n"
        "Pass --force to proceed anyway."
    )


def save_computed(session_dir: str, transcript_raw: dict, transcript: dict, diarization: dict) -> None:
    """Save computed artifacts to session directory.

    Creates:
        {session_dir}/
            transcript-raw.json
            transcript-rich.json
            diarization.json

    Args:
        session_dir: Path to session directory
        transcript_raw: Clean Whisper output before enrichment (immutable)
        transcript: Enriched transcript with corrections, speakers, and audio info
        diarization: Diarization result from diarize()
    """
    os.makedirs(session_dir, exist_ok=True)

    # Save raw transcript (pre-enrichment snapshot)
    with open(os.path.join(session_dir, "transcript-raw.json"), "w") as f:
        json.dump(transcript_raw, f, indent=2)

    # Save enriched transcript
    with open(os.path.join(session_dir, "transcript-rich.json"), "w") as f:
        json.dump(transcript, f, indent=2)

    # Save diarization
    with open(os.path.join(session_dir, "diarization.json"), "w") as f:
        json.dump(diarization, f, indent=2)


def realign_words(transcript: dict, audio_path: str, verbose: bool = True) -> tuple[dict, dict]:
    """Forced-alignment word realignment — transcript repair, not enrichment (TMAS-54).

    Re-times each segment's words from its trusted `text`, fixing Whisper's
    segment-initial drift and rescuing the words `clean_transcript` pruned from
    `words[]` but kept in `text`. Runs in run_pipeline as part of transcript
    cleanup (right after clean_transcript, BEFORE the raw snapshot) so
    transcript-raw.json and transcript-rich.json agree on timings, and --re-enrich
    reuses the corrected timings instead of re-running this deterministic aligner.

    Independent of diarization — both consume only the audio; neither reads the
    other's output. A failure is caught and recorded in the returned entry
    (status='error') without aborting the run.

    Returns (transcript, processing_entry).
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    try:
        import numpy as np
        from realign import load_aligner, realign_transcript
        from mlx_whisper.audio import load_audio
        audio = np.array(load_audio(str(audio_path))).astype(np.float32)
        transcript, stats = realign_transcript(transcript, audio, load_aligner())
        entry = {
            "stage": "word_realignment",
            "model": "torchaudio-MMS_FA",
            "status": "success",
            "started_at": started_at,
            "duration_seconds": round(time.monotonic() - t0, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **{k: stats[k] for k in
               ("realigned", "rescued", "rescued_words", "guarded", "skipped")},
        }
        if verbose:
            logger.info(
                f"Word realignment: {stats['realigned']} realigned, "
                f"{stats['rescued']} rescued (+{stats['rescued_words']} words), "
                f"{stats['guarded']} guarded")
    except Exception as e:
        logger.warning(f"Word realignment failed: {e}")
        entry = {
            "stage": "word_realignment",
            "status": "error",
            "error": str(e),
            "started_at": started_at,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return transcript, entry


def run_pipeline(audio_path: str, verbose: bool = True, library_path: str = None,
                 run_name_fixers: bool = True) -> dict:
    """Run full pipeline on an audio file.

    Args:
        audio_path: Path to audio file
        verbose: Print progress messages
        library_path: Path to dictionary library JSON. When None (default),
            dictionary normalization is skipped.

    Returns:
        Dict with 'session_id', 'transcript_raw', 'transcript', 'diarization',
        and 'embeddings' keys. 'embeddings' is None if extraction failed.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    session_id = Path(audio_path).parent.name
    if not re.fullmatch(r"\d{8}-\d{6}", session_id):
        raise ValueError(
            f"Invalid session ID '{session_id}' — expected YYYYMMDD-HHMMSS format "
            f"(from parent directory of {audio_path})"
        )

    pipeline_start = time.monotonic()

    # Get audio info
    audio_info = get_audio_info(audio_path)

    # Transcription
    if verbose:
        print(f"Transcribing: {audio_path}")
        print(f"Using: {TRANSCRIPTION_MODEL}")

    transcript_time = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    raw_transcript = transcribe(audio_path, model=TRANSCRIPTION_MODEL)
    transcript = clean_transcript(raw_transcript)
    transcription_duration = round(time.monotonic() - t0, 2)

    # Word realignment — the final step of transcript cleanup (transcribe → clean →
    # realign). Runs BEFORE the raw snapshot so transcript-raw carries corrected
    # timings, and before diarize() (the two are independent). See realign_words().
    if verbose:
        print("\nRealigning word timings...")
    transcript, realignment_entry = realign_words(transcript, audio_path, verbose=verbose)

    transcript_raw = copy.deepcopy(transcript)

    transcription_entry = make_transcription_entry(compute_file_hash(audio_path), transcript_time)
    transcription_entry["started_at"] = transcript_time
    transcription_entry["duration_seconds"] = transcription_duration

    # Diarization
    if verbose:
        print("\nDiarizing (this takes a few minutes)...")

    diar_started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    diarization = diarize(audio_path)
    diarization_entry = {
        "stage": "diarization",
        "model": DIARIZATION_MODEL,
        "status": "success",
        "started_at": diar_started_at,
        "duration_seconds": round(time.monotonic() - t0, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Embedding extraction (diarization model already freed by diarize())
    embeddings_result = None
    embedding_entry = None
    try:
        if verbose:
            print("\nExtracting speaker embeddings...")
        emb_started_at = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        embedding_model = load_embedding_model()
        embeddings_result = extract_speaker_embeddings(embedding_model, audio_path, diarization)
        embedding_entry = {
            "stage": "embedding_extraction",
            "model": EMBEDDING_MODEL,
            "status": "success",
            "started_at": emb_started_at,
            "duration_seconds": round(time.monotonic() - t0, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        del embedding_model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:
        logger.warning(f"Embedding extraction failed: {e}")
        embedding_entry = {
            "stage": "embedding_extraction",
            "model": EMBEDDING_MODEL,
            "status": "error",
            "error": str(e),
            "started_at": emb_started_at,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Enrichment (diarization labels, gaps, stories, namefix, dictionary)
    transcript, enrichment_processing, _ = enrich_transcript(
        transcript, diarization, library_path=library_path, verbose=verbose,
        cache_dir=Path(audio_path).parent, audio_path=audio_path,
        run_name_fixers=run_name_fixers,
    )

    # Fold audio info into enriched transcript
    transcript["audio"] = audio_info

    processing = [transcription_entry, realignment_entry, diarization_entry] + enrichment_processing
    if embedding_entry is not None:
        processing.append(embedding_entry)
    transcript["_processing"] = processing

    # Build _stats
    segments = transcript["segments"]
    total_words = sum(len(seg["words"]) for seg in segments)
    speakers = {w["_speaker"]["label"] for seg in segments for w in seg["words"] if "_speaker" in w and w["_speaker"]["label"] is not None}
    transcript["_stats"] = {
        "pipeline_started_at": transcript_time,
        "pipeline_duration_seconds": round(time.monotonic() - pipeline_start, 2),
        "segments": len(segments),
        "words": total_words,
        "speakers": len(speakers),
        "stories": len(transcript.get("_stories", [])),
    }

    return {
        "session_id": session_id,
        "transcript_raw": transcript_raw,
        "transcript": transcript,
        "diarization": diarization,
        "embeddings": embeddings_result,
    }


def to_utterances(labeled_words: list[dict]) -> list[dict]:
    """Convert labeled words to utterances, consolidating same-speaker runs.

    Accepts both enriched format (speaker at word["_speaker"]["label"]) and
    flat format (speaker at word["speaker"]). Enriched format takes priority.

    Args:
        labeled_words: Words with '_speaker.label' or 'speaker' field

    Returns:
        List of utterance dicts with consolidated text and word arrays
    """
    if not labeled_words:
        return []

    utterances = []
    current = None

    for word in labeled_words:
        if "_speaker" in word:
            speaker = word["_speaker"].get("label")
        else:
            speaker = word.get("speaker")
        text = word.get("word", "").strip()

        # Start new utterance if speaker changes (or first word)
        # Also start new if current speaker is None (don't consolidate unknowns)
        if current is None or speaker != current["speaker"] or current["speaker"] is None:
            if current is not None:
                utterances.append(current)
            current = {
                "speaker": speaker,
                "start": word.get("start", 0),
                "end": word.get("end", 0),
                "text": text,
                "words": [word]
            }
        else:
            # Same speaker - extend current utterance
            current["end"] = word.get("end", current["end"])
            current["text"] = f"{current['text']} {text}"
            current["words"].append(word)

    # Don't forget the last utterance
    if current is not None:
        utterances.append(current)

    return utterances


def format_transcript(utterances: list[dict]) -> str:
    """Format utterances as human-readable transcript.

    Args:
        utterances: List of utterance dicts from to_utterances()

    Returns:
        Formatted string with "SPEAKER: text" lines
    """
    lines = []
    for utt in utterances:
        speaker = utt.get("speaker") or "UNKNOWN"
        text = utt.get("text", "")
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run transcription pipeline")
    parser.add_argument("path", help="Path to audio file, or session directory with --re-enrich")
    parser.add_argument("--re-enrich", action="store_true",
                        help="Re-enrich an existing session from transcript-raw.json")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich even a coded session (may orphan its EMP axial codes)")
    parser.add_argument("--library", default=None,
                        help="Path to dictionary library JSON")
    args = parser.parse_args()

    if args.re_enrich:
        # Re-enrich mode: load raw transcript + diarization, skip transcription
        session_dir = args.path
        guard_re_enrich(session_dir, force=args.force)
        raw_path = os.path.join(session_dir, "transcript-raw.json")
        diarization_path = os.path.join(session_dir, "diarization.json")

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"No transcript-raw.json in {session_dir}")
        if not os.path.exists(diarization_path):
            raise FileNotFoundError(f"No diarization.json in {session_dir}")

        with open(raw_path) as f:
            transcript_raw = json.load(f)
        with open(diarization_path) as f:
            diarization = json.load(f)

        # Deep-copy raw so enrichment doesn't mutate it
        transcript = copy.deepcopy(transcript_raw)

        # Find the session audio first — namefix re-decodes clips from it
        audio_matches = glob.glob(os.path.join(session_dir, "audio.*"))
        audio_path = audio_matches[0] if audio_matches else None

        print("Running enrichment stages...")
        enrich_started_at = datetime.now(timezone.utc).isoformat()
        enrich_start = time.monotonic()
        transcript, enrichment_processing, counts = enrich_transcript(
            transcript, diarization, library_path=args.library, cache_dir=session_dir,
            audio_path=audio_path,
        )
        enrich_duration = round(time.monotonic() - enrich_start, 2)

        # Build transcription stub from raw's generator version
        transcription_entry = {
            "stage": "transcription",
            "model": transcript_raw.get("_generator_version", "unknown"),
            "status": "prior_run",
        }
        if audio_path:
            audio_info = get_audio_info(audio_path)
            transcript["audio"] = audio_info
            transcription_entry["audio_hash"] = compute_file_hash(audio_path)

        processing = [transcription_entry] + enrichment_processing
        transcript["_processing"] = processing

        # Build _stats for re-enrich
        segments = transcript["segments"]
        total_words = sum(len(seg["words"]) for seg in segments)
        speakers = {w["_speaker"]["label"] for seg in segments for w in seg["words"] if "_speaker" in w and w["_speaker"]["label"] is not None}
        transcript["_stats"] = {
            "pipeline_started_at": enrich_started_at,
            "pipeline_duration_seconds": enrich_duration,
            "segments": len(segments),
            "words": total_words,
            "speakers": len(speakers),
            "stories": len(transcript.get("_stories", [])),
        }

        # Save enriched transcript
        with open(os.path.join(session_dir, "transcript-rich.json"), "w") as f:
            json.dump(transcript, f, indent=2)

        # Measure file sizes and re-save with them
        file_sizes = {}
        for name in ["transcript-raw.json", "transcript-rich.json", "diarization.json"]:
            fpath = os.path.join(session_dir, name)
            if os.path.exists(fpath):
                file_sizes[name] = os.path.getsize(fpath)
        emb_path = os.path.join(session_dir, "embeddings.json")
        if os.path.exists(emb_path):
            file_sizes["embeddings.json"] = os.path.getsize(emb_path)
        transcript["_stats"]["file_sizes"] = file_sizes
        with open(os.path.join(session_dir, "transcript-rich.json"), "w") as f:
            json.dump(transcript, f, indent=2)

        print(f"\nSaved to: {session_dir}/transcript-rich.json")
        print(f"  Enrichment: {enrich_duration}s")
        print(f"  Dictionary corrections: {counts['dict_count']}")
    else:
        # Full pipeline mode
        result = run_pipeline(args.path, library_path=args.library)

        session_id = result["session_id"]
        session_dir = f"sessions/{session_id}"

        save_computed(
            session_dir=session_dir,
            transcript_raw=result["transcript_raw"],
            transcript=result["transcript"],
            diarization=result["diarization"],
        )

        # Save embeddings if extraction succeeded
        if result["embeddings"] is not None:
            embeddings_path = os.path.join(session_dir, "embeddings.json")
            save_embeddings(result["embeddings"], embeddings_path)

        # Measure file sizes and attach to _stats
        file_sizes = {}
        for name in ["transcript-raw.json", "transcript-rich.json", "diarization.json"]:
            file_sizes[name] = os.path.getsize(os.path.join(session_dir, name))
        if result["embeddings"] is not None:
            file_sizes["embeddings.json"] = os.path.getsize(os.path.join(session_dir, "embeddings.json"))
        result["transcript"]["_stats"]["file_sizes"] = file_sizes
        # Re-save with file sizes
        with open(os.path.join(session_dir, "transcript-rich.json"), "w") as f:
            json.dump(result["transcript"], f, indent=2)

        # Auto-identify against existing profiles
        from identify import identify_speakers, save_identifications
        from profiles import load_profiles
        profiles = load_profiles()
        if profiles.get("profiles") and result["embeddings"] is not None:
            identifications = identify_speakers(os.path.join(session_dir, "embeddings.json"))
            save_identifications(identifications, os.path.join(session_dir, "identifications.json"))
            matched = [i for i in identifications["identifications"] if i["status"] != "unknown"]
            if matched:
                print(f"\nSpeaker identification: {len(matched)} matched")
                for m in matched:
                    print(f"  {m['speaker_key']} → {m['profile_name']} ({m['status']}, {m['confidence']})")
            else:
                print(f"\nSpeaker identification: no matches (0/{len(identifications['identifications'])} speakers)")

        # Flatten enriched words for utterance grouping
        labeled_words = []
        for seg in result["transcript"]["segments"]:
            labeled_words.extend(seg.get("words", []))
        utterances = to_utterances(labeled_words)

        print("\n--- Speaker-labeled transcript ---\n")
        print(format_transcript(utterances))

        print(f"\nSaved session to: {session_dir}/")
        print(f"  transcript-raw.json")
        print(f"  transcript-rich.json")
        print(f"  diarization.json")
        if result["embeddings"] is not None:
            n_speakers = len(result["embeddings"]["speakers"])
            print(f"  embeddings.json ({n_speakers} speakers)")

        processing = result["transcript"]["_processing"]
        for entry in processing:
            if entry["stage"] == "dictionary_normalization":
                print(f"Dictionary Normalization: {entry.get('corrections_applied', 0)} corrections")

        # Print timing summary
        stats = result["transcript"]["_stats"]
        print(f"\nPipeline stats:")
        print(f"  Total: {stats['pipeline_duration_seconds']}s")
        print(f"  Segments: {stats['segments']}, Words: {stats['words']}, Speakers: {stats['speakers']}")
        for entry in processing:
            if "duration_seconds" in entry:
                print(f"  {entry['stage']}: {entry['duration_seconds']}s")
