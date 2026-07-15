# Tell Me a Story

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)
![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)
![Processing: 100% local](https://img.shields.io/badge/processing-100%25%20local-brightgreen.svg)

A local system for capturing bedtime stories. Audio goes in, speaker-labeled transcripts come out, and the system flags its own likely mistakes for a human to check.

Every night, my daughter and I tell stories together. Characters recur and plot threads tangle across weeks of bedtime improv.

I wanted to capture those conversations, and eventually "see" the stories we're building together. So I built my own pipeline.

## Contents

- [The Mahabharata](#the-mahabharata)
- [What It Does](#what-it-does)
- [Hallucination Handling](#hallucination-handling)
- [Catching Its Own Mistakes](#catching-its-own-mistakes)
- [Why Local](#why-local)
- [Capture Device](#capture-device)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Running It](#running-it)
- [Background](#background)

## The Mahabharata

Our test recordings are of my daughter asking questions about the Mahabharata — the ancient Indian epic I grew up with, now being passed down to her. "Why did Duryodhana want to be king?" "What happened to the Pandavas?" These are real questions from a four-year-old, and in some ways, the heart of the project.

The Mahabharata is the first content domain, not the only one. The pipeline is built to handle any storytelling session — improvised adventures, fairy tale retellings, interdimensional portal hopping tales, whatever.

## What It Does

Five-stage pipeline that turns a raw audio recording into a speaker-labeled, corrected transcript:

1. **Transcription** — Converts audio to text with word-level timestamps. Long-form models work best for capturing quiet child speech; short clips lose too much context.

2. **Diarization** — Identifies *who* is speaking and *when*, producing time-stamped speaker segments independent of the transcript. A father and daughter sound quite different, which helps — but toddler interjections are still the hardest case.

3. **LLM Normalization** — A local language model reads the full transcript and corrects words that sound like names but got mangled by the transcriber. For our Mahabharata sessions, "Durian" becomes "Duryodhana" and "fondos" becomes "Pandavas."

4. **Speaker Enrichment** — Merges the diarization segments onto the transcript at word level, so every word knows who said it and how confident that attribution is. Words that fall in silence gaps or between speakers get flagged rather than guessed at.

5. **Identify** — Recognizes *who* each speaker actually is across recordings, not just "speaker A vs. speaker B" inside one session. From a short voice sample of each of us, it labels me and my daughter even on a session it has never heard, and marks an attribution "suggested" rather than forcing a confident match when a silly voice or a far-from-the-mic moment makes it unsure.

Each stage adds information without destroying what came before. Original transcriptions are preserved alongside corrections, and every change is tracked with an audit trail. The raw transcript is saved separately and never modified — it's the honest record of what the transcriber heard.

### Before and After

Raw Whisper output (what the transcriber hears):

> Dad, why do the **fondos** and the **goros** want to be king?
> Well, so the oldest brother of the **goros**, his name was, do you remember?
> **Durioden**.

After enrichment (LLM + dictionary corrections, speaker labels):

> **CHILD:** Dad, why do the **Pandavas** and the **Kauravas** want to be king?
> **DAD:** Well, so the oldest brother of the **Kauravas**, his name was, do you remember?
> **CHILD:** **Duryodhana**.

## Hallucination Handling

Speech transcription models hallucinate — they generate confident-sounding text during silence, breathing, or background noise. With a toddler's soft voice and long pauses, this happens a lot.

Three detection filters flag suspect content: silence gaps where no speaker was detected, near-zero probability words where the transcriber itself wasn't confident, and duplicate segments that repeat at processing boundaries. These are review aids, not automated deletion — the filters surface candidates for human judgment.

## Catching Its Own Mistakes

The transcriber mishears names constantly, and names are the heart of these stories. So the system runs its own checks and flags suspect words for me to review, instead of trusting the transcript blind.

It looks for three kinds of name error, sorted by where the right spelling should come from:

- **Family names** — our own names (my daughter's name alone is the single most common mistake). A phonetic match against a small roster catches these with plain code, no model needed.
- **Made-up names** — characters invented on the spot, where the only tell is *inconsistency*: the same name spelled several ways across one story. Code clusters the spellings; a small local model settles the ambiguous cases.
- **Known characters** — names with a real source, like the Mahabharata or Thomas & Friends. A local model reads the story's names, works out which world the story is in, and flags a misheard canon name along with its correct spelling.

Everything surfaces in a review screen (the Monitor), with a play button on each flag so I can check the audio by ear before accepting a fix. Nothing is auto-corrected on the model's say-so.

Each detector is validated honestly: I mark the right answers by ear on recordings the detector has never seen, then score it against that held-out key and read *both* kinds of error — false alarms and misses — before trusting it. More in the [build log](https://saurinchoksi.com/tell-me-a-story-build-log.html) and the case studies at [saurinchoksi.com](https://saurinchoksi.com).

## Why Local

For recordings of my kid, I wanted everything to run on local hardware with no cloud dependency. No API keys, no data leaving the house.

The constraint forced me to understand the full pipeline — model behavior, hallucination patterns, speaker alignment — rather than outsourcing it to a service.

It also let me work with technology I was excited about.

## Capture Device

On the horizon: an ESP32-based capture device so recording disappears into the background. Tap to start, tap to stop — not always listening, no screens, no attention-grabbing LEDs, operable in the dark.

Audio records to an SD card initially, with WiFi sync to the processing machine planned for later.

## Tech Stack

- **MLX Whisper** — Transcription, optimized for Apple Silicon
- **Pyannote** — Speaker diarization
- **Qwen 3.5 4B (via MLX-VLM)** — one small local model for all the language work: name correction, story splitting, and world recognition (consolidated from two models down to one in June 2026; tracked in the [build log](https://saurinchoksi.com/tell-me-a-story-build-log.html))
- **Python** — Pipeline, automated tests (fast/slow split), CLI
- **Flask** — Validation player with waveform visualization and word-level highlighting

## Project Structure

```
tell-me-a-story/
├── src/
│   ├── pipeline.py          # end-to-end pipeline
│   ├── process_inbox.py     # batch ingestion from the inbox
│   ├── transcribe.py        # transcription (MLX Whisper)
│   ├── diarize.py           # diarization (pyannote)
│   ├── normalize.py         # name correction
│   ├── identify.py          # cross-session speaker ID
│   ├── stories.py           # story segmentation
│   └── detectors/           # error-detection framework
│       ├── family_names.py      # family names
│       ├── name_consistency.py  # made-up names
│       └── story_names/         # known characters (canon)
├── api/                     # Flask API layer
├── ui/                      # validation player (waveform + word highlighting)
├── data/                    # name reference library
└── tests/                   # test suite (fast/slow split)
```

## Requirements

- Apple Silicon Mac (for MLX Whisper and MLX-VLM)
- Python 3.14
- Hugging Face token (for Pyannote model download)
- FFmpeg (for audio format conversion)

## Running It

```bash
# Clone and set up
git clone https://github.com/saurinchoksi/tell-me-a-story.git
cd tell-me-a-story
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set Hugging Face token (required for Pyannote)
export HF_TOKEN=your_token_here

# Drop audio in the inbox and process
mkdir -p inbox
cp ~/your-recording.m4a inbox/
python src/process_inbox.py

# Or run directly on an existing session
python src/pipeline.py sessions/<session-id>/audio.m4a

# Re-enrich (skip transcription and diarization)
python src/pipeline.py sessions/<session-id>/ --re-enrich
```

Models download on first run. Supported formats: `.m4a`, `.mp3`, `.wav`.

## Tests

```bash
# Fast tests only (no model loading)
pytest -m "not slow"

# All tests including model integration
pytest
```

## Build Log

Development decisions and discoveries are tracked in the [build log](https://saurinchoksi.com/tell-me-a-story-build-log.html).

## Background

I'm a Sesame Workshop Writers Room fellow. I wrote for Mo Willems. I earned an Emmy nomination for Daniel Tiger's Neighborhood. Most recently, I built tools for the content team at Kibeam Learning, a company that makes interactive products for kids.

More about this project and my other work at [saurinchoksi.com](https://saurinchoksi.com).

## License

MIT
