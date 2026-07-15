#!/usr/bin/env python3
"""namefix — the audio-grounded name stage, running in SHADOW during calibration (2.0.0).

The removed world-blind normalizer guessed "correct" spellings from garbled text and rewrote
the child's word for the heroes as an enemy's name. This stage grounds correction in the
AUDIO instead. As of the calibration reboot (2026-07-02, Choksi's design — see
docs/name-architecture.md), it never writes anything: every opinion it forms becomes a
LABELED QUESTION in sessions/<id>/pending-name-corrections.json, and all transcripts stay
honest/blind until a deliberate apply pass after classes graduate.

The chain (unchanged from the validated 1.x gate):
  1. NAME CANDIDATES from the blind transcript (the M9c card machinery). Not a candidate ->
     never touched. A token already spelled EXACTLY as a cast member is never a candidate
     (the cast-member guard — Thomas must not become Gordon on a neighbor's bleed).
  2. Per story, recognize the WORLD from the name list (abstains on invented/unknown worlds).
  3. Generate the world's correction CAST — characters AND groups, split prompts (worldcast).
  4. Re-transcribe a short clip around each candidate with the world+cast in Whisper's ear.
  5. The gate judges ONLY the nearest non-ordinary re-decoded word at the target's time
     offset; an ordinary-word re-decode never maps to a cast name (father->Vidura leak).

What used to be lanes (auto/queue) is now a LABEL on each group — grouped by why the
machine believed it, so Choksi's verdicts become a dataset that can graduate a class:
  - "sounds-like":  non-word blind token, re-decode matched cast, blind SOUNDS LIKE the
                    suggestion (Bushma->Bhishma). The would-have-auto'd strong class.
  - "audio-only":   same but the blind does NOT sound like the suggestion. Known-mixed:
                    holds both Bandos->Pandavas (right) and Jammus->Thomas (wrong) — the
                    class calibration exists to split, by data, not by guess.
  - "ordinary-word": the blind token is a real English word (arrows, Sir) — never trusted
                    to the machine, always a human call.
Signals logged per group (the flags his verdicts will judge): dm_share, echo_in_sentence
(the suggested name already appears correctly spelled in the same segment), and
n_sibling_spellings (garbles wobble across spellings; invented names hold steady). Each
group carries a stable `gid` — the identity every verdict endpoint and the UI key on
(one heard word can carry two suggestions, so heard alone is not an identity).

STAGE 2 of home canon (TMAS-57, Choksi's go 2026-07-03) lives here too:
  - THE HOME EAR: on a story whose world the recognizer can't place — usually the
    family's own invented story, where this stage previously did NOTHING — the
    registered home cast (home_canon) rides in Whisper's ear instead of nothing, and
    questions come back under world "Home". Same gate, same labels, still shadow.
  - THE HOME GUARD: a token spelled exactly as a REGISTERED home character is never a
    candidate, in ANY world (Jammus must not become James; identity beats sound-alike —
    the cast-member guard extended to the family's own characters).

Two SEQUENTIAL model subprocesses (model_runner.run_model; one model at a time): worker A
loads Qwen3.5 (world + cast per story), worker B loads Whisper (clip re-decodes + gate).
This stage does NOT touch the Name Book at all (post-review tightening, 2026-07-02): the
book acts at VIEW time (api/routes/name_corrections.py folds questions covered by standing
yeses/nos, live) and at the future apply pass — never inside a run. The calibration rule:
a fixer may only add labeled questions; only Choksi's taps write the book, and nothing
writes a transcript until the apply pass.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import home_canon
from commonwords import is_common, wordlist_fingerprint
from detectors.phonetics import clean, codes
from detectors.story_names._audit import story_name_cards
from detectors.story_names._names import story_segments, proper_name_candidates
from detectors.story_names._worker import build_regions

MODEL = "mlx-community/whisper-large-v3-mlx"
LEAD, WINDOW = 1.5, 2.0     # the E4/E5-tuned clip: [start-LEAD, start+WINDOW]
TEMPORAL_TOL = 0.7          # max offset drift between blind word time and its re-decode
PROMPT_CAP = 20            # names primed into Whisper's ear — the appositive listing's hard
                           # cap; the gate's confirmation vocabulary is built from the SAME
                           # capped set (TMAS-67), so it can never confirm a name that was
                           # never in the ear
GROUPS_LEAD = 6            # the validated shape leads with up to this many (the group names,
                           # the highest-value corrections)
VERSION = "2.2.0"           # 2.x: SHADOW — labeled questions only, nothing ever applied
                            # 2.1.0: gid identity; the run no longer reads the Name Book
                            # 2.2.0: home-canon stage 2 — the home ear on unrecognized
                            #        stories + the home guard everywhere (TMAS-57)

# The label vocabulary, in display order — the SINGLE source (the API and the CLI import
# this; the UI keeps only display copy). Standing-ruling folds (covered by a yes / a no)
# are NOT labels: they are view-time partitions computed live against the Name Book.
# "inconsistent-spelling" is the spelling picker's (spellingfix) — an invented name
# spelled several ways in one night, proposing the one spelling to follow the character
# by. Its verdicts route to the Name Book's spellings slot, never to word-level rulings.
SPELLING_LABEL = "inconsistent-spelling"
LABELS = ("sounds-like", "audio-only", "ordinary-word", SPELLING_LABEL)


def config_fingerprint(home_path=None) -> str:
    """Ties a pending file / processing entry to the exact chain config that produced it —
    incl. the common-words list and the registered home cast (stage 2: the home ear and
    the home guard both change what gets asked), so a wordlist change OR a character
    registration invalidates cached decisions. Registration was a one-time errand
    (2026-07-03), so the invalidation cost is paid once, not per-add."""
    import worldcast
    blob = (worldcast.CHARACTERS_PROMPT + worldcast.GROUPS_PROMPT +
            f"{MODEL}|{LEAD}|{WINDOW}|{TEMPORAL_TOL}|{VERSION}|{wordlist_fingerprint()}"
            f"|{home_canon.fingerprint(home_path)}")
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def transcript_fingerprint(transcript: dict) -> str:
    """Word-level fingerprint (mirrors the detectors' convention): a transcript edit stales
    the pending file so the UI can prompt a re-run instead of showing drifted positions."""
    parts = []
    for s in transcript.get("segments", []):
        for w in s.get("words", []):
            parts.append(w.get("word", ""))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def pre_correction_fingerprint(transcript: dict) -> str:
    """transcript_fingerprint of the words as they stood BEFORE any applied correction:
    a corrected word contributes its preserved `_original` token (re-attached to the
    current token's leading whitespace — apply_corrections keeps the leading space and
    stores the original stripped), every untouched word its `word` verbatim. Why a
    reader wants this stream: see _is_stale in api/routes/name_corrections.py."""
    parts = []
    for s in transcript.get("segments", []):
        for w in s.get("words", []):
            token = w.get("word", "")
            if "_original" in w:
                lead = token[:len(token) - len(token.lstrip())]
                token = lead + w["_original"]
            parts.append(token)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


# --------------------------- stage 1: candidates (pure) ---------------------------
def candidate_positions(transcript: dict):
    """Per-story candidate occurrences + per-story name surfaces + the capitalized-in-story
    guard set. Returns (regions_out, singles) where regions_out[i] = {region, names, cands}."""
    seg_by_id = {s["id"]: s for s in transcript["segments"]}
    pos_of = {s["id"]: i for i, s in enumerate(transcript["segments"])}
    singles = set()
    out = []
    for r in build_regions(transcript.get("_stories") or [], pos_of):
        segs = story_segments(transcript, r, pos_of)
        cards = story_name_cards(segs, recover=True)
        singles |= proper_name_candidates(segs)
        names = sorted({s for c in cards for s in c["surface"]})
        cands, seen = [], set()
        for card in cards:
            for o in card["occ"]:
                seg = seg_by_id.get(o["seg_id"])
                words = (seg or {}).get("words") or []
                if o["wi"] >= len(words) or words[o["wi"]].get("start") is None:
                    continue
                k = (o["seg_id"], o["wi"])
                if k in seen:
                    continue
                seen.add(k)
                w = words[o["wi"]]
                cands.append({"story": r["idx"], "start": float(w["start"]),
                              "blind": w["word"].strip(), "seg_id": o["seg_id"], "wi": o["wi"]})
        out.append({"region": r, "names": names, "cands": cands})
    return out, singles


# --------------------------- stage 2+3: worlds + casts (Qwen worker) ---------------------------
def _worlds_and_casts_worker(region_names: list[dict]) -> list[dict]:
    """Subprocess worker A: ONE Qwen3.5 load; per story recognize the world from its name
    list and generate the split cast. Picklable in/out."""
    from qwen35 import make_reader
    from detectors.story_names import _qwen35
    import worldcast
    gen = make_reader()
    out = []
    for rn in region_names:
        world = _qwen35.recognize_world(gen, rn["names"])
        split = worldcast.cached_cast_split(gen, world) if world else {"characters": [], "groups": []}
        out.append({"story": rn["story"], "world": world, "split": split})
    return out


def cast_listing(cast: list[str]) -> tuple[str, list[str]]:
    """The ONE shared cast-listing helper (TMAS-67) — the single place the validated
    appositive shape lives, so build_prompt and build_home_prompt can never drift apart
    again. The shape: the first up-to-GROUPS_LEAD names comma-joined (the group names,
    which carry the highest-value corrections), then ", and " + the remaining primed names
    comma-joined; an empty cast yields an empty listing (no IndexError — matches
    build_prompt's long-standing tolerance, and build_home_prompt's only caller already
    guards on a non-empty cast). The primed set is hard-capped at PROMPT_CAP and returned
    alongside the string so a caller can build the gate's confirmation vocabulary from
    EXACTLY the names Whisper's ear was primed with — past the cap the gate must never
    confirm a name that was never primed. Returns (listing, primed_names)."""
    primed = cast[:PROMPT_CAP]
    groups, rest = primed[:GROUPS_LEAD], primed[GROUPS_LEAD:]
    listing = ", ".join(groups) + (", and " + ", ".join(rest) if rest else "")
    return listing, primed


def build_prompt(world: str, cast: list[str]) -> str:
    """The validated appositive prompt shape: groups lead, then the characters — built
    through the shared `cast_listing` helper (byte-identical to the tuned 1.x prompt)."""
    listing, _ = cast_listing(cast)
    return f"This is a bedtime story about the {world}. Its characters include {listing}."


def build_home_prompt(cast: list[str]) -> str:
    """The home-ear prompt (stage 2): the SAME validated appositive shape as build_prompt —
    now via the shared `cast_listing` helper (TMAS-67), where it used a plain Oxford join
    before — but with NO world clause (the world is unrecognized by definition, and naming
    one would invite Whisper to hallucinate around it). Just the family's own names in the
    ear."""
    listing, _ = cast_listing(cast)
    return f"This is a bedtime story. Its characters include {listing}."


# --------------------------- stage 4+5: re-decode + gate (Whisper worker) ---------------------------
def gate_decision(near_words, blind: str, cast_clean: dict, singles: set) -> dict | None:
    """The gate on one candidate. `near_words` = [(word, offset)] within TEMPORAL_TOL of
    the expected clip offset, sorted nearest-first. Returns None (untouched) or a decision
    {canonical, redecoded, how, action}. Pure — unit-testable without any model. The Name
    Book is deliberately NOT consulted here: rulings act at view time and the apply pass."""
    blind_c = clean(blind)
    # Target = nearest re-decoded word that is either name-shaped (not a common word, per the
    # frequency list — see commonwords.py), capitalized-in-story (`singles`, the recall guard),
    # or an EXACT cast spelling (the very vocabulary we primed Whisper with — belt-and-
    # suspenders now that the frequency list contains no canon names, unlike Webster's-1934,
    # which literally listed "Kauravas"). Sound-alikes get no exemption, which keeps the
    # father(FTR)->Vidura(FTR) leak closed ("father" is common -> skipped).
    target = next(((w, off) for w, off in near_words
                   if clean(w) and (clean(w) in cast_clean or clean(w) in singles
                                    or not is_common(clean(w)))), None)
    if target is None:
        return None
    wc = clean(target[0])
    if not wc or wc == blind_c:
        return None
    if wc in cast_clean:
        canonical, how = cast_clean[wc], "exact"
    else:
        canonical = next((cast_clean[cc] for cc in cast_clean if codes(wc) & codes(cc)), None)
        how = "dm"
        if canonical is None:
            return None
    # Choksi's rule: a real English word in the transcript is never auto-overwritten — it
    # queues for his bless. "Real word" = the frequency list (exact membership, no plural
    # stripping): "arrows"/"beam" queue; "bushma"/"bandos" auto-fix. (His follow-up call,
    # 2026-07-02, replacing Webster's-1934 whose ghosts caused the Bandos/jami bumps.)
    action = "queued" if is_common(blind_c) else "auto"
    return {"canonical": canonical, "redecoded": target[0], "how": how, "action": action}


def _redecode_worker(audio_path: str, jobs: list[dict],
                     singles: list[str]) -> list[dict]:
    """Subprocess worker B: ONE Whisper load; per candidate cut the in-memory clip, re-decode
    with the story's world+cast prompt, run the gate. Picklable in/out."""
    import numpy as np
    import mlx_whisper
    from mlx_whisper.audio import SAMPLE_RATE, load_audio
    audio = np.array(load_audio(audio_path)).astype(np.float32)
    singles = set(singles)
    decisions = []
    for j in jobs:
        p = j["cand"]
        i0 = max(0, int((p["start"] - LEAD) * SAMPLE_RATE))
        i1 = int((p["start"] + WINDOW) * SAMPLE_RATE)
        r = mlx_whisper.transcribe(audio[i0:i1], path_or_hf_repo=MODEL, language="en",
                                   word_timestamps=True, initial_prompt=j["prompt"],
                                   condition_on_previous_text=True, verbose=False)
        words = [((w.get("word") or "").strip(), float(w["start"]))
                 for s in r.get("segments", []) for w in s.get("words", [])
                 if w.get("start") is not None]
        expected = min(p["start"], LEAD)
        near = sorted(((w, off) for w, off in words if abs(off - expected) <= TEMPORAL_TOL),
                      key=lambda t: abs(t[1] - expected))
        d = gate_decision(near, p["blind"], j["cast_clean"], singles)
        if d:
            decisions.append({**p, **d, "world": j["world"]})
    return decisions


# --------------------------- orchestration ---------------------------
def decision_label(d: dict, dm_share: bool) -> str:
    """The calibration label: WHY the machine believed this decision. Replaces the old
    auto/queue lanes — during shadow, a label sorts the question for the human; nothing
    is ever designated for application."""
    if d["action"] == "queued":       # the blind token is a real English word
        return "ordinary-word"
    return "sounds-like" if dm_share else "audio-only"


def group_decisions(decisions: list[dict]) -> list[dict]:
    """Fold per-occurrence decisions into per-(story, heard, suggestion) groups, each
    carrying its calibration label + the dm_share signal + a stable `gid`. The gid IS the
    group's identity — one heard word can carry two suggestions (Jammus->Thomas AND
    Jammus->James in one night), so every verdict endpoint keys on gid, never on heard."""
    groups = {}
    for d in decisions:
        key = (d["story"], clean(d["blind"]), clean(d["canonical"]))
        g = groups.get(key)
        if g is None:
            heard_cleaned = clean(d["blind"])
            dm_share = bool(codes(heard_cleaned) & codes(clean(d["canonical"])))
            g = {"gid": f"{d['story']}:{heard_cleaned}:{clean(d['canonical'])}",
                 "world": d["world"], "story_id": d["story"], "heard": d["blind"].strip(" .,?!"),
                 "heard_cleaned": heard_cleaned,
                 "canonical": d["canonical"], "method": d["how"],
                 "label": decision_label(d, dm_share), "dm_share": dm_share,
                 "occurrences": []}
            groups[key] = g
        g["occurrences"].append({"segment_id": d["seg_id"], "word_index": d["wi"],
                                 "start": d["start"], "token": d["blind"]})
    return list(groups.values())


def annotate_signals(groups: list[dict], transcript: dict) -> list[dict]:
    """The two suspicion flags the calibration data will judge (never acted on in code —
    they ride with each group so Choksi's verdicts can prove or refute them):

    - echo_in_sentence: the suggested name already appears, correctly spelled, at another
      position in an occurrence's segment ("Thomas and Pataki and Jammus..." — the real
      Thomas is RIGHT THERE, so the re-listen was probably echoing it).
    - n_sibling_spellings: how many OTHER spellings in this world sound-alike into the same
      suggestion. Garbles of a real name wobble (Bushma/Bushna/Bishma); an invented name
      holds one steady spelling (Jammus, every time).
    """
    seg_by_id = {s["id"]: s for s in transcript["segments"]}
    for g in groups:
        canon_c = clean(g["canonical"])
        echo = False
        for o in g["occurrences"]:
            # fail loud on a broken occurrence->segment join: a missing id means id drift,
            # the one thing the whole pipeline promises never happens (CLAUDE.md)
            words = seg_by_id[o["segment_id"]]["words"]
            if any(i != o["word_index"] and clean(w.get("word", "")) == canon_c
                   for i, w in enumerate(words)):
                echo = True
                break
        g["echo_in_sentence"] = echo
    # sibling spellings: precompute codes/cleaned-canonical once per group and bucket by
    # (world, canonical) so only same-target spellings are compared (was O(n^2) DM calls)
    pre = [(g, codes(g["heard_cleaned"]), clean(g["canonical"])) for g in groups]
    buckets: dict = {}
    for g, gc, cc in pre:
        buckets.setdefault((g["world"], cc), []).append((g, gc))
    for bucket in buckets.values():
        for g, gc in bucket:
            g["n_sibling_spellings"] = len(
                {g2["heard_cleaned"] for g2, gc2 in bucket
                 if g2["heard_cleaned"] != g["heard_cleaned"] and gc2 & gc})
    return groups


def run_namefix(transcript: dict, audio_path: str, worlds_and_casts=None,
                redecode=None, timeout: int = 3600, home_path=None) -> dict:
    """The full stage. Returns {groups, worlds}. `worlds_and_casts` and `redecode` are
    injectable for tests (fakes); production runs the two model subprocesses. Reads no
    Name Book: rulings act at view time and the apply pass, never inside a run.
    `home_path` overrides the home-canon store (tests); default = the real registry
    (test-safe via the conftest DEFAULT_PATH monkeypatch)."""
    import worldcast
    from model_runner import run_model

    regions, singles = candidate_positions(transcript)
    region_names = [{"story": r["region"]["idx"], "names": r["names"]} for r in regions]

    if worlds_and_casts is None:
        worlds_and_casts = run_model(_worlds_and_casts_worker, region_names, timeout=timeout)
    wac_by_story = {w["story"]: w for w in worlds_and_casts}

    # home_forms (FULL registry) drives the guard — every registered spelling is protected
    # everywhere. The home ear's gate confirms only what actually rides in the ear, so its
    # vocabulary is built from the SAME capped set as the prompt (TMAS-67).
    home_forms = home_canon.canonical_forms(home_path)   # cleaned -> registered spelling
    home_prompt, home_gate_clean = "", {}
    if home_forms:
        home_names = sorted(home_forms.values())
        home_prompt = build_home_prompt(home_names)
        _, home_primed = cast_listing(home_names)
        home_gate_clean = {clean(n): n for n in home_primed if clean(n)}

    jobs, worlds_out = [], []
    for r in regions:
        sid = r["region"]["idx"]
        wac = wac_by_story.get(sid, {"world": "", "split": {"characters": [], "groups": []}})
        world = wac["world"]
        worlds_out.append({"story_id": sid, "title": r["region"].get("title", ""),
                           "saved_world": r["region"].get("world", ""),
                           "recognized_world": world, "n_names": len(r["names"]),
                           "n_candidates": len(r["cands"])})
        if not world:
            # THE HOME EAR (stage 2, TMAS-57): a story the recognizer can't place is
            # usually the family's OWN invented story — exactly where garbles of home
            # characters live. Instead of doing nothing, re-listen with the registered
            # home cast in Whisper's ear; questions come back under world "Home".
            # Same gate, same labels, still shadow. No Qwen involved: the cast comes
            # from the registry, not from world knowledge.
            if home_forms and r["cands"]:
                worlds_out[-1]["home_ear"] = True
                for cand in r["cands"]:
                    if clean(cand["blind"]) in home_forms:
                        continue  # the cast-member guard, home edition (FULL registry)
                    jobs.append({"cand": cand, "world": home_canon.WORLD,
                                 "prompt": home_prompt, "cast_clean": home_gate_clean})
            continue
        cast = worldcast.correction_cast(wac["split"])
        if not cast:
            continue
        cast_clean = {clean(n): n for n in cast if clean(n)}        # FULL cast — the cast-member guard
        prompt = build_prompt(world, cast)
        _, primed = cast_listing(cast)                              # the names actually primed into the ear
        gate_clean = {clean(n): n for n in primed if clean(n)}      # what the gate may confirm (== the prompt cap, TMAS-67)
        for cand in r["cands"]:
            # the cast-member guard: a token already spelled exactly as a cast member is
            # never second-guessed (Thomas must not become Gordon on a neighbor's bleed)
            # + THE HOME GUARD (stage 2): a token spelled exactly as a REGISTERED home
            # character is untouchable in ANY world — never a candidate, so never a
            # correction (Jammus must not become James; identity beats sound-alike)
            bc = clean(cand["blind"])
            if bc in cast_clean or bc in home_forms:
                continue
            jobs.append({"cand": cand, "world": world, "prompt": prompt,
                         "cast_clean": gate_clean})

    if not jobs:
        return {"groups": [], "worlds": worlds_out}

    if redecode is None:
        decisions = run_model(_redecode_worker, audio_path, jobs, sorted(singles),
                              timeout=timeout)
    else:
        decisions = redecode(audio_path, jobs, sorted(singles))

    groups = annotate_signals(group_decisions(decisions), transcript)
    return {"groups": groups, "worlds": worlds_out}


FIXER = "namefix"   # this module's tag in the shared queue file

# Plain words for the skip/error tokens recorded in queue metadata (TMAS-72) — the one
# Python-side vocabulary; terminal surfaces print through it. The UI strip keeps a
# mirrored TS map (NameReview.tsx) with a raw-token fallback for tokens it doesn't know.
SKIP_REASON_COPY = {
    "no_roster": "no roster yet — register names at /roster",
    "no_audio_path": "no audio file",
    "no_stories": "no stories found on the transcript",
    "no_cache_dir": "no session directory",
    "deferred": "deferred — run later via fix_session",
    "error": "the check failed partway — re-run the fixers",
}


def write_queue(session_dir: Path, fixer: str, groups: list, transcript: dict,
                version: str, config_fp: str, worlds: list | None = None,
                skipped_reason: str | None = None) -> Path:
    """Persist ONE fixer's labeled questions into the session's SHARED shadow queue
    (pending-name-corrections.json) — read-merge-write, atomic; the same per-section
    convention detections.json uses per detector.

    Every group is stamped with its `fixer` tag (namefix / family / spelling); a write
    replaces only that fixer's groups and leaves the others standing. The `_blessed`/
    `_rejected` audit lists are human history — ALWAYS carried over.

    `skipped_reason` records a fixer that DIDN'T run (TMAS-72: "checked, nothing found"
    and "never ran" must stay distinguishable at the queue, not just in _processing):
    the fixer's metadata entry becomes {"status": "skipped", "reason": ...} and its
    prior same-fingerprint groups are left standing — they were computed against this
    exact transcript; the skip only means THIS run couldn't re-check. For the same
    reason, a prior same-fingerprint RAN record survives an environment skip untouched:
    that check of this exact transcript still stands (a deleted roster doesn't un-run
    it). A CRASHED run (reason "error") is different — the crash is the newest truth
    and overwrites, so a failed re-check can never hide behind an old green record.
    `groups` is ignored on a skip write. Readers treat an entry without "status" as ran.

    ONE top-level transcript_fingerprint governs the whole file (the API's staleness
    guard and every verdict endpoint read only it). If the transcript changed since the
    prior write, the OTHER fixers' groups were computed against drifted word positions —
    they are dropped, along with their `fixers` metadata entries and the stale worlds
    report. They are machine questions, regenerable by re-running those fixers; keeping
    them under a fresh fingerprint would let verdicts record against drifted spots, the
    exact corruption the staleness guard exists to prevent.

    `worlds` is namefix's world-recognition report; fixers with no world concept pass
    None and the prior report is preserved. Legacy single-fixer files (no `fixers` dict,
    untagged groups) read as namefix's."""
    path = Path(session_dir) / "pending-name-corrections.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    tf = transcript_fingerprint(transcript)

    if prior and prior.get("transcript_fingerprint") != tf:
        kept_groups, fixers_meta, worlds_out = [], {}, (worlds or [])
    else:
        prior_groups = prior.get("groups", [])
        for g in prior_groups:
            g.setdefault("fixer", FIXER)          # legacy files predate the tag
        keep_own = skipped_reason is not None     # a skip re-checks nothing — keep them
        kept_groups = [g for g in prior_groups if keep_own or g["fixer"] != fixer]
        fixers_meta = dict(prior.get("fixers") or {})
        if not fixers_meta and "namefix_version" in prior:
            # legacy metadata lived at the top level and belonged to namefix
            fixers_meta[FIXER] = {"version": prior["namefix_version"],
                                  "config_fingerprint": prior.get("config_fingerprint", ""),
                                  "run_at": prior.get("run_at", "")}
        worlds_out = worlds if worlds is not None else prior.get("worlds", [])

    if skipped_reason is not None:
        if not skipped_reason:
            raise ValueError("skipped_reason must be a non-empty token")
        prior_entry = fixers_meta.get(fixer)
        if prior_entry is None or "status" in prior_entry or skipped_reason == "error":
            # no valid same-fingerprint run on record (or a crash, which must show) —
            # the skip IS the state
            fixers_meta[fixer] = {"status": "skipped", "reason": skipped_reason,
                                  "version": version,
                                  "run_at": datetime.now(timezone.utc).isoformat()}
        groups = []
    else:
        fixers_meta[fixer] = {"version": version, "config_fingerprint": config_fp,
                              "run_at": datetime.now(timezone.utc).isoformat()}
    payload = {
        "_about": "Labeled name questions from the shadow fixers (SHADOW — calibration "
                  "phase). Nothing here has been applied and nothing will be until the "
                  "apply pass; each group carries the fixer that asked it and the label "
                  "for why the machine believed it, with the suspicion signals riding "
                  "along. Human verdicts via the bless/reject endpoints are RECORDED "
                  "(Name Book + the audit lists below), never applied.",
        "transcript_fingerprint": tf,
        "fixers": fixers_meta,
        "worlds": worlds_out,
        "groups": kept_groups + [{**g, "fixer": fixer} for g in groups],
        "_blessed": prior.get("_blessed", []),
        "_rejected": prior.get("_rejected", []),
    }
    from jsonio import atomic_write_json
    atomic_write_json(path, payload)
    return path


def write_pending(session_dir: Path, result: dict, transcript: dict,
                  home_path=None) -> Path:
    """Namefix's own write into the shared queue (result = run_namefix's return).
    `home_path` must match the run's (both default to the real registry)."""
    return write_queue(session_dir, FIXER, result["groups"], transcript,
                       VERSION, config_fingerprint(home_path), worlds=result["worlds"])


def write_skip(session_dir: Path, fixer: str, version: str, transcript: dict,
               reason: str) -> Path:
    """Record that a fixer DIDN'T run this time — a skip ("no_roster", "no_audio_path",
    "no_stories") or a failed run ("error"). The TMAS-72 trace, one call for every
    site; semantics (kept groups, surviving ran records) live in write_queue."""
    return write_queue(session_dir, fixer, [], transcript, version, "",
                       skipped_reason=reason)
