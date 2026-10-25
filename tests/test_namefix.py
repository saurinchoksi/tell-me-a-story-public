"""Unit tests for the SHADOW namefix stage (namefix.py) + the split-prompt cast (worldcast.py).

Fast/pure: fake gen + fake redecode, no models. The gate rules were each bought by a
measured failure in the 2026-07-01 validation (emp/emp.md); the labels + signals + gid are
the calibration reboot (2026-07-02, docs/name-architecture.md) — nothing is ever applied,
and the run never reads the Name Book (rulings act at view time + the apply pass):
  - temporal nearest-non-ordinary-word matching (the neighboring-Krishna leak)
  - ordinary re-decode never maps to cast (father->Vidura)
  - common blind token -> label "ordinary-word" (arrows, Choksi's ear-check)
  - cast-member guard: a correctly-spelled cast member is never a candidate
  - unrecognized world -> the stage does nothing (generality 6/6)
"""
import json

import pytest

import namefix
import worldcast


# ------------------------------ worldcast ------------------------------
def fake_gen_factory(world_answer="Mahabharata"):
    def gen(prompt, max_tokens=256):
        if "named GROUPS" in prompt:
            return "The Pandavas\nKauravas\n"
        if "main characters" in prompt:
            return "Arjuna\nBhima\nBhishma\nKrishna\nDuryodhana\n"
        if "REAL, widely-known" in prompt:
            return "original" if not world_answer else f"World: {world_answer}"
        return ""
    return gen


def test_generate_cast_split_parses_both():
    split = worldcast.generate_cast_split(fake_gen_factory(), "Mahabharata")
    assert split["groups"] == ["Pandavas", "Kauravas"]   # "The " stripped
    assert "Bhishma" in split["characters"]


def test_correction_cast_groups_first_dedup():
    split = {"groups": ["Pandavas", "Kauravas"], "characters": ["Arjuna", "pandavas", "Bhima"]}
    cast = worldcast.correction_cast(split)
    assert cast[:2] == ["Pandavas", "Kauravas"]
    assert "pandavas" not in cast[2:]          # deduped by cleaned form
    assert cast == ["Pandavas", "Kauravas", "Arjuna", "Bhima"]


def test_cached_cast_split_round_trip(tmp_path):
    calls = []
    def counting_gen(prompt, max_tokens=256):
        calls.append(1)
        return fake_gen_factory()(prompt, max_tokens)
    s1 = worldcast.cached_cast_split(counting_gen, "Mahabharata", cache_dir=tmp_path)
    n_after_first = len(calls)
    s2 = worldcast.cached_cast_split(counting_gen, "Mahabharata", cache_dir=tmp_path)
    assert s1 == s2 and len(calls) == n_after_first   # second hit served from cache
    assert (tmp_path / "mahabharata.json").exists()


# ------------------------------ the gate (pure) ------------------------------
CAST_CLEAN = {"pandavas": "Pandavas", "kauravas": "Kauravas", "bhishma": "Bhishma",
              "bhima": "Bhima", "arjuna": "Arjuna", "vidura": "Vidura"}


def test_gate_sound_match_decides_nonword():
    d = namefix.gate_decision([("Bhishma", 1.5)], "Bushma", CAST_CLEAN, set())
    assert d["action"] == "auto" and d["canonical"] == "Bhishma"


def test_gate_common_blind_token_queues():
    # Choksi's ear-check rule: "arrows" is a real word -> always a human call.
    d = namefix.gate_decision([("Kauravas", 1.5)], "arrows", CAST_CLEAN, {"arrows"})
    assert d["action"] == "queued" and d["canonical"] == "Kauravas"


def test_gate_ordinary_redecode_never_maps():
    # father (FTR) sound-matches Vidura (FTR) but is an ordinary word -> skipped -> untouched.
    d = namefix.gate_decision([("father", 1.5)], "Fondos", CAST_CLEAN, set())
    assert d is None


def test_gate_neighboring_name_is_not_the_target():
    # nearest non-ordinary word is the unmapped "Bandhas" -> untouched; the loud "Kauravas"
    # sitting further away must NOT grab the decision (the v2 leak).
    near = [("Bandhas", 1.5), ("Kauravas", 2.1)]
    d = namefix.gate_decision(near, "Bandos", CAST_CLEAN, set())
    assert d is None


def test_gate_unchanged_target_untouched():
    d = namefix.gate_decision([("Bushma", 1.5)], "Bushma", CAST_CLEAN, set())
    assert d is None


# ------------------------------ labels + identity ------------------------------
def test_decision_label_mapping():
    assert namefix.decision_label({"how": "exact", "action": "queued"}, False) == "ordinary-word"
    assert namefix.decision_label({"how": "dm", "action": "auto"}, True) == "sounds-like"
    assert namefix.decision_label({"how": "exact", "action": "auto"}, False) == "audio-only"


def test_labels_registry_is_the_single_source():
    assert namefix.LABELS == ("sounds-like", "audio-only", "ordinary-word",
                              "inconsistent-spelling")
    assert namefix.SPELLING_LABEL == "inconsistent-spelling"


def test_group_identity_gid_separates_same_heard_two_suggestions():
    """One heard word, two suggestions -> TWO groups with distinct gids (the Jammus
    Thomas-vs-James case). gid is the identity every verdict endpoint keys on."""
    decisions = [
        {"story": 2, "world": "T", "blind": "Jammus", "canonical": "Thomas",
         "how": "exact", "action": "auto", "seg_id": 1, "wi": 0, "start": 1.0},
        {"story": 2, "world": "T", "blind": "Jammus", "canonical": "James",
         "how": "exact", "action": "auto", "seg_id": 2, "wi": 0, "start": 2.0},
    ]
    groups = namefix.group_decisions(decisions)
    assert len(groups) == 2
    gids = {g["gid"] for g in groups}
    assert gids == {"2:jammus:thomas", "2:jammus:james"}


# ------------------------------ run_namefix (fakes end-to-end) ------------------------------
def make_transcript():
    def w(word, start):
        return {"word": word, "start": start, "end": start + 0.3, "probability": 0.9}
    segments = [
        {"id": 0, "text": " Tell me about Bushma today.",
         "words": [w("Tell", 0.0), w("me", 0.4), w("about", 0.8), w("Bushma", 1.2), w("today.", 1.6)]},
        {"id": 1, "text": " The Bondos were good.",
         "words": [w("The", 2.0), w("Bondos", 2.4), w("were", 2.8), w("good.", 3.2)]},
    ]
    return {"text": "", "segments": segments,
            "_stories": [{"start_id": 0, "end_id": 1, "world": "", "title": "T"}]}


def fake_wac(world="Mahabharata"):
    return [{"story": 0, "world": world,
             "split": {"characters": ["Bhishma", "Arjuna"], "groups": ["Pandavas", "Kauravas"]}}]


def fake_redecode_factory(mapping):
    """mapping: blind cleaned -> re-decoded word placed at the expected offset."""
    def fake(audio_path, jobs, singles):
        from detectors.phonetics import clean as _c
        out = []
        for j in jobs:
            p = j["cand"]
            re_word = mapping.get(_c(p["blind"]))
            near = [(re_word, min(p["start"], namefix.LEAD))] if re_word else []
            d = namefix.gate_decision(near, p["blind"], j["cast_clean"], set(singles))
            if d:
                out.append({**p, **d, "world": j["world"]})
        return out
    return fake


def groups_by_heard(res):
    return {g["heard_cleaned"]: g for g in res["groups"]}


def test_run_namefix_labels_and_shadow_shape():
    res = namefix.run_namefix(make_transcript(), "unused.m4a",
                              worlds_and_casts=fake_wac(),
                              redecode=fake_redecode_factory({"bushma": "Bhishma",
                                                              "bondos": "Pandavas"}))
    assert set(res.keys()) == {"groups", "worlds"}      # shadow: no auto/pending lanes
    by = groups_by_heard(res)
    # Bushma sounds like Bhishma -> the strong class
    assert by["bushma"]["label"] == "sounds-like" and by["bushma"]["dm_share"] is True
    assert by["bushma"]["gid"] == "0:bushma:bhishma"
    # Bondos does NOT sound like Pandavas -> the shaky class (holds Bandos AND Jammus alike)
    assert by["bondos"]["label"] == "audio-only" and by["bondos"]["dm_share"] is False
    assert res["worlds"][0]["recognized_world"] == "Mahabharata"


def test_run_namefix_unmapped_redecode_untouched():
    res = namefix.run_namefix(make_transcript(), "unused.m4a",
                              worlds_and_casts=fake_wac(),
                              redecode=fake_redecode_factory({"bushma": "Bhishma",
                                                              "bondos": "Bandhas"}))
    by = groups_by_heard(res)
    assert "bondos" not in by                # Bandhas unmapped -> no question at all


def test_run_namefix_unrecognized_world_does_nothing():
    res = namefix.run_namefix(make_transcript(), "unused.m4a",
                              worlds_and_casts=[{"story": 0, "world": "",
                                                 "split": {"characters": [], "groups": []}}],
                              redecode=fake_redecode_factory({"bushma": "Bhishma"}))
    assert res["groups"] == []
    assert res["worlds"][0]["recognized_world"] == ""


def test_run_namefix_cast_member_guard():
    """A blind token already spelled exactly as a cast member is never a candidate — the
    fixer must not propose replacing a correct Thomas with a neighbor's Gordon."""
    t = make_transcript()
    t["segments"][0]["words"][3]["word"] = "Bhishma"
    t["segments"][0]["text"] = " Tell me about Bhishma today."
    res = namefix.run_namefix(t, "unused.m4a", worlds_and_casts=fake_wac(),
                              # a malicious re-decode that would rename it if consulted
                              redecode=fake_redecode_factory({"bhishma": "Arjuna",
                                                              "bondos": "Bandhas"}))
    assert res["groups"] == []


def test_echo_in_sentence_signal():
    """The suggested name already appears correctly spelled in the same segment — the
    Jammus signal ('Thomas and Pataki and Jammus...')."""
    def w(word, start):
        return {"word": word, "start": start, "end": start + 0.3, "probability": 0.9}
    t = {"text": "", "segments": [
        {"id": 0, "text": " Bhishma helped Bushma today.",
         "words": [w("Bhishma", 0.0), w("helped", 0.4), w("Bushma", 0.8), w("today.", 1.2)]},
        {"id": 1, "text": " The Bondos were good.",
         "words": [w("The", 2.0), w("Bondos", 2.4), w("were", 2.8), w("good.", 3.2)]},
    ], "_stories": [{"start_id": 0, "end_id": 1, "world": "", "title": "T"}]}
    res = namefix.run_namefix(t, "unused.m4a", worlds_and_casts=fake_wac(),
                              redecode=fake_redecode_factory({"bushma": "Bhishma",
                                                              "bondos": "Pandavas"}))
    by = groups_by_heard(res)
    assert by["bushma"]["echo_in_sentence"] is True      # real Bhishma is right there
    assert by["bondos"]["echo_in_sentence"] is False


def test_sibling_spellings_signal():
    """Two sound-alike spellings suggesting the same name are siblings (garble wobble);
    an unrelated group is not."""
    decisions = [
        {"story": 0, "world": "W", "blind": "Bushma", "canonical": "Bhishma",
         "how": "dm", "action": "auto", "seg_id": 0, "wi": 0, "start": 0.0},
        {"story": 0, "world": "W", "blind": "Bishma", "canonical": "Bhishma",
         "how": "dm", "action": "auto", "seg_id": 1, "wi": 0, "start": 1.0},
        {"story": 0, "world": "W", "blind": "Urzi", "canonical": "Arjuna",
         "how": "dm", "action": "auto", "seg_id": 2, "wi": 0, "start": 2.0},
    ]
    transcript = {"segments": [{"id": i, "words": []} for i in range(3)]}
    groups = namefix.annotate_signals(namefix.group_decisions(decisions), transcript)
    by = {g["heard_cleaned"]: g for g in groups}
    assert by["bushma"]["n_sibling_spellings"] == 1      # Bishma
    assert by["bishma"]["n_sibling_spellings"] == 1      # Bushma
    assert by["urzi"]["n_sibling_spellings"] == 0


def test_annotate_signals_fails_loud_on_id_drift():
    """A broken occurrence->segment join must raise, never silently compute a signal —
    id drift is the one thing the pipeline promises never happens."""
    decisions = [{"story": 0, "world": "W", "blind": "Bushma", "canonical": "Bhishma",
                  "how": "dm", "action": "auto", "seg_id": 99, "wi": 0, "start": 0.0}]
    with pytest.raises(KeyError):
        namefix.annotate_signals(namefix.group_decisions(decisions), {"segments": []})


def test_write_pending_shape(tmp_path):
    t = make_transcript()
    res = {"worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}], "groups": [
        {"gid": "0:arrows:kauravas", "world": "Mahabharata", "story_id": 0,
         "heard": "arrows", "heard_cleaned": "arrows", "canonical": "Kauravas",
         "method": "exact", "label": "ordinary-word", "dm_share": False,
         "echo_in_sentence": False, "n_sibling_spellings": 0,
         "occurrences": [{"segment_id": 0, "word_index": 1, "start": 2.4, "token": "arrows"}]}]}
    path = namefix.write_pending(tmp_path, res, t)
    data = json.loads(path.read_text())
    assert data["groups"][0]["label"] == "ordinary-word"
    assert data["groups"][0]["gid"] == "0:arrows:kauravas"
    assert data["groups"][0]["fixer"] == "namefix"
    assert data["fixers"]["namefix"]["config_fingerprint"]
    assert data["transcript_fingerprint"]
    assert data["_rejected"] == [] and data["_blessed"] == []
    assert "SHADOW" in data["_about"]


def test_write_pending_carries_human_history(tmp_path):
    """A namefix re-run must never lose the _blessed/_rejected audit lists."""
    t = make_transcript()
    prior = {"groups": [], "worlds": [],
             "_blessed": [{"heard_cleaned": "bandos", "canonical": "Pandavas"}],
             "_rejected": [{"heard_cleaned": "jammus", "canonical": "James"}]}
    (tmp_path / "pending-name-corrections.json").write_text(json.dumps(prior))
    res = {"groups": [],
           "worlds": [{"story_id": 0, "recognized_world": "Mahabharata"}]}
    data = json.loads(namefix.write_pending(tmp_path, res, t).read_text())
    assert data["_blessed"][0]["heard_cleaned"] == "bandos"
    assert data["_rejected"][0]["heard_cleaned"] == "jammus"


# ------------------------------ the shared multi-fixer queue ------------------------------
def _group(gid, heard, canonical, **kw):
    hc = heard.lower()
    return {"gid": gid, "world": kw.get("world", "Mahabharata"), "story_id": 0,
            "heard": heard, "heard_cleaned": hc, "canonical": canonical,
            "method": "dm", "label": kw.get("label", "sounds-like"), "dm_share": True,
            "echo_in_sentence": False, "n_sibling_spellings": 0,
            "occurrences": [{"segment_id": 0, "word_index": 0, "start": 0.0, "token": heard}]}


def test_write_queue_merges_across_fixers(tmp_path):
    """Each fixer replaces only its OWN groups; the others' stand (the detections.json
    per-section convention). Metadata is per fixer."""
    t = make_transcript()
    namefix.write_queue(tmp_path, "namefix", [_group("0:bushma:bhishma", "Bushma", "Bhishma")],
                        t, "2.1.0", "nf-fp", worlds=[{"story_id": 0}])
    namefix.write_queue(tmp_path, "family", [_group("family:marta:marta", "Martha", "Marta")],
                        t, "1.0.0", "fam-fp")
    data = json.loads((tmp_path / "pending-name-corrections.json").read_text())
    assert {g["fixer"] for g in data["groups"]} == {"namefix", "family"}
    assert data["fixers"]["namefix"]["config_fingerprint"] == "nf-fp"
    assert data["fixers"]["family"]["config_fingerprint"] == "fam-fp"
    assert data["worlds"] == [{"story_id": 0}]   # family passed None — report preserved

    # a namefix re-run replaces namefix's group, keeps family's
    namefix.write_queue(tmp_path, "namefix", [_group("0:bondos:pandavas", "Bondos", "Pandavas")],
                        t, "2.1.0", "nf-fp2", worlds=[])
    data = json.loads((tmp_path / "pending-name-corrections.json").read_text())
    by_fixer = {g["fixer"]: g for g in data["groups"]}
    assert by_fixer["namefix"]["gid"] == "0:bondos:pandavas"
    assert by_fixer["family"]["gid"] == "family:marta:marta"
    assert data["fixers"]["namefix"]["config_fingerprint"] == "nf-fp2"


def test_write_queue_drops_other_fixers_on_transcript_change(tmp_path):
    """A transcript change means the other fixers' occurrence positions drifted — their
    groups and metadata are dropped (regenerable), never kept under a fresh fingerprint
    where verdicts could record against drifted spots. Audits still carried."""
    t1 = make_transcript()
    namefix.write_queue(tmp_path, "namefix", [_group("0:bushma:bhishma", "Bushma", "Bhishma")],
                        t1, "2.1.0", "nf-fp", worlds=[{"story_id": 0}])
    path = tmp_path / "pending-name-corrections.json"
    data = json.loads(path.read_text())
    data["_blessed"] = [{"gid": "0:x:y", "heard_cleaned": "x"}]
    path.write_text(json.dumps(data))

    t2 = json.loads(json.dumps(t1))
    t2["segments"][0]["words"][0]["word"] = " Changed"
    namefix.write_queue(tmp_path, "family", [_group("family:marta:marta", "Martha", "Marta")],
                        t2, "1.0.0", "fam-fp")
    data = json.loads(path.read_text())
    assert [g["fixer"] for g in data["groups"]] == ["family"]
    assert "namefix" not in data["fixers"]
    assert data["worlds"] == []
    assert data["_blessed"][0]["heard_cleaned"] == "x"   # human history survives


def test_write_queue_reads_legacy_file_as_namefix(tmp_path):
    """Pre-multi-fixer files (untagged groups, top-level metadata) read as namefix's:
    another fixer's write must keep their groups and lift the metadata into `fixers`."""
    t = make_transcript()
    legacy = {"namefix_version": "2.1.0", "config_fingerprint": "legacy-fp",
              "run_at": "2026-07-02T00:00:00+00:00",
              "transcript_fingerprint": namefix.transcript_fingerprint(t),
              "worlds": [{"story_id": 0}],
              "groups": [_group("0:bushma:bhishma", "Bushma", "Bhishma")],
              "_blessed": [], "_rejected": []}
    path = tmp_path / "pending-name-corrections.json"
    path.write_text(json.dumps(legacy))
    namefix.write_queue(tmp_path, "family", [_group("family:marta:marta", "Martha", "Marta")], t,
                        "1.0.0", "fam-fp")
    data = json.loads(path.read_text())
    by_fixer = {g["fixer"]: g for g in data["groups"]}
    assert by_fixer["namefix"]["gid"] == "0:bushma:bhishma"
    assert data["fixers"]["namefix"]["config_fingerprint"] == "legacy-fp"
    assert data["fixers"]["family"]["config_fingerprint"] == "fam-fp"


def test_write_skip_records_reason(tmp_path):
    """A fixer that DIDN'T run leaves a trace: its metadata entry says
    skipped + why, while another fixer's same-fingerprint groups and ran record
    stand untouched."""
    t = make_transcript()
    namefix.write_queue(tmp_path, "namefix", [_group("0:bushma:bhishma", "Bushma", "Bhishma")],
                        t, "2.1.0", "nf-fp", worlds=[{"story_id": 0}])
    namefix.write_skip(tmp_path, "family", "1.0.0", t, "no_roster")
    data = json.loads((tmp_path / "pending-name-corrections.json").read_text())
    fam = data["fixers"]["family"]
    assert fam["status"] == "skipped" and fam["reason"] == "no_roster"
    assert "config_fingerprint" not in fam   # a skip has no config to fingerprint
    assert "status" not in data["fixers"]["namefix"]   # namefix's ran record untouched
    assert [g["gid"] for g in data["groups"]] == ["0:bushma:bhishma"]


def test_write_skip_never_clobbers_a_same_fingerprint_ran_record(tmp_path):
    """A deleted roster (or missing audio) must not relabel an already-checked session
    'never ran': the prior run checked THIS exact transcript, so its ran record — and
    its groups — survive a later skip write (the review page keeps saying checked)."""
    t = make_transcript()
    namefix.write_queue(tmp_path, "family", [_group("family:marta:marta", "Martha", "Marta")],
                        t, "1.0.0", "fam-fp")
    namefix.write_skip(tmp_path, "family", "1.0.0", t, "no_roster")
    data = json.loads((tmp_path / "pending-name-corrections.json").read_text())
    fam = data["fixers"]["family"]
    assert "status" not in fam and fam["config_fingerprint"] == "fam-fp"
    assert [g["gid"] for g in data["groups"]] == ["family:marta:marta"]


def test_write_skip_error_overwrites_a_ran_record(tmp_path):
    """A CRASHED run is the newest truth: unlike an environment skip, reason 'error'
    replaces a same-fingerprint ran record — a failed re-check (e.g. after a roster
    edit) must never hide behind an old green record. Groups still stand."""
    t = make_transcript()
    namefix.write_queue(tmp_path, "family", [_group("family:marta:marta", "Martha", "Marta")],
                        t, "1.0.0", "fam-fp")
    namefix.write_skip(tmp_path, "family", "1.0.0", t, "error")
    data = json.loads((tmp_path / "pending-name-corrections.json").read_text())
    fam = data["fixers"]["family"]
    assert fam["status"] == "skipped" and fam["reason"] == "error"
    assert [g["gid"] for g in data["groups"]] == ["family:marta:marta"]


def test_write_skip_on_changed_transcript_drops_drifted_groups(tmp_path):
    """The staleness rule outranks everything: a skip write against a CHANGED
    transcript re-stamps the fingerprint, so drifted groups AND the old ran record
    drop as always — the skip entry is then the honest state."""
    t1 = make_transcript()
    namefix.write_queue(tmp_path, "family", [_group("family:marta:marta", "Martha", "Marta")],
                        t1, "1.0.0", "fam-fp")
    t2 = json.loads(json.dumps(t1))
    t2["segments"][0]["words"][0]["word"] = " Changed"
    namefix.write_skip(tmp_path, "family", "1.0.0", t2, "no_roster")
    data = json.loads((tmp_path / "pending-name-corrections.json").read_text())
    assert data["groups"] == []
    assert data["fixers"]["family"]["status"] == "skipped"
    assert data["transcript_fingerprint"] == namefix.transcript_fingerprint(t2)


# ------------------------------ the frequency wordlist ------------------------------
def test_commonwords_kills_the_websters_ghosts():
    """The three Webster's-1934 failures, pinned against the new frequency list:
    real words are common; archaic ghosts and canon names are not."""
    from commonwords import is_common
    for real in ("arrows", "beam", "father", "wars"):
        assert is_common(real), f"{real} should be protected"
    for ghost in ("bando", "jami", "kauravas", "bushma", "bandos", "bheem"):
        assert not is_common(ghost), f"{ghost} must not be falsely protected"


def test_gate_bandos_decides_without_dictionary():
    # Under Webster's, "Bandos" was falsely protected (bando ghost). Under the frequency
    # list the gate reaches a decision on the sound/cast match — in shadow it becomes an
    # audio-only QUESTION, never an applied fix.
    d = namefix.gate_decision([("Pandavas", 1.5)], "Bandos", CAST_CLEAN, set())
    assert d["action"] == "auto" and d["canonical"] == "Pandavas"


# ------------------------------ home-canon stage 2 ------------------------------
# The home ear: on an unrecognized story, the registered home cast rides in Whisper's
# ear instead of nothing. The home guard: a token spelled exactly as a registered home
# character is never a candidate, in ANY world. Both shadow, both label-identical.

def home_file(tmp_path, *names):
    """Registry fixture with ids derived by the PRODUCTION rule (home_canon.
    add_character's slug), so the fixture can't drift from real registered data."""
    import re
    p = tmp_path / "home-canon.json"
    p.write_text(json.dumps({"characters": [
        {"id": re.sub(r"[^a-z0-9]+", "-", n.lower()).strip("-"), "canonical": n}
        for n in names]}))
    return p


def make_home_transcript():
    def w(word, start):
        return {"word": word, "start": start, "end": start + 0.3, "probability": 0.9}
    segments = [
        {"id": 0, "text": " Then Kadija flew up high.",
         "words": [w("Then", 0.0), w("Kadija", 0.4), w("flew", 0.8), w("up", 1.2), w("high.", 1.6)]},
    ]
    return {"text": "", "segments": segments,
            "_stories": [{"start_id": 0, "end_id": 0, "world": "", "title": "K"}]}


UNRECOGNIZED = [{"story": 0, "world": "", "split": {"characters": [], "groups": []}}]


def test_home_ear_on_unrecognized_story(tmp_path):
    """The night the stage used to skip entirely: home cast in the ear, questions come
    back under world 'Home' with the registered spelling as the suggestion."""
    res = namefix.run_namefix(make_home_transcript(), "unused.m4a",
                              worlds_and_casts=UNRECOGNIZED,
                              redecode=fake_redecode_factory({"kadija": "Khadija"}),
                              home_path=home_file(tmp_path, "Khadija", "Soro"))
    by = groups_by_heard(res)
    g = by["kadija"]
    assert g["world"] == "Home" and g["canonical"] == "Khadija"
    assert g["gid"] == "0:kadija:khadija"
    assert g["label"] == "sounds-like" and g["dm_share"] is True   # still shadow, still labeled
    assert res["worlds"][0]["recognized_world"] == ""              # the report stays honest
    assert res["worlds"][0]["home_ear"] is True


def test_home_ear_empty_registry_still_does_nothing(tmp_path):
    """No registered characters -> the unrecognized branch behaves exactly as before."""
    res = namefix.run_namefix(make_home_transcript(), "unused.m4a",
                              worlds_and_casts=UNRECOGNIZED,
                              redecode=fake_redecode_factory({"kadija": "Khadija"}),
                              home_path=tmp_path / "absent.json")
    assert res["groups"] == []
    assert "home_ear" not in res["worlds"][0]


def test_home_ear_cast_member_guard(tmp_path):
    """A correctly-spelled home character on an unrecognized night is never a candidate."""
    t = make_home_transcript()
    t["segments"][0]["words"][1]["word"] = "Khadija"
    t["segments"][0]["text"] = " Then Khadija flew up high."
    res = namefix.run_namefix(t, "unused.m4a", worlds_and_casts=UNRECOGNIZED,
                              # malicious: would rename her if the guard ever slipped
                              redecode=fake_redecode_factory({"khadija": "Soro"}),
                              home_path=home_file(tmp_path, "Khadija", "Soro"))
    assert res["groups"] == []


def test_home_guard_in_recognized_world(tmp_path):
    """The Jammus->James class killed by identity: a token spelled exactly as a
    REGISTERED home character is untouchable even inside a recognized world."""
    t = make_transcript()
    t["segments"][1]["words"][1]["word"] = "Jammus"
    t["segments"][1]["text"] = " The Jammus were good."
    malicious = fake_redecode_factory({"jammus": "Arjuna"})
    # without the registration the gate happily maps it (the pre-stage-2 world)...
    res_before = namefix.run_namefix(t, "unused.m4a", worlds_and_casts=fake_wac(),
                                     redecode=malicious,
                                     home_path=tmp_path / "absent.json")
    assert "jammus" in groups_by_heard(res_before)
    # ...with it, the home guard removes the candidate entirely
    res_after = namefix.run_namefix(t, "unused.m4a", worlds_and_casts=fake_wac(),
                                    redecode=malicious,
                                    home_path=home_file(tmp_path, "Jammus"))
    assert "jammus" not in groups_by_heard(res_after)


def test_config_fingerprint_tracks_home_registry(tmp_path):
    """Registering a character invalidates cached namefix decisions (the one-time cost
    accepted with home canon); an absent registry fingerprints stably."""
    absent = tmp_path / "absent.json"
    fp_absent = namefix.config_fingerprint(absent)
    assert fp_absent == namefix.config_fingerprint(absent)
    fp_registered = namefix.config_fingerprint(home_file(tmp_path, "Khadija"))
    assert fp_registered != fp_absent


def test_build_home_prompt_shape():
    """No world clause (nothing to hallucinate around) — just the names in the ear.
    build_home_prompt now routes through the SHARED cast_listing helper (the
    validated groups-lead + rest shape), NOT a plain Oxford join. At <= GROUPS_LEAD names
    the shape is a plain comma join with no ', and' — the concrete change to the home ear."""
    p = namefix.build_home_prompt(["Khadija", "Marley", "Soro"])
    assert p == "This is a bedtime story. Its characters include Khadija, Marley, Soro."
    assert namefix.build_home_prompt(["Soro"]).endswith("include Soro.")
    assert "about the" not in p
    # exactly GROUPS_LEAD (6) names: still no ', and' (all sit in the groups block)
    six = ["Ana", "Bel", "Cy", "Dot", "Em", "Fox"]
    assert namefix.build_home_prompt(six) == (
        "This is a bedtime story. Its characters include Ana, Bel, Cy, Dot, Em, Fox.")
    # past GROUPS_LEAD: the ', and' sits between the 6-name groups block and the rest
    assert namefix.build_home_prompt(six + ["Gus", "Hana"]) == (
        "This is a bedtime story. Its characters include Ana, Bel, Cy, Dot, Em, Fox, and Gus, Hana.")


def test_build_home_prompt_empty_cast_no_indexerror():
    """The confirmed edge case: a module-public build_home_prompt on an empty
    cast used to raise IndexError; through the shared helper it yields an empty listing
    gracefully, matching build_prompt's long-standing tolerance."""
    assert namefix.build_home_prompt([]) == "This is a bedtime story. Its characters include ."


def _legacy_build_prompt(world, cast):
    """The original build_prompt, verbatim — the reference the shared-helper version
    must stay byte-identical to for every input."""
    groups, rest = cast[:6], cast[6:20]
    return (f"This is a bedtime story about the {world}. Its characters include "
            + ", ".join(groups) + (", and " + ", ".join(rest) if rest else "") + ".")


@pytest.mark.parametrize("n", [0, 1, 2, 5, 6, 7, 8, 14, 19, 20, 21, 25, 30])
def test_build_prompt_byte_identical_to_legacy(n):
    """PIN: extracting the shared cast_listing helper must not change build_prompt's output
    for ANY input — the validated (E4/E5-tuned) prompt is frozen. Checked against a verbatim
    copy of the old implementation across the cap boundary."""
    cast = [f"Name{chr(65 + i)}{i}" for i in range(n)]   # distinct fictional names
    assert namefix.build_prompt("Kingdom of Eldol", cast) == _legacy_build_prompt(
        "Kingdom of Eldol", cast)


def test_build_prompt_exact_strings_pinned():
    """PIN the exact validated shape at representative sizes: groups+rest, the <=6 no-'and'
    case, exactly 20, over 20 (capped), one name, empty."""
    g = ["Ash", "Bru", "Cle", "Dov", "Elm", "Fig", "Gwen", "Holt"]   # 8: 6 groups + 2 rest
    assert namefix.build_prompt("Vale", g) == (
        "This is a bedtime story about the Vale. Its characters include "
        "Ash, Bru, Cle, Dov, Elm, Fig, and Gwen, Holt.")
    assert namefix.build_prompt("Vale", g[:5]) == (   # <=6 -> plain comma join, no ', and'
        "This is a bedtime story about the Vale. Its characters include Ash, Bru, Cle, Dov, Elm.")
    twenty = [f"C{chr(65 + i)}{i}" for i in range(20)]
    twenty_out = namefix.build_prompt("Vale", twenty)
    assert namefix.build_prompt("Vale", twenty + ["Overflow", "Excess"]) == twenty_out  # capped at 20
    assert "Overflow" not in twenty_out and "Excess" not in twenty_out
    assert namefix.build_prompt("Vale", ["Solo"]) == (
        "This is a bedtime story about the Vale. Its characters include Solo.")
    assert namefix.build_prompt("Vale", []) == (
        "This is a bedtime story about the Vale. Its characters include .")


def test_cast_listing_is_the_one_shared_shape():
    """Both prompt builders route through cast_listing, so their listing portions are
    identical for the same cast — the whole point of the unification. cast_listing
    also returns the capped primed set for the gate."""
    cast = ["Ash", "Bru", "Cle", "Dov", "Elm", "Fig", "Gwen", "Holt"]
    listing, primed = namefix.cast_listing(cast)
    assert listing == "Ash, Bru, Cle, Dov, Elm, Fig, and Gwen, Holt"
    assert primed == cast                                   # under the cap -> unchanged
    assert namefix.build_prompt("Vale", cast).endswith(f"include {listing}.")
    assert namefix.build_home_prompt(cast).endswith(f"include {listing}.")
    # the cap lives here, once: past PROMPT_CAP names are not primed and not listed
    over = [f"N{chr(65 + i)}{i}" for i in range(namefix.PROMPT_CAP + 5)]
    listing_over, primed_over = namefix.cast_listing(over)
    assert len(primed_over) == namefix.PROMPT_CAP
    assert primed_over == over[:namefix.PROMPT_CAP]
    for beyond in over[namefix.PROMPT_CAP:]:
        assert beyond not in listing_over


def test_run_namefix_gate_vocabulary_capped_to_primed():
    """ past PROMPT_CAP cast members, the gate's confirmation vocabulary (the
    cast_clean handed to each re-decode job) is EXACTLY the primed names — never a name
    that was never in Whisper's ear. The prompt is capped to the same set, so cap == gate.
    (The cast-member guard keeps the full cast; it is applied before the job is formed.)"""
    split = {"groups": ["Volens", "Drakes", "Mirrows", "Thornes", "Kestrels", "Wolders"],
             "characters": ["Aldric", "Bevin", "Corin", "Doran", "Elowen", "Fenwick",
                            "Garrick", "Halvard", "Isolde", "Joran", "Kaelin", "Lorcan",
                            "Merrin", "Nyla", "Ophira", "Perrin"]}   # 6 + 16 = 22 names
    cast = worldcast.correction_cast(split)
    assert len(cast) == 22
    _, primed = namefix.cast_listing(cast)
    assert len(primed) == 20 and cast[20:] == ["Ophira", "Perrin"]   # past-cap names

    seen = {}
    def spy_redecode(audio_path, jobs, singles):
        assert jobs, "expected at least one re-decode job"
        for j in jobs:
            seen["cast_clean"] = j["cast_clean"]
            seen["prompt"] = j["prompt"]
        return []
    namefix.run_namefix(make_transcript(), "unused.m4a",
                        worlds_and_casts=[{"story": 0, "world": "Realm", "split": split}],
                        redecode=spy_redecode)
    from detectors.phonetics import clean as _c
    assert seen["cast_clean"] == {_c(n): n for n in primed if _c(n)}
    assert _c("Nyla") in seen["cast_clean"]                 # position 19: last primed, confirmable
    assert _c("Ophira") not in seen["cast_clean"]           # position 20: past cap, NOT confirmable
    assert _c("Perrin") not in seen["cast_clean"]           # position 21: past cap, NOT confirmable
    assert "Ophira" not in seen["prompt"] and "Perrin" not in seen["prompt"]   # nor primed
