"""Name-pass endpoints — build a per-session gold name key by ear (TMAS-97 step 1).

A thin Flask wrapper over the model-free recall engine in emp/src/name_truth.py:

  GET  /api/sessions/<id>/name-pass   -> the transcript's name candidates as per-name
                                          cards + a flat candidate index for the
                                          highlighter, the M9 coverage guarantee, the
                                          current fingerprint, the category vocabulary,
                                          and the existing key as a cross-check seed.
  POST /api/sessions/<id>/name-truth  -> field-routed save into the key sidecar
                                          (emp/results/visuals/<id>/name-truth.json).

The engine is EMP tooling, so we add emp/src to sys.path here (the app already hosts
eval surfaces like the Validator and /name-review). collect_flags / load_notes /
update_notes are pure JSON + Double Metaphone — no model load — so the GET stays
sub-second. net-vs-ear is DERIVED (the key's names vs these candidates), never stored,
so the key schema stays a pure answer key (category / true_spelling / notes).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from api.helpers import get_session_dir

# The recall engine lives in emp/src, which is not on the app's import path by default.
_EMP_SRC = Path(__file__).resolve().parents[2] / "emp" / "src"
if str(_EMP_SRC) not in sys.path:
    sys.path.insert(0, str(_EMP_SRC))

import name_truth  # noqa: E402

bp = Blueprint("name_pass", __name__)


def _split_occ_id(occ_id: str):
    """occ_id is '<seg_id>:<wi>'. seg_id may be an int id or a string gap id
    ('gap_782.052'), so split on the LAST colon and re-type the seg_id to match a
    transcript segment's `id` field (int stays int, gap id stays str)."""
    seg_str, _, wi_str = occ_id.rpartition(":")
    seg_id = int(seg_str) if seg_str.lstrip("-").isdigit() else seg_str
    return seg_id, int(wi_str)


@bp.route("/sessions/<session_id>/name-pass")
def get_name_pass(session_id: str):
    """Every name candidate the model-free sweep surfaces, as per-name cards, plus a
    flat candidate index (occ_id -> card key) for the transcript highlighter, the M9
    coverage guarantee, the current transcript fingerprint, the category vocabulary,
    and the existing key as a cross-check seed (the prior pass, assumed incomplete)."""
    try:
        get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    seed = name_truth.load_notes(session_id)
    try:
        groups, m9_meta = name_truth.collect_flags(session_id, seed.get("_watch", []))
    except FileNotFoundError:
        return jsonify({"error": f"No transcript for session: {session_id}"}), 404

    # Serialize into cards. NB: collect_flags returns Python sets (surface, detectors,
    # source, m9_meta['coded']) — sort them to lists or jsonify raises.
    cards, candidate_index = [], {}
    for title, rows in groups:
        for r in rows:
            occurrences = []
            for o in r["occ_list"]:
                seg_id, wi = _split_occ_id(o["occ_id"])
                occurrences.append({
                    "occ_id": o["occ_id"],
                    "seg_id": seg_id,
                    "wi": wi,
                    "start": o["start"],
                    "context": o["text"],
                    "cap": o["cap"],
                    "m9_coded": o["m9_coded"],
                    "window": o["window"],
                })
                candidate_index[o["occ_id"]] = r["cleaned"]
            cards.append({
                "cleaned": r["cleaned"],
                "group": title,
                "surfaces": sorted(r["surface"]),
                "detectors": sorted(r["detectors"]),
                "source": sorted(r["source"]),
                "watch": r["watch"],
                "phrase": r.get("phrase", False),
                "cluster": r["cluster"],
                "primary": r["primary"],
                "occurrences": occurrences,
            })

    # Opus's draft spellings (machine scaffolding) — rendered inline; {} when none exist.
    # Missing file = legitimate empty; a corrupt file fails loud (the Fail-Loud convention).
    sug_path = name_truth.suggestions_path(session_id)
    suggestions = json.loads(sug_path.read_text()).get("suggestions", {}) if sug_path.exists() else {}

    return jsonify({
        "session_id": session_id,
        "cards": cards,
        "candidate_index": candidate_index,
        "suggestions": suggestions,
        "m9_meta": {
            "coded": sorted(m9_meta["coded"], key=str),
            "uncovered": m9_meta["uncovered"],
        },
        "fingerprint": name_truth.transcript_fingerprint(session_id),
        "categories": name_truth.CATEGORIES,
        "seed": {
            "items": seed.get("items", {}),
            "occurrences": seed.get("occurrences", {}),
            "watch": seed.get("_watch", []),
            "fingerprint": seed.get("_transcript_fingerprint"),
            "marked_at": seed.get("_marked_at"),
            "heard_confirmed": seed.get("_heard_confirmed"),
        },
    })


@bp.route("/sessions/<session_id>/name-truth", methods=["POST"])
def save_name_truth(session_id: str):
    """Field-routed save into the key sidecar — ported from name_truth.py's --serve
    do_POST (the count.py lesson: write only the changed field so one save never wipes
    a sibling field or Claude's between-turn edits). Every save stamps the transcript
    fingerprint + marked_at, so the first save also backfills the currently-unstamped
    key. Body: {kind:'item'|'occ', id, field, value}."""
    try:
        get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    body = request.get_json(silent=True) or {}
    kind, item_id, field = body.get("kind"), body.get("id"), body.get("field")
    bucket = {"item": ("items", name_truth.ITEM_DEFAULT),
              "occ": ("occurrences", name_truth.OCC_DEFAULT)}.get(kind)
    if not bucket or not item_id or field not in bucket[1]:
        return jsonify({"error": "need {kind:item|occ, id, field, value}"}), 400

    # A category outside the vocabulary corrupts the ruler silently — not blank, so the
    # scorer's `unjudged` guard stays quiet, but in no name-kind bucket either, so the
    # decision disappears from every per-kind table. Same guard as the commit route.
    if field == "category":
        valid = {value for value, _label in name_truth.CATEGORIES}
        cat = body.get("value") or ""
        if cat not in valid:
            return jsonify({"error": f"unknown category {cat!r} — must be one of {sorted(valid)}"}), 400

    # Same reasoning for a scope the key does not know: the scorer reads `scope` to decide
    # WHICH JOB a spot records, and a typo'd one ("tails", "Name") reads as unstamped, so
    # the spot is silently graded as something the ear never said. Blank stays legal — it
    # means unstamped, and the scorer infers it and prints the inference.
    if field == "scope":
        sc = body.get("value") or ""
        if sc and sc not in name_truth.OCC_SCOPES:
            return jsonify({"error": f"unknown scope {sc!r} — must be one of "
                                     f"{sorted(name_truth.OCC_SCOPES)} (or blank)"}), 400

    key, default = bucket

    def mutate(data):
        # `or ""` (not a .get default): an explicit JSON null must also land as "" —
        # a None here would later crash render_html's html.escape.
        data[key].setdefault(item_id, dict(default))[field] = body.get("value") or ""

    # Locked, atomic read-modify-write — serialized so racing per-edit saves can't lose a
    # sibling field or corrupt the key (TMAS-97; the 20260304 loss).
    name_truth.update_notes(session_id, mutate)
    return jsonify({"ok": True})


@bp.route("/sessions/<session_id>/name-truth/commit", methods=["POST"])
def commit_name_truth(session_id: str):
    """Bulk ruling + the ear-gate stamp. Body: {items: {cleaned: {field: value, ...}}, heard?: bool}.
    Sets only the given (valid) item fields — never wipes a sibling. `heard: true` stamps
    `_heard_confirmed` (his attestation that he listened through). Serves both an inline
    correction (one item, provenance 'ear-corrected') and the 'Mark heard' commit (the
    un-corrected suggestions as 'suggestion-confirmed')."""
    try:
        get_session_dir(current_app.config["SESSIONS_DIR"], session_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    body = request.get_json(silent=True) or {}
    items = body.get("items")
    if not isinstance(items, dict):
        return jsonify({"error": "need {items: {cleaned: {field: value}}, heard?: bool}"}), 400

    # Validate every item up front — fail loud, no partial write into the answer key
    # (an empty-string id or an unknown field would pollute what the scorers iterate).
    #
    # Values, not just field NAMES: a `category` outside the vocabulary is the quietest way
    # to corrupt the ruler. It is not blank, so the scorer's `unjudged` guard never fires —
    # but it belongs to neither KINDS nor NEGATIVE_KINDS, so the decision silently vanishes
    # from every per-kind table. Drafts now post a category (TMAS-102), so one typo in a
    # hand-written sidecar ("improvized", "canon") would reach the key. Refuse it here.
    valid_categories = {value for value, _label in name_truth.CATEGORIES}
    for cleaned, fields in items.items():
        if not cleaned or not isinstance(fields, dict):
            return jsonify({"error": "each item needs a non-empty id and a dict of fields"}), 400
        bad = [f for f in fields if f not in name_truth.ITEM_DEFAULT]
        if bad:
            return jsonify({"error": f"unknown field(s): {bad}"}), 400
        cat = fields.get("category")
        if cat is not None and cat not in valid_categories:
            return jsonify({"error": f"unknown category {cat!r} for {cleaned!r} — "
                                     f"must be one of {sorted(valid_categories)}"}), 400

    def mutate(data):
        for cleaned, fields in items.items():
            data["items"].setdefault(cleaned, dict(name_truth.ITEM_DEFAULT)).update(fields)
        if body.get("heard"):
            data["_heard_confirmed"] = datetime.now(timezone.utc).isoformat()

    # Locked, atomic read-modify-write (TMAS-97) — never lose a sibling item to a race.
    name_truth.update_notes(session_id, mutate)
    return jsonify({"ok": True, "committed": len(items)})
