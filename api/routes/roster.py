"""Roster + home-canon endpoints — the "who exists" page's add-only writes (TMAS-56/57).

The roster answers "which real people exist"; the home cast answers "which of OUR
characters exist". Only explicit human gestures on the page write either (the Name
Book's rule). Every write validates BEFORE saving (src/roster_store.py and
src/home_canon.py fail-loud rules) so a bad entry is refused with a reason, never
persisted. Add-only by design: removals/renames stay rare, deliberate hand edits —
recorded verdicts and book rulings reference forms, and a UI delete would orphan them.

  GET  /api/roster                          the parsed roster (people + aliases)
  POST /api/roster/people                   {canonical, circle?}    add a person
  POST /api/roster/people/<pid>/nicknames   {nickname}              add a name form
  POST /api/roster/people/<pid>/aliases     {token}                 add a distortion
  GET  /api/home-canon                      the registered characters
  POST /api/home-canon/characters           {canonical}             register a character

A successful write changes the store's content hash, so the fixers (and detector
scans, for the roster) refresh themselves on their next run via config fingerprints —
and any re-asked questions the human already answered fold under the spot-answer fold.
"""
from flask import Blueprint, current_app, jsonify, request

import home_canon
import roster_store

bp = Blueprint("roster", __name__)


def _path():
    return current_app.config["ROSTER_PATH"]


def _home_path():
    return current_app.config["HOME_CANON_PATH"]


@bp.route("/home-canon", methods=["GET"])
def get_home_canon():
    return jsonify(home_canon.load(_home_path()))


@bp.route("/home-canon/characters", methods=["POST"])
def add_character():
    body = request.get_json(silent=True) or {}
    try:
        character = home_canon.add_character(body.get("canonical") or "",
                                             path=_home_path(),
                                             roster_path=_path())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "added", "character": character})


@bp.route("/roster", methods=["GET"])
def get_roster():
    return jsonify(roster_store.load(_path()))


@bp.route("/roster/people", methods=["POST"])
def add_person():
    body = request.get_json(silent=True) or {}
    try:
        person = roster_store.add_person(
            body.get("canonical") or "",
            circle=body.get("circle") or "family",
            path=_path())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "added", "person": person})


@bp.route("/roster/people/<person_id>/nicknames", methods=["POST"])
def add_nickname(person_id):
    body = request.get_json(silent=True) or {}
    try:
        person = roster_store.add_nickname(person_id, body.get("nickname") or "",
                                           path=_path())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "added", "person": person})


@bp.route("/roster/people/<person_id>/aliases", methods=["POST"])
def add_alias(person_id):
    body = request.get_json(silent=True) or {}
    try:
        alias = roster_store.add_alias(person_id, body.get("token") or "",
                                       path=_path())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "added", "alias": alias})
