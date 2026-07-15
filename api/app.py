"""Flask application factory for the tell-me-a-story API.

Serves session data, speaker profiles, and audio files.
The frontend (Vite dev server) proxies /api requests here.
"""

import math
import sys
from pathlib import Path

from flask import Flask
from flask.json.provider import DefaultJSONProvider

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "sessions"
DEFAULT_PROFILES_PATH = str(PROJECT_ROOT / "data" / "speaker_profiles.json")

# Add src/ to path so identify.py and profiles.py can be imported by bare name
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))  # api package importable from any cwd


def _json_safe(o):
    """NaN/Infinity -> None, recursively. See JsonSafeProvider."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


class JsonSafeProvider(DefaultJSONProvider):
    """Serialize NaN as null instead of a bare `NaN` token.

    Whisper leaves NaN in a segment's `avg_logprob` when it cannot score a decode —
    it happens on the repetition storms (a child shouting "No!" 15 times). Python's
    json writes that as a literal `NaN`, which is NOT valid JSON, so the browser's
    JSON.parse rejects the ENTIRE response: one unscored segment blanks a whole page,
    and only in the browser (curl and Python both accept NaN, which is why this hid
    for so long). 27 of 111 sessions carry one.

    Not a silent fallback — null is the honest JSON for "no value here", and the
    alternative (allow_nan=False) raises, trading a blank page for a 500. Fixed at the
    provider so every route inherits it, not per-route where the next one would forget.
    """

    def dumps(self, obj, **kwargs):
        return super().dumps(_json_safe(obj), **kwargs)


def create_app(sessions_dir=None, profiles_path=None, detectors=None,
               name_book_dir=None, roster_path=None, home_canon_path=None) -> Flask:
    """Create and configure the Flask app.

    Args:
        sessions_dir: Path to sessions directory. Defaults to PROJECT_ROOT/sessions.
            Injectable for test isolation with temp directories.
        profiles_path: Path to speaker_profiles.json. Defaults to PROJECT_ROOT/data/speaker_profiles.json.
            Injectable for test isolation.
        detectors: Failure-mode detector list for the detections routes.
            Defaults to the src/detectors registry with the roster-reading
            detectors (m9a and m9b) re-bound to this app's roster_path — the
            registry's own instances are wired to the real roster at import
            time — so the path kwargs alone yield an isolated app. Inject a
            list only to use fakes in tests.
        name_book_dir: Path to the Name Book folder (human name rulings, one file per
            world). Defaults to src/namebook.py's data/name-book/. Injectable so tests
            never read or write the real rulings.
        roster_path: Path to the roster file (who exists — people/nicknames/aliases,
            the /roster page's store). Defaults to data/name_roster.json. Injectable
            so tests never touch the real gitignored roster.
        home_canon_path: Path to the home cast (the family's registered characters,
            TMAS-57). Defaults to data/home-canon.json. Injectable for tests.
    """
    app = Flask(__name__)
    app.json = JsonSafeProvider(app)

    app.config["SESSIONS_DIR"] = Path(sessions_dir) if sessions_dir else DEFAULT_SESSIONS_DIR
    app.config["PROFILES_PATH"] = profiles_path or DEFAULT_PROFILES_PATH
    if name_book_dir is None:
        import namebook
        name_book_dir = namebook.BOOK_DIR
    app.config["NAME_BOOK_DIR"] = Path(name_book_dir)
    if roster_path is None:
        import roster_store
        roster_path = roster_store.DEFAULT_ROSTER_PATH
    app.config["ROSTER_PATH"] = Path(roster_path)
    if home_canon_path is None:
        import home_canon
        home_canon_path = home_canon.DEFAULT_PATH
    app.config["HOME_CANON_PATH"] = Path(home_canon_path)
    if detectors is None:
        # Re-bind the roster-reading detectors to this app's roster (TMAS-73);
        # rationale in the docstring above. type(d) preserves any subclass.
        from detectors import DETECTORS
        from detectors.family_names import FamilyNameDetector
        from detectors.name_consistency import NameConsistencyDetector
        detectors = [type(d)(roster_path=app.config["ROSTER_PATH"])
                     if isinstance(d, (FamilyNameDetector, NameConsistencyDetector))
                     else d
                     for d in DETECTORS]
    app.config["DETECTORS"] = detectors

    # CORS — needed when serving production builds (dev uses Vite proxy)
    try:
        from flask_cors import CORS
        CORS(app)
    except ImportError:
        pass  # flask-cors is optional; dev works fine without it via Vite proxy

    # Register blueprints under /api prefix
    from api.routes.sessions import bp as sessions_bp
    from api.routes.profiles import bp as profiles_bp
    from api.routes.audio import bp as audio_bp
    from api.routes.speakers import bp as speakers_bp
    from api.routes.notes import bp as notes_bp
    from api.routes.axial_labels import bp as axial_labels_bp
    from api.routes.detections import bp as detections_bp
    from api.routes.name_corrections import bp as name_corrections_bp
    from api.routes.roster import bp as roster_bp
    from api.routes.missed_speech import bp as missed_speech_bp
    from api.routes.name_pass import bp as name_pass_bp

    app.register_blueprint(sessions_bp, url_prefix="/api")
    app.register_blueprint(profiles_bp, url_prefix="/api")
    app.register_blueprint(audio_bp, url_prefix="/api")
    app.register_blueprint(speakers_bp, url_prefix="/api")
    app.register_blueprint(notes_bp, url_prefix="/api")
    app.register_blueprint(axial_labels_bp, url_prefix="/api")
    app.register_blueprint(detections_bp, url_prefix="/api")
    app.register_blueprint(name_corrections_bp, url_prefix="/api")
    app.register_blueprint(roster_bp, url_prefix="/api")
    app.register_blueprint(missed_speech_bp, url_prefix="/api")
    app.register_blueprint(name_pass_bp, url_prefix="/api")

    return app


if __name__ == "__main__":
    app = create_app()
    print(f"Sessions dir: {app.config['SESSIONS_DIR']}")
    print(f"Profiles path: {app.config['PROFILES_PATH']}")
    app.run(host="localhost", port=5002, debug=True)
