"""Failure-mode detector registry.

Detectors are read-only monitors over session transcripts — they emit flags,
never transcript edits. Register new detectors here explicitly.
"""

from detectors.family_names import FamilyNameDetector
from detectors.name_consistency import NameConsistencyDetector
from detectors.story_names import CanonNameDetector

# CanonNameDetector (M9c) is offline_only — scan_session skips it unless a caller passes
# run_offline=True (the CLI --story-names / process_inbox), so registering it here is
# safe: it never runs in a web request, but it auto-surfaces in the Monitor. One detector
# per name-error case: M9a (family), M9b (improvised), M9c (sourced canon).
DETECTORS = [FamilyNameDetector(), NameConsistencyDetector(), CanonNameDetector()]


def get_detector(detector_id: str):
    for det in DETECTORS:
        if det.id == detector_id:
            return det
    raise ValueError(
        f"Unknown detector: {detector_id!r}. Registered: {[d.id for d in DETECTORS]}"
    )
