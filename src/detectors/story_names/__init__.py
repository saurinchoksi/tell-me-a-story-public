"""Offline per-story name engine — one local-Gemma pass per story does improvised-name
inconsistency (M9b) *and* sourced-canon recognition (M9c) together, with a canon shield
that uses the M9b clustering to protect correctly-spelled real names. Graduated from the
sealed EMP Stage-1 v2 probe; the validated logic is ported verbatim into the private
_segment/_audit/_names modules (the family_names.py precedent).

The registered detector (CanonNameDetector) surfaces only the M9c slice of that engine;
M9b is served by the separate m9b-name-consistency detector."""
from detectors.story_names.detector import CanonNameDetector

__all__ = ["CanonNameDetector"]
