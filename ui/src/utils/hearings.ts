import type { MissedSpeechCandidate, MissedHearing } from '../types';

/**
 * Find the hear-box entry recorded for a candidate (TMAS-75), or null.
 *
 * The span IS the join key: candidates and hearings both originate from the
 * same gap computation, so equality holds bit-for-bit; the epsilon only
 * guards JSON float round-trips. A re-run that shifts a span simply finds no
 * hearing — a new candidate is a new listen. min_gap is deliberately not
 * consulted: spans are computed threshold-independently (src/gaps.py), so
 * the same span at any tier is the same acoustic stretch.
 */
export function findHearing(
  hearings: MissedHearing[],
  candidate: MissedSpeechCandidate,
): MissedHearing | null {
  return (
    hearings.find(
      (h) =>
        Math.abs(h.start - candidate.start) < 1e-6 &&
        Math.abs(h.end - candidate.end) < 1e-6,
    ) ?? null
  );
}
