/**
 * Filter predicates and utility functions for the transcript validator.
 *
 * These mirror the Python predicates in src/filters.py — single-word segment
 * checks for silence gaps, near-zero probability, and duplicate detection.
 */

import type { ValidatorSegment, FilterState, SpeakerIdentification, SegmentId } from '../types';

/** Silence gap: single-word segment where diarization found no speaker. */
export function isSilenceGap(segment: ValidatorSegment): boolean {
  if (segment.words.length !== 1) return false;
  const w = segment.words[0];
  return w._speaker?.label === null && w._speaker?.coverage === 0.0;
}

/** Near-zero probability: single-word segment with prob < 0.01. */
export function isNearZeroProbability(segment: ValidatorSegment): boolean {
  if (segment.words.length !== 1) return false;
  return segment.words[0].probability < 0.01;
}

/**
 * Find duplicate segment IDs — later occurrences of text that appeared earlier.
 * Case-insensitive, trimmed. Returns a Set of segment IDs to mark as duplicates.
 */
export function findDuplicateSegmentIds(segments: ValidatorSegment[]): Set<SegmentId> {
  const seen = new Map<string, SegmentId>();
  const duplicates = new Set<SegmentId>();

  for (const seg of segments) {
    const key = seg.text.trim().toLowerCase();
    if (key === '') continue;

    if (seen.has(key)) {
      duplicates.add(seg.id);
    } else {
      seen.set(key, seg.id);
    }
  }
  return duplicates;
}

/** Get active filter reasons for a segment (only when filter is toggled on). */
export function getFilterReasons(
  segment: ValidatorSegment,
  duplicateIds: Set<SegmentId>,
  filters: FilterState,
): string[] {
  const reasons: string[] = [];
  if (filters.silenceGap && isSilenceGap(segment)) reasons.push('silence-gap');
  if (filters.nearZero && isNearZeroProbability(segment)) reasons.push('near-zero');
  if (filters.duplicates && duplicateIds.has(segment.id)) reasons.push('duplicate');
  return reasons;
}

/** Get all matching filter reasons regardless of toggle state (for badge display). */
export function getAllFilterReasons(
  segment: ValidatorSegment,
  duplicateIds: Set<SegmentId>,
): string[] {
  const reasons: string[] = [];
  if (isSilenceGap(segment)) reasons.push('silence-gap');
  if (isNearZeroProbability(segment)) reasons.push('near-zero');
  if (duplicateIds.has(segment.id)) reasons.push('duplicate');
  return reasons;
}

/** CSS class for word probability coloring: high (>0.7), mid (0.3–0.7), low (<0.3). */
export function getWordProbabilityClass(prob: number): string {
  if (prob >= 0.7) return 'prob-high';
  if (prob >= 0.3) return 'prob-mid';
  return 'prob-low';
}

/** Dominant speaker for a segment — the label with highest coverage among its words. */
export function getDominantSpeaker(segment: ValidatorSegment): string | null {
  const counts = new Map<string, number>();
  for (const w of segment.words) {
    const label = w._speaker?.label;
    if (label) counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  if (counts.size === 0) return null;

  let best = '';
  let bestCount = 0;
  for (const [label, count] of counts) {
    if (count > bestCount) {
      best = label;
      bestCount = count;
    }
  }
  return best;
}

/** Stable speaker class name for CSS coloring (speaker-0, speaker-1, etc). */
export function getSpeakerClass(label: string | null): string {
  if (!label) return 'speaker-unknown';
  const num = parseInt(label.replace('SPEAKER_', ''), 10);
  if (isNaN(num)) return 'speaker-unknown';
  return `speaker-${num % 5}`;
}

/**
 * Build identity-based color map from identification data.
 *
 * All raw labels (SPEAKER_00, SPEAKER_03) that resolve to the same profile
 * get the same CSS class, anchored to the lowest speaker number in the group.
 * Falls back gracefully — unidentified speakers keep their raw-label color.
 */
export function buildSpeakerColorMap(
  identifications: SpeakerIdentification[]
): Map<string, string> {
  // Group speaker keys by resolved identity (profile_id, or raw key if unidentified)
  const identityGroups = new Map<string, string[]>();
  for (const ident of identifications) {
    const identity = ident.profile_id ?? ident.speaker_key;
    const keys = identityGroups.get(identity) ?? [];
    keys.push(ident.speaker_key);
    identityGroups.set(identity, keys);
  }

  // Assign color by lowest speaker number in each group (preserves existing semantics)
  const colorMap = new Map<string, string>();
  for (const [, keys] of identityGroups) {
    const nums = keys
      .map(k => parseInt(k.replace('SPEAKER_', ''), 10))
      .filter(n => !isNaN(n));
    const baseNum = nums.length > 0 ? Math.min(...nums) : 0;
    const cls = `speaker-${baseNum % 5}`;
    for (const key of keys) {
      colorMap.set(key, cls);
    }
  }
  return colorMap;
}
