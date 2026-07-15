/**
 * SegmentCard — renders a single transcript segment with badges and word spans.
 *
 * Gap segments (_source === 'diarization_gap') get distinct rendering.
 * React.memo with custom comparator keeps this frozen during audio playback —
 * the .active class is applied imperatively by ValidatorPage for 60fps sync.
 */

import { memo, type RefCallback } from 'react';
import type { ValidatorSegment, AxialCode, SegmentId } from '../types';
import { getDominantSpeaker, getSpeakerClass } from '../utils/filters';
import { formatTime } from '../utils/time';
import WordSpan from './WordSpan';
import './SegmentCard.css';

/**
 * The 10 EMP failure modes + NotA. Order matches the chip-row layout (1..9, M10, N/A).
 * Tooltip text mirrors the mode names defined in emp.md. M10 is click-only — the
 * digit keys 1–9 and 0 are already taken (see useKeyboardShortcuts).
 */
const AXIAL_CHIPS: Array<{ code: AxialCode; label: string; title: string }> = [
  { code: 'M1', label: '1', title: 'M1 — Wrong words on real speech' },
  { code: 'M2', label: '2', title: 'M2 — Words on silence or noise' },
  { code: 'M3', label: '3', title: 'M3 — Missed real speech' },
  { code: 'M4', label: '4', title: 'M4 — Wrong speaker' },
  { code: 'M5', label: '5', title: 'M5 — Overlapping speech' },
  { code: 'M6', label: '6', title: 'M6 — Wrong segment boundaries' },
  { code: 'M7', label: '7', title: 'M7 — Word at the wrong timestamp' },
  { code: 'M8', label: '8', title: 'M8 — Non-speech marked as unintelligible' },
  { code: 'M9', label: '9', title: 'M9 — Name mistranscription' },
  { code: 'M10', label: '10', title: 'M10 — Broken-Whisper stretch (one event per filler-loop region)' },
  { code: 'NotA', label: 'N/A', title: 'None of the above (clean segment or unmapped failure)' },
];

interface SegmentCardProps {
  segment: ValidatorSegment;
  index: number;
  isFiltered: boolean;
  filterReasons: string[];
  allFilterReasons: string[];
  hasNotes: boolean;
  axialCodes: AxialCode[];
  onToggleAxialLabel: (segmentId: SegmentId, code: AxialCode) => void;
  speakerNames: Map<string, string>;
  speakerColorMap: Map<string, string>;
  onSeek: (time: number) => void;
  onPlayWord: (start: number, end: number) => void;
  onContextMenu: (e: React.MouseEvent, segmentIndex: number, wordIndex: number) => void;
  onHoverRange?: (start: number, end: number) => void;
  onHoverEnd?: () => void;
  cardRef: RefCallback<HTMLDivElement>;
}

function SegmentCardInner({
  segment,
  index,
  isFiltered,
  filterReasons,
  allFilterReasons,
  hasNotes,
  axialCodes,
  onToggleAxialLabel,
  speakerNames,
  speakerColorMap,
  onSeek,
  onPlayWord,
  onContextMenu,
  onHoverRange,
  onHoverEnd,
  cardRef,
}: SegmentCardProps) {
  const isGap = segment._source === 'diarization_gap';
  const dominant = getDominantSpeaker(segment);
  const displayName = dominant ? (speakerNames.get(dominant) ?? dominant) : null;
  const isIdentified = dominant !== null && speakerNames.has(dominant);
  const speakerClass = (dominant && speakerColorMap.get(dominant)) ?? getSpeakerClass(dominant);
  const duration = (segment.end - segment.start).toFixed(2);

  return (
    <div
      ref={cardRef}
      className={[
        'segment-card',
        isGap ? 'segment-gap' : '',
        isFiltered ? 'filtered' : '',
        axialCodes.length > 0 ? 'segment-labeled' : 'segment-unlabeled',
        speakerClass,
      ].filter(Boolean).join(' ')}
      data-segment-index={index}
      data-segment-id={segment.id}
    >
      {/* Speaker name — prominent top-left label */}
      {displayName && (
        <div
          className={`segment-speaker-name ${speakerClass} ${isIdentified ? 'speaker-identified' : 'speaker-raw'}`}
          title={isIdentified && dominant ? dominant : undefined}
        >
          {displayName}
        </div>
      )}

      {/* Header row — clickable to seek, hoverable to highlight on waveform */}
      <div
        className="segment-header"
        onClick={() => onSeek(segment.start)}
        onContextMenu={(e) => onContextMenu(e, index, -1)}
        onMouseEnter={() => onHoverRange?.(segment.start, segment.end)}
        onMouseLeave={() => onHoverEnd?.()}
      >
        <span className="segment-id">
          {isGap ? `Gap ${segment.id}` : `Segment ${segment.id}`}
        </span>
        <span className="segment-time">
          {formatTime(segment.start)} — {formatTime(segment.end)} ({duration}s)
        </span>
      </div>

      {/* Badge row */}
      <div className="segment-badges">
        {!isGap && (
          <>
            <span className="badge badge-temp" title="Temperature">
              T: {segment.temperature?.toFixed(2) ?? '—'}
            </span>
            <span className="badge badge-cr" title="Compression Ratio">
              CR: {segment.compression_ratio?.toFixed(2) ?? '—'}
            </span>
          </>
        )}
        {hasNotes && <span className="badge badge-note" title="Has notes">Note</span>}
        {allFilterReasons.map((r) => (
          <span
            key={r}
            className={`badge badge-filter ${filterReasons.includes(r) ? 'badge-filter-active' : ''}`}
            title={r}
          >
            {r}
          </span>
        ))}
      </div>

      {/* Axial-code chip row — multi-select; NotA is mutually exclusive with M codes (enforced in reducer) */}
      <div className="segment-axial-chips" aria-label="Axial codes">
        {AXIAL_CHIPS.map(({ code, label, title }) => {
          const selected = axialCodes.includes(code);
          return (
            <button
              key={code}
              type="button"
              aria-pressed={selected}
              className={`axial-chip ${selected ? 'axial-chip-selected' : ''} axial-chip-${code.toLowerCase()}`}
              title={title}
              onClick={() => onToggleAxialLabel(segment.id, code)}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Content */}
      {isGap ? (
        <div className="segment-gap-body">
          <span className="gap-label">[unintelligible]</span>
          <span className="gap-duration">{duration}s</span>
        </div>
      ) : (
        <div className="segment-words">
          {segment.words.map((w, wi) => (
            <WordSpan
              key={wi}
              word={w}
              segmentIndex={index}
              wordIndex={wi}
              dominantSpeaker={dominant}
              onSeek={onSeek}
              onPlayWord={onPlayWord}
              onContextMenu={onContextMenu}
              onHoverRange={onHoverRange}
              onHoverEnd={onHoverEnd}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function areEqual(prev: SegmentCardProps, next: SegmentCardProps): boolean {
  return (
    prev.segment.id === next.segment.id &&
    prev.isFiltered === next.isFiltered &&
    prev.hasNotes === next.hasNotes &&
    prev.axialCodes.length === next.axialCodes.length &&
    prev.axialCodes.every((c, i) => c === next.axialCodes[i]) &&
    prev.onToggleAxialLabel === next.onToggleAxialLabel &&
    prev.onHoverRange === next.onHoverRange &&
    prev.onHoverEnd === next.onHoverEnd &&
    prev.filterReasons.length === next.filterReasons.length &&
    prev.filterReasons.every((r, i) => r === next.filterReasons[i]) &&
    prev.allFilterReasons.length === next.allFilterReasons.length &&
    prev.allFilterReasons.every((r, i) => r === next.allFilterReasons[i]) &&
    prev.speakerNames === next.speakerNames &&
    prev.speakerColorMap === next.speakerColorMap
  );
}

const SegmentCard = memo(SegmentCardInner, areEqual);
export default SegmentCard;
