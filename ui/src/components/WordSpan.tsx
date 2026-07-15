/**
 * WordSpan — a single word within a segment card.
 *
 * Rendered as a <span> with probability coloring, speaker mismatch indicator,
 * click-to-seek, and right-click context menu support. The `.active` class is
 * toggled imperatively by ValidatorPage for 60fps audio sync.
 */

import type { ValidatorWord } from '../types';
import { getWordProbabilityClass, getSpeakerClass } from '../utils/filters';

interface WordSpanProps {
  word: ValidatorWord;
  segmentIndex: number;
  wordIndex: number;
  dominantSpeaker: string | null;
  onSeek: (time: number) => void;
  onPlayWord: (start: number, end: number) => void;
  onContextMenu: (e: React.MouseEvent, segmentIndex: number, wordIndex: number) => void;
  onHoverRange?: (start: number, end: number) => void;
  onHoverEnd?: () => void;
}

export default function WordSpan({
  word,
  segmentIndex,
  wordIndex,
  dominantSpeaker,
  onSeek,
  onPlayWord,
  onContextMenu,
  onHoverRange,
  onHoverEnd,
}: WordSpanProps) {
  // A rescued/realigned word may lack a Whisper probability; coalesce so a
  // single missing value can never crash the whole render (.toFixed on null).
  const prob = word.probability ?? 0;
  const probClass = getWordProbabilityClass(prob);
  const speakerLabel = word._speaker?.label ?? null;
  const mismatch = dominantSpeaker !== null && speakerLabel !== null && speakerLabel !== dominantSpeaker;
  const speakerClass = getSpeakerClass(speakerLabel);
  const hasCorrection = word._corrections && word._corrections.length > 0;

  const tooltip = [
    `prob: ${prob.toFixed(3)}`,
    `${word.start.toFixed(2)}s → ${word.end.toFixed(2)}s`,
    `dur: ${(word.end - word.start).toFixed(2)}s`,
    speakerLabel ? `${speakerLabel} (${(word._speaker?.coverage ?? 0).toFixed(2)})` : 'no speaker',
    hasCorrection ? `was: "${word._original}"` : null,
    '⌥-click: play just this word',
  ].filter(Boolean).join(' | ');

  return (
    <span
      className={[
        'word-span',
        probClass,
        speakerClass,
        mismatch ? 'speaker-mismatch' : '',
        hasCorrection ? 'word-corrected' : '',
      ].filter(Boolean).join(' ')}
      data-segment={segmentIndex}
      data-word={wordIndex}
      data-start={word.start}
      data-end={word.end}
      data-tooltip={tooltip}
      onClick={(e) => {
        if (e.altKey) {
          e.preventDefault();
          onPlayWord(word.start, word.end);
        } else {
          onSeek(word.start);
        }
      }}
      onContextMenu={(e) => onContextMenu(e, segmentIndex, wordIndex)}
      onMouseEnter={() => onHoverRange?.(word.start, word.end)}
      onMouseLeave={() => onHoverEnd?.()}
    >
      {word.word}
    </span>
  );
}
