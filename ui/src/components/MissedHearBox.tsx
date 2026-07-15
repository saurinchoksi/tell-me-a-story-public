/**
 * MissedHearBox — the hear-box for missed-speech candidates (TMAS-75).
 *
 * A missed candidate has NO transcript segment (that's its definition), so
 * the validator's segment-anchored notes can't hold an observation about
 * one. This panel appears with the stepper and captures the ear-check pair —
 * "what I hear" free text + a verdict — keyed by the candidate's span.
 * Verdict clicks save immediately; text saves on blur. Writes go to the
 * per-session sidecar (missed-speech-verdicts.json), never a transcript.
 *
 * The parent keys this component by candidate span, so landing on a new
 * candidate remounts it and useState seeds the draft fresh; a save echo
 * (same span, new hearing object) doesn't remount and can't clobber typing.
 * Save status is owned by the parent — a hearing is by-ear ground truth, and
 * a "save failed" notice must survive stepping between candidates.
 */

import { useState } from 'react';
import type { MissedSpeechCandidate, MissedHearing, HearingVerdict } from '../types';
import { formatTime } from '../utils/time';

/**
 * Display labels for the four verdicts, in the order the ear asks them.
 * Mirrors HEARING_VERDICTS (api/routes/missed_speech.py) and HearingVerdict
 * (../types) — the three move together.
 */
const VERDICTS: { value: HearingVerdict; label: string }[] = [
  { value: 'real-speech', label: 'real speech' },
  { value: 'hallucination-adjacent', label: 'hallucination-adjacent' },
  { value: 'boundary-tail', label: 'boundary tail' },
  { value: 'nothing', label: 'nothing' },
];

interface MissedHearBoxProps {
  candidate: MissedSpeechCandidate;
  /** Stepper position, e.g. "2/4" — repeated here so the panel is self-describing. */
  cursorLabel: string;
  /** The saved entry for this candidate, or null if nothing recorded yet. */
  hearing: MissedHearing | null;
  /** Outcome of the latest save — page-owned so it survives remounts. */
  saveState: 'idle' | 'saved' | 'failed';
  onSave: (hear: string, verdict: HearingVerdict | null) => void;
}

export default function MissedHearBox({
  candidate,
  cursorLabel,
  hearing,
  saveState,
  onSave,
}: MissedHearBoxProps) {
  // Draft text lives locally so keystrokes don't round-trip the store;
  // the parent's key= remounts this component per candidate (see header).
  const [draft, setDraft] = useState(hearing?.hear ?? '');

  const verdict = hearing?.verdict ?? null;

  return (
    <div className="missed-hearbox">
      <span className="missed-hearbox-span" title="Candidate span (diarization heard a voice, no segment exists)">
        {cursorLabel} &middot; {formatTime(candidate.start)}&ndash;{formatTime(candidate.end)} &middot; {candidate.dur.toFixed(1)}s &middot; {candidate.speaker}
      </span>
      <textarea
        className="missed-hearbox-text"
        placeholder="what I hear — type the words as you hear them"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => {
          if (draft !== (hearing?.hear ?? '')) onSave(draft, verdict);
        }}
        rows={1}
      />
      <div className="missed-hearbox-verdicts">
        {VERDICTS.map(({ value, label }) => (
          <button
            key={value}
            className={`btn btn--small ${verdict === value ? 'btn--active' : ''}`}
            title={`Verdict: ${label} (click again to clear)`}
            // Keep focus in the textarea: without this, mousedown blurs it
            // and the blur save races this click's save (two POSTs).
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => onSave(draft, verdict === value ? null : value)}
          >
            {label}
          </button>
        ))}
      </div>
      {saveState !== 'idle' && (
        <span
          className={`missed-hearbox-status ${saveState === 'failed' ? 'missed-hearbox-status--warn' : ''}`}
        >
          {saveState === 'saved' ? 'saved' : '⚠ save failed — retry before leaving the session'}
        </span>
      )}
    </div>
  );
}
