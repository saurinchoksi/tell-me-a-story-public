import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getSessionDetections, scanSession, saveNameVerdict, audioURL } from '../api/client';
import { formatSessionDate, formatTime } from '../utils/time';
import type { DetectionFlag, SessionDetectionsData, NameVerdict, SessionDetectorResult, CanonNameFlag } from '../types';
import StoryHeading from '../components/StoryHeading';
import './SessionDetections.css';

// Clip window padding (seconds). We play the flagged token's whole containing
// segment so the audio matches the sentence shown on the card and survives
// word-timestamp drift; the pads just give a running start and tail.
const PREROLL = 0.5;
const TAIL = 0.4;
// If a segment is pathologically long (e.g. a broken-Whisper filler run), fall
// back to a tight window around the word so one click can't play 30 seconds.
const MAX_SEGMENT_CLIP = 12;

/** The [start, end] audio window to play for a flag, or null if untimed. */
function clipWindow(flag: DetectionFlag): [number, number] | null {
  const segStart = flag.segment_start;
  const segEnd = flag.segment_end;
  if (segStart != null && segEnd != null && segEnd - segStart <= MAX_SEGMENT_CLIP) {
    return [segStart, segEnd];
  }
  if (flag.start != null && flag.end != null) {
    return [flag.start, flag.end];
  }
  // Long segment but we have a word start: a tight word window beats nothing.
  if (segStart != null && segEnd != null && flag.start != null) {
    return [flag.start, Math.min(flag.start + 4, segEnd)];
  }
  return null;
}

interface FlagCardProps {
  flag: DetectionFlag;
  isPlaying: boolean;
  canPlay: boolean;
  onToggle: () => void;
}

// One occurrence row inside a name group: play button, the heard token, time · speaker, the
// sentence, and the Double Metaphone codes. The group header owns the name/badge/world, so every
// detector's rows look identical.
function FlagCard({ flag, isPlaying, canPlay, onToggle }: FlagCardProps) {
  return (
    <div className="detection-flag">
      <div className="detection-flag-top">
        <button
          className="detection-flag-play"
          onClick={onToggle}
          disabled={!canPlay}
          title={canPlay ? 'Play the audio around this flag' : 'No audio for this session'}
          aria-label={isPlaying ? 'Pause' : 'Play the audio around this flag'}
        >
          {isPlaying ? '❚❚' : '▶'}
        </button>
        <span className="detection-flag-token">{flag.token}</span>
        <span className="detection-flag-meta">
          {flag.start != null ? formatTime(flag.start) : '—'}
          {flag.segment_speaker ? ` · ${flag.segment_speaker}` : ''}
        </span>
      </div>
      {flag.segment_text != null ? (
        <p className="detection-flag-text">{flag.segment_text}</p>
      ) : (
        <p className="detection-flag-text detection-flag-text--missing">
          (segment {String(flag.segment_id)} not found in transcript)
        </p>
      )}
      <div className="detection-flag-codes">
        Double Metaphone: {flag.dm_codes.join(' · ')}
      </div>
    </div>
  );
}

interface FlagGroup {
  key: string;
  ref: string;             // the reference spelling shown in the header
  badgeLabel: string;
  badgeClass: string;      // 'detection-flag-type--family' | '--inconsistent' | '--canon'
  world?: string | null;   // M9c only
  spellings: string[];     // distinct heard spellings, punctuation-stripped
  flags: DetectionFlag[];
}

/** Group a section's per-occurrence flags into one entry per name, the SAME shape for all three
 *  detectors so the page reads as one family: M9a groups by family name, M9b by spelling cluster,
 *  M9c by corrected name. Each carries a reference spelling + a colored badge. Order is first-seen. */
function buildGroups(flags: DetectionFlag[]): FlagGroup[] {
  const groups = new Map<string, FlagGroup>();
  for (const f of flags) {
    let key: string;
    let header: Omit<FlagGroup, 'key' | 'spellings' | 'flags'>;
    if ('matched_canonicals' in f) {              // M9a — family-name match
      const ref = f.matched_canonicals.join(', ');
      key = ref;
      header = { ref, badgeLabel: f.match_type, badgeClass: 'detection-flag-type--family' };
    } else if ('cluster_spellings' in f) {        // M9b — inconsistent spelling
      key = f.cluster_id;
      header = { ref: f.majority_spelling ?? '(varies)', badgeLabel: 'inconsistent',
                 badgeClass: 'detection-flag-type--inconsistent' };
    } else {                                       // M9c — canon misspelling
      key = f.canonical;
      header = { ref: f.canonical, badgeLabel: 'canon', badgeClass: 'detection-flag-type--canon',
                 world: f.story_world };
    }
    const existing = groups.get(key);
    if (existing) existing.flags.push(f);
    else groups.set(key, { key, ...header, spellings: [], flags: [f] });
  }
  for (const grp of groups.values()) {
    grp.spellings = [...new Set(grp.flags.map((f) => f.token.replace(/[.,!?'’"]+$/, '')))];
  }
  return [...groups.values()];
}

/** Distinct {display, cleaned} spellings in a group — for the clickable "mark correct"
 *  chips. Keyed by `cleaned` (what a verdict matches on); first display wins. */
function spellingPairs(grp: FlagGroup): { display: string; cleaned: string }[] {
  const seen = new Map<string, string>();
  for (const f of grp.flags) {
    if (!seen.has(f.cleaned)) seen.set(f.cleaned, f.token.replace(/[.,!?'’"]+$/, ''));
  }
  return [...seen.entries()].map(([cleaned, display]) => ({ display, cleaned }));
}

// Keyed by session id so navigating A→B remounts the view — a fresh <audio>
// element (the old one unmounts, stopping playback) and clean refs/state,
// instead of leaking the prior session's audio and flags into the next.
export default function SessionDetections() {
  const { id } = useParams<{ id: string }>();
  return <SessionDetectionsView key={id} id={id} />;
}

function SessionDetectionsView({ id }: { id: string | undefined }) {
  const [data, setData] = useState<SessionDetectionsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  // Reveal the lowest M9c tier (judge votes < 4). Default off — those are the shakiest guesses.
  const [showAll, setShowAll] = useState(false);

  // Shared clip player — one <audio> element drives every flag's play button.
  const audioRef = useRef<HTMLAudioElement>(null);
  const stopAtRef = useRef<number | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    getSessionDetections(id)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id]);

  function playClip(key: string, start: number, end: number) {
    const audio = audioRef.current;
    if (!audio) return;
    if (playingKey === key) {
      audio.pause(); // toggle off (onPause clears playingKey)
      return;
    }
    stopAtRef.current = end + TAIL;
    audio.currentTime = Math.max(0, start - PREROLL);
    audio.play().then(() => setPlayingKey(key)).catch(() => setPlayingKey(null));
  }

  function handleTimeUpdate() {
    const audio = audioRef.current;
    if (!audio || stopAtRef.current == null) return;
    if (audio.currentTime >= stopAtRef.current) {
      audio.pause(); // auto-stop at the clip's end
    }
  }

  function handleStopped() {
    stopAtRef.current = null;
    setPlayingKey(null);
  }

  async function handleRescan() {
    if (!id) return;
    setScanning(true);
    setError(null);
    try {
      setData(await scanSession(id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setScanning(false);
    }
  }

  /** Toggle a human name verdict; the server returns the re-applied detail. */
  async function applyVerdict(verdict: NameVerdict) {
    if (!id) return;
    setError(null);
    try {
      setData(await saveNameVerdict(id, verdict));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (loading) return <p>Loading detections...</p>;
  if (error) return <p className="error">Error: {error}</p>;
  if (!data || !id) return null;

  const { date, time } = formatSessionDate(id);
  const sections = Object.entries(data.detectors);
  const hasAudio = data.has_audio;

  // One name group's card — identical for every detector and every M9c tier. The header owns the
  // name/badge/world + verdict control; occurrences nest beneath. Factored out so the three M9c
  // confidence tiers can each render the same group shape.
  const renderGroup = (grp: FlagGroup, result: SessionDetectorResult) => (
    <div className="detection-group" key={grp.key}>
      <div className="detection-group-header">
        <span className="detection-flag-arrow" aria-hidden="true">→</span>
        <span className="detection-flag-canonical">{grp.ref}</span>
        <span className={`detection-flag-type ${grp.badgeClass}`}>{grp.badgeLabel}</span>
        {grp.world ? <span className="detection-flag-world">{grp.world}</span> : null}
        <span className="detection-group-count">heard {grp.flags.length}×</span>
        {result.failure_mode === 'M9b' ? (
          <span className="detection-group-spellings">
            {spellingPairs(grp).map(({ display, cleaned }) => (
              <button
                key={cleaned}
                className="spelling-chip"
                title="This spelling is correct — click to remove it from the flags"
                onClick={() => applyVerdict({ type: 'correct', cleaned, spelling: display })}
              >
                {display}
                <span className="spelling-chip-x" aria-hidden="true">×</span>
              </button>
            ))}
          </span>
        ) : (
          <span className="detection-group-spellings">{grp.spellings.join(' · ')}</span>
        )}
        {result.failure_mode === 'M9c' && (
          <button
            className="detection-verdict-btn"
            title={`These flagged spellings are a made-up name, not the real ${grp.ref} — so they aren't a misspelled character`}
            onClick={() =>
              applyVerdict({
                type: 'not_canon',
                name: grp.ref,
                cleaned_at_review: [...new Set(grp.flags.map((f) => f.cleaned))],
              })
            }
          >
            Made-up name
          </button>
        )}
      </div>
      <div className="detection-flags">
        {grp.flags.map((flag) => {
          // detector-prefixed so the same word flagged by two detectors can't collide
          const key = `${result.failure_mode}-${String(flag.segment_id)}-${flag.word_index}`;
          const clip = clipWindow(flag);
          return (
            <FlagCard
              key={key}
              flag={flag}
              isPlaying={playingKey === key}
              canPlay={hasAudio && clip !== null}
              onToggle={() => clip && playClip(key, clip[0], clip[1])}
            />
          );
        })}
      </div>
    </div>
  );

  // M9c renders in confidence tiers (api.helpers.canon_tier): confident + best-guess show by
  // default; the lowest tier (judge votes < 4) hides behind "Show all". A made-up-name verdict
  // works on every tier's groups.
  const renderM9cBody = (result: SessionDetectorResult) => {
    // Default a missing tier to 'low'. The server always sets it, but a monitor must never silently
    // swallow a flag — surfacing it under "Show all" beats hiding it under a "clean scan" message.
    const tierGroups = (t: string) =>
      buildGroups(result.flags.filter((f) => ((f as CanonNameFlag).tier ?? 'low') === t));
    const confident = tierGroups('confident');
    const bestGuess = tierGroups('best_guess');
    const low = tierGroups('low');
    const hasLower = bestGuess.length > 0 || low.length > 0;

    return (
      <>
        {confident.length > 0 && (
          <div className="detection-groups">
            {hasLower && <div className="detection-tier-label">Confident</div>}
            {confident.map((g) => renderGroup(g, result))}
          </div>
        )}

        {bestGuess.length > 0 && (
          <div className="detection-tier detection-tier--bestguess">
            <div className="detection-tier-label">
              Best guess<span className="detection-tier-hint">the model's read, but it doesn't sound like the canon name</span>
            </div>
            <div className="detection-groups">{bestGuess.map((g) => renderGroup(g, result))}</div>
          </div>
        )}

        {confident.length === 0 && bestGuess.length === 0 && low.length === 0 && (
          <p className="detection-section-clean">No flags — clean scan.</p>
        )}

        {low.length > 0 && (
          <>
            {showAll && (
              <div className="detection-tier detection-tier--low">
                <div className="detection-tier-label">
                  Lowest confidence<span className="detection-tier-hint">a minority of the judge's rounds agreed</span>
                </div>
                <div className="detection-groups">{low.map((g) => renderGroup(g, result))}</div>
              </div>
            )}
            <button className="detection-tier-toggle" onClick={() => setShowAll((v) => !v)}>
              {showAll
                ? '− Show fewer'
                : `+ Show all (${low.length} lower-confidence ${low.length === 1 ? 'guess' : 'guesses'})`}
            </button>
          </>
        )}
      </>
    );
  };

  return (
    <div className="session-detections-page">
      <Link to="/monitor" className="session-detections-back">← Monitor</Link>

      {data.has_audio && (
        <audio
          ref={audioRef}
          src={audioURL(id)}
          preload="metadata"
          onTimeUpdate={handleTimeUpdate}
          onPause={handleStopped}
          onEnded={handleStopped}
        />
      )}

      <div className="session-detections-header">
        <div className="session-detections-header-row">
          <h1>{date} · {time}</h1>
          <button
            className="session-detections-rescan"
            onClick={handleRescan}
            disabled={scanning}
            title="Re-scan this session (code detectors + the M9b LLM judge)"
          >
            {scanning ? 'Scanning…' : 'Full re-scan'}
          </button>
        </div>
        <StoryHeading stories={data.stories} />
        <p className="session-detections-subtitle">session {id}</p>
        {data.warning && <p className="session-detections-warning">⚠ {data.warning}</p>}
      </div>

      {data.name_verdicts.length > 0 && (
        <div className="detection-corrections">
          <span className="detection-corrections-label">Your corrections</span>
          {data.name_verdicts.map((v, i) => (
            <span key={i} className="detection-correction">
              {v.type === 'not_canon'
                ? `Made-up name (not ${v.name})`
                : `“${v.spelling ?? v.cleaned}” — correct spelling`}
              {v.stale && (
                <span className="detection-correction-stale" title="The name's spellings changed since you set this — re-check">
                  ⟳
                </span>
              )}
              <button
                className="detection-correction-undo"
                onClick={() =>
                  applyVerdict(
                    v.type === 'not_canon'
                      ? { type: 'not_canon', name: v.name }
                      : { type: 'correct', cleaned: v.cleaned },
                  )
                }
              >
                undo
              </button>
            </span>
          ))}
        </div>
      )}

      {sections.length === 0 ? (
        <div className="session-detections-empty">
          <h2>No detections</h2>
          <p>This session hasn't been scanned (or has no transcript). Use Full re-scan.</p>
        </div>
      ) : (
        sections.map(([detId, result]) => (
          <section key={detId} className="detection-section">
            <div className="detection-section-header">
              <span className="detection-section-mode">{result.failure_mode}</span>
              <h2 className="detection-section-label">{result.label}</h2>
              <span className="detection-section-meta">
                {result.n_flags} flag{result.n_flags !== 1 ? 's' : ''} /{' '}
                {result.n_word_tokens} tokens · v{result.detector_version}
                {result.judge_applied ? ' +judge' : ''} · scanned{' '}
                {new Date(result.run_at).toLocaleString()}
                {result.stale && (
                  <span className="detection-stale" title="Transcript changed since this scan — re-scan">
                    {' '}⟳ stale
                  </span>
                )}
              </span>
            </div>
            {result.failure_mode === 'M9c' ? (
              renderM9cBody(result)
            ) : result.n_flags === 0 ? (
              <p className="detection-section-clean">No flags — clean scan.</p>
            ) : (
              // M9a / M9b: one header per name, occurrences nested under it
              <div className="detection-groups">
                {buildGroups(result.flags).map((grp) => renderGroup(grp, result))}
              </div>
            )}
          </section>
        ))
      )}
    </div>
  );
}
