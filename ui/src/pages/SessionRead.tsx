/**
 * SessionRead — the plain reading view for a session (TMAS-71).
 *
 * The default click-through from the Sessions list: read the story, play the
 * audio, see who said what. No axial-coding chips, no probability colors, no
 * shortcut bar — that research rig lives on the Validator, one click away.
 *
 * Renders the transcript as speaker-labeled lines in reading order, grouped by
 * story. Clicking a line plays from its start; the line playing highlights and
 * gently follows as the audio moves. Identified speakers show their name;
 * everyone else keeps their honest raw SPEAKER_00/01 label. [unintelligible]
 * gaps render as-is — and stay clickable: listening to the unclear moment is a
 * first-class action in this project.
 */

import { useEffect, useMemo, useRef, useState, useCallback, Fragment } from 'react';
import { useParams, Link } from 'react-router-dom';
import type {
  SessionDetail,
  ValidatorSegment,
  StoryRegion,
  StorySummary,
  SegmentId,
} from '../types';
import { getSession, audioURL } from '../api/client';
import { getDominantSpeaker, getSpeakerClass, buildSpeakerColorMap } from '../utils/filters';
import { formatSessionDate } from '../utils/time';
import AudioPlayer from '../components/AudioPlayer';
import StoryDivider from '../components/StoryDivider';
import StoryHeading from '../components/StoryHeading';
import './SessionRead.css';

/** The speaker a segment belongs to. Gap segments carry their speaker at the
 *  segment level (no words); real segments derive it from their words. */
function speakerOf(seg: ValidatorSegment): string | null {
  if (seg._source === 'diarization_gap') return seg._speaker?.label ?? null;
  return getDominantSpeaker(seg);
}

/**
 * The segment whose span contains `time`, or null. Injected gap segments carry
 * the raw diarization span and can overlap a neighboring real segment's tail,
 * so when more than one segment contains the time a real (non-gap) one wins —
 * a gap only highlights when nothing real covers the moment.
 */
function findActiveSegId(segments: ValidatorSegment[], time: number): SegmentId | null {
  let gapMatch: SegmentId | null = null;
  for (const s of segments) {
    if (s.start > time) break; // segments are time-ordered by start
    if (time <= s.end) {
      if (s._source !== 'diarization_gap') return s.id;
      if (gapMatch === null) gapMatch = s.id;
    }
  }
  return gapMatch;
}

/** Fold the transcript's story regions into the glanceable header summary the
 *  Sessions list already uses, so a drilled-in session stays recognizable. */
function storySummaryFromRegions(regions: StoryRegion[]): StorySummary | null {
  if (regions.length === 0) return null;
  const worlds = [...new Set(regions.map((r) => r.world).filter(Boolean))];
  const titles = regions.map((r) => r.title).filter(Boolean);
  return { label: '', n_stories: regions.length, worlds, titles };
}

export default function SessionRead() {
  const { id } = useParams<{ id: string }>();

  // Loaded data and per-session UI state, each stamped with the session id it
  // belongs to. Everything below DERIVES against the current :id, so
  // navigating /sessions/A → /sessions/B (back/forward included) instantly
  // reads as loading — no stale transcript, error, seek, or highlight from A —
  // without any synchronous setState inside the effect body (which this repo's
  // react-hooks lint forbids). Same idiom as ValidatorPage's id-stamped
  // hearingSave. (loaded is stamped by its own `id` field from the API.)
  const [loaded, setLoaded] = useState<SessionDetail | null>(null);
  const [loadError, setLoadError] = useState<{ id: string; message: string } | null>(null);
  const [seek, setSeek] = useState<{ id: string; t: number } | null>(null);
  const [active, setActive] = useState<{ id: string; seg: SegmentId } | null>(null);

  useEffect(() => {
    if (!id) return;
    // Stale-flag the fetch: the route reuses this mounted component across
    // session switches, so a slow response from the previous session must
    // not land in (or after) this one's.
    let stale = false;
    getSession(id)
      .then((s) => {
        if (!stale) setLoaded(s);
      })
      .catch((e: Error) => {
        if (!stale) setLoadError({ id, message: e.message });
      });
    return () => {
      stale = true;
    };
  }, [id]);

  // Derive the current session's view of the stamped state.
  const session = loaded && loaded.id === id ? loaded : null;
  const error = loadError && loadError.id === id ? loadError.message : null;
  const loading = !session && !error;
  const seekTo = seek && seek.id === id ? seek.t : undefined;
  const activeSegId = active && active.id === id ? active.seg : null;

  const segments = useMemo(
    () => (session?.transcript?.segments ?? []) as ValidatorSegment[],
    [session],
  );
  const stories = useMemo(
    () => session?.transcript?._stories ?? [],
    [session],
  );

  // Identified speaker names + identity-based color grouping (same source the
  // Validator reads): speaker_key → profile_name, and a stable color class.
  const { speakerNames, speakerColorMap } = useMemo(() => {
    const names = new Map<string, string>();
    const idents = session?.identifications?.identifications ?? [];
    for (const ident of idents) {
      if (ident.profile_name) names.set(ident.speaker_key, ident.profile_name);
    }
    return { speakerNames: names, speakerColorMap: buildSpeakerColorMap(idents) };
  }, [session]);

  // Keep the current segment list reachable from the (stable) time handler
  // without re-creating it on every playback tick.
  const segmentsRef = useRef<ValidatorSegment[]>([]);
  useEffect(() => { segmentsRef.current = segments; }, [segments]);

  const handleTimeUpdate = useCallback((time: number) => {
    if (!id) return;
    const next = findActiveSegId(segmentsRef.current, time);
    // Only re-render when the highlighted line actually changes.
    setActive((prev) => {
      if (next === null) return prev === null ? prev : null;
      if (prev && prev.id === id && prev.seg === next) return prev;
      return { id, seg: next };
    });
  }, [id]);

  // Clicking a line plays from its start. The tiny delta guarantees the seek
  // effect re-fires even when the same line is clicked twice.
  const handlePlayFrom = useCallback((start: number) => {
    if (!id) return;
    setSeek({ id, t: start + Math.random() * 0.001 });
  }, [id]);

  // Follow playback: keep the active line in view. block: 'nearest' stays
  // gentle — no movement while the line is already visible — so it nudges
  // rather than recenters, and never fights a reader scrolled elsewhere.
  const lineRefs = useRef(new Map<SegmentId, HTMLParagraphElement>());
  const getLineRef = useCallback(
    (segId: SegmentId) => (el: HTMLParagraphElement | null) => {
      if (el) lineRefs.current.set(segId, el);
      else lineRefs.current.delete(segId);
    },
    [],
  );
  useEffect(() => {
    if (activeSegId === null) return;
    lineRefs.current
      .get(activeSegId)
      ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [activeSegId]);

  const storyHeader = useMemo(() => storySummaryFromRegions(stories), [stories]);

  // Resolve each segment into a render row by looking back at its predecessor
  // (no running accumulator — that reads as a render-body mutation): whether a
  // story divider precedes it, whether its speaker label shows, and the
  // display name / color. A divider always re-shows the speaker label so a new
  // story names its speaker. Computed here, not in the JSX map, so playback
  // ticks (activeSegId) never recompute it.
  const rows = useMemo(() => {
    return segments.map((seg, i) => {
      const prevSeg = i > 0 ? segments[i - 1] : undefined;
      const story =
        seg._story !== undefined && seg._story !== prevSeg?._story
          ? stories[seg._story]
          : undefined;
      const spk = speakerOf(seg);
      const prevSpk = prevSeg ? speakerOf(prevSeg) : undefined;
      const showSpeaker = story !== undefined || spk !== prevSpk;

      return {
        seg,
        story,
        showSpeaker,
        isGap: seg._source === 'diarization_gap',
        displayName: spk ? (speakerNames.get(spk) ?? spk) : 'Unknown speaker',
        isIdentified: spk !== null && speakerNames.has(spk),
        hasSpeaker: spk !== null,
        speakerClass: (spk && speakerColorMap.get(spk)) || getSpeakerClass(spk),
      };
    });
  }, [segments, stories, speakerNames, speakerColorMap]);

  if (loading) return <p>Loading...</p>;
  if (error) return <p className="error">Error: {error}</p>;
  if (!session) return <p className="error">Session not found</p>;

  const { date, time } = formatSessionDate(id!);
  const hasTranscript = segments.length > 0;
  // No audio file → a click could play nothing, so lines render as plain text:
  // no button role, no pointer, no dead affordance.
  const canPlay = session.has_audio;

  return (
    <div className="session-read">
      <div className="session-read-topbar">
        <Link to="/sessions" className="session-read-back">&larr; Sessions</Link>
        <Link
          to={`/sessions/${id}/validate`}
          className="session-read-validator-link"
          title="Open this session in the Validator (research / calibration tool)"
        >
          Open Validator &rarr;
        </Link>
        <Link
          to={`/sessions/${id}/name-pass`}
          className="session-read-validator-link"
          title="Build the by-ear gold name key for this session"
        >
          Name pass &rarr;
        </Link>
      </div>

      <div className="session-read-header">
        <h1>{date}</h1>
        {time && <span className="session-read-time">{time}</span>}
        <StoryHeading stories={storyHeader} />
      </div>

      {canPlay && (
        <div className="session-read-audio">
          <AudioPlayer
            src={audioURL(id!)}
            seekTo={seekTo}
            onTimeUpdate={handleTimeUpdate}
            playOnSeek
          />
          <p className="session-read-hint">Tap any line to play from there.</p>
        </div>
      )}

      {!hasTranscript ? (
        <p className="session-read-empty">This session hasn&rsquo;t been transcribed yet.</p>
      ) : (
        <div className="session-read-transcript">
          {rows.map(({ seg, story, showSpeaker, isGap, displayName, isIdentified, hasSpeaker, speakerClass }) => (
            <Fragment key={seg.id}>
              {story && <StoryDivider title={story.title} world={story.world} />}
              {showSpeaker && (
                <div className={`session-read-speaker ${speakerClass}`}>
                  <span className="session-read-speaker-name">{displayName}</span>
                  {!isIdentified && hasSpeaker && (
                    <span className="session-read-speaker-raw" title="Speaker not yet identified">
                      unidentified
                    </span>
                  )}
                </div>
              )}
              <p
                ref={getLineRef(seg.id)}
                className={[
                  'session-read-line',
                  speakerClass,
                  canPlay ? 'session-read-line--playable' : '',
                  isGap ? 'session-read-line--gap' : '',
                  seg.id === activeSegId ? 'active' : '',
                ].filter(Boolean).join(' ')}
                {...(canPlay
                  ? {
                      onClick: () => handlePlayFrom(seg.start),
                      role: 'button',
                      tabIndex: 0,
                      onKeyDown: (e: React.KeyboardEvent) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          handlePlayFrom(seg.start);
                        }
                      },
                    }
                  : {})}
              >
                {isGap ? <span className="session-read-gap">[unintelligible]</span> : seg.text}
              </p>
            </Fragment>
          ))}
        </div>
      )}
    </div>
  );
}
