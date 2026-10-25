/**
 * ValidatorPage — transcript validation tool embedded in the React app.
 *
 * Renders outside <Layout> for full viewport width. Loads session transcript +
 * notes, renders a WaveSurfer waveform and scrollable segment cards. Active
 * segment/word highlighting runs at 60fps via refs + direct DOM manipulation
 * (binary search for current segment, class toggle on the card element).
 */

import { useEffect, useRef, useState, useCallback, Fragment } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { getSession, getNotes, saveNotes, getAxialLabels, saveAxialLabels, audioURL, listSessions, getMissedSpeech, getMissedHearings, saveMissedHearings } from '../api/client';
import type { ValidatorSegment, Note, ContextTarget, SegmentId, AxialCode, StoryRegion, HearingVerdict } from '../types';
import { formatTime } from '../utils/time';
import { buildSpeakerColorMap } from '../utils/filters';
import WaveformPlayer, { type WaveformPlayerHandle } from '../components/WaveformPlayer';
import SegmentCard from '../components/SegmentCard';
import StoryDivider from '../components/StoryDivider';
import ContextMenu from '../components/ContextMenu';
import NoteModal from '../components/NoteModal';
import NotesDrawer from '../components/NotesDrawer';
import MissedHearBox from '../components/MissedHearBox';
import { findHearing } from '../utils/hearings';
import { useValidatorState } from './validator/useValidatorState';
import { useAutoScroll } from './validator/useAutoScroll';
import { useKeyboardShortcuts } from './validator/useKeyboardShortcuts';
import './ValidatorPage.css';

/** Audio playback speeds offered in the validator's speed dropdown. */
const PLAYBACK_RATES = [0.5, 1, 1.5, 2] as const;

/** Binary search: find the segment containing `time`, or -1. */
function findSegmentIndex(segments: ValidatorSegment[], time: number): number {
  let lo = 0;
  let hi = segments.length - 1;
  let result = -1;

  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (segments[mid].start <= time) {
      result = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }

  // Verify the found segment actually contains the time
  if (result >= 0 && time <= segments[result].end) {
    return result;
  }
  return -1;
}

/** Find the active word within a segment at a given time. */
function findWordIndex(segment: ValidatorSegment, time: number): number {
  for (let i = 0; i < segment.words.length; i++) {
    if (time >= segment.words[i].start && time <= segment.words[i].end) {
      return i;
    }
  }
  return -1;
}

export default function ValidatorPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { state, dispatch, derived } = useValidatorState();
  const waveformRef = useRef<WaveformPlayerHandle>(null);

  // Map of segment ID → card DOM element for imperative .active toggling
  const segmentRefsMap = useRef(new Map<SegmentId, HTMLDivElement>());
  // Track current active indices to clear previous .active classes
  const activeSegRef = useRef(-1);
  const activeWordRef = useRef(-1);
  const segmentsContainerRef = useRef<HTMLDivElement>(null);
  // Ref to segments for 60fps callback (avoids closing over state)
  const segmentsRef = useRef<ValidatorSegment[]>([]);
  // Top-level story regions (title/world per story) for the divider; set on load.
  const [stories, setStories] = useState<StoryRegion[]>([]);

  // Throttle SET_TIME dispatches (~4fps for display only)
  const lastDispatchRef = useRef(0);

  // Guards the debounced axial-labels save from echoing the load.
  // Reset to true on every session change.
  const initialLabelsRef = useRef(true);

  // Auto-scroll: scrolls active segment into view, pauses on user scroll
  const { scrollToSegment } = useAutoScroll(segmentsContainerRef);

  // Keep segmentsRef in sync with state
  useEffect(() => { segmentsRef.current = state.segments; }, [state.segments]);

  // Load session data
  useEffect(() => {
    if (!id) return;

    // Reset save-guard so the new session's LOAD_SESSION doesn't trigger an
    // echo POST, and so any chip-click that happens during the fetch window
    // doesn't get saved under the wrong session id.
    initialLabelsRef.current = true;
    dispatch({ type: 'SET_LOADING', loading: true });

    Promise.all([getSession(id), getNotes(id), getAxialLabels(id)])
      .then(([session, notes, axialLabels]) => {
        const segments = (session.transcript?.segments ?? []) as ValidatorSegment[];
        setStories(session.transcript?._stories ?? []);
        const speakerNames = new Map<string, string>();
        const identifications = session.identifications?.identifications ?? [];
        for (const ident of identifications) {
          if (ident.profile_name) {
            speakerNames.set(ident.speaker_key, ident.profile_name);
          }
        }
        const speakerColorMap = buildSpeakerColorMap(identifications);
        dispatch({ type: 'LOAD_SESSION', segments, notes, axialLabels, speakerNames, speakerColorMap });
      })
      .catch((e) => dispatch({ type: 'SET_ERROR', error: e.message }));
  }, [id, dispatch]);

  // Load sessions list for dropdown
  useEffect(() => {
    listSessions()
      .then((sessions) => dispatch({ type: 'SET_SESSIONS', sessions }))
      .catch(() => {}); // non-critical — dropdown just stays empty
  }, [dispatch]);

  // Load missed-speech candidates — separate from the critical
  // session load so a session without diarization still opens; the Missed
  // button then carries the error instead of silently showing zero.
  useEffect(() => {
    if (!id) return;
    // The route reuses this mounted component across session switches, so a
    // slow response from the previous session must not land in this one.
    let stale = false;
    dispatch({ type: 'SET_MISSED_SPEECH', candidates: null, minGap: null }); // reset while fetching
    getMissedSpeech(id)
      .then((data) => {
        if (!stale) dispatch({ type: 'SET_MISSED_SPEECH', candidates: data.candidates, minGap: data.min_gap });
      })
      .catch((e: Error) => {
        if (!stale) dispatch({ type: 'SET_MISSED_ERROR', error: e.message });
      });
    return () => {
      stale = true;
    };
  }, [id, dispatch]);

  // Serializes sidecar POSTs: a textarea-blur save and a verdict-click save
  // can fire back-to-back, and two unordered full-array writes would let the
  // earlier state land last on disk. Chaining guarantees dispatch order.
  const hearingsSaveQueue = useRef<Promise<unknown>>(Promise.resolve());

  // Save outcome ("saved" / "save failed") lives on the page, not in the
  // panel: the panel remounts per candidate, and a failure notice that
  // vanished on stepping would leave an unsaved edit looking recorded. A
  // hearing is by-ear ground truth; a silently lost save is a wasted listen.
  // Stamped with its session id — another session's status derives to
  // 'idle', so no reset-on-switch effect is needed.
  const [hearingSave, setHearingSave] = useState<{ id: string; state: 'saved' | 'failed' } | null>(null);
  const hearingSaveState = hearingSave && hearingSave.id === id ? hearingSave.state : 'idle';

  // Load the hear-box sidecar. On failure hearings stay null and
  // the panel never renders (saves are full-array replacement, so allowing a
  // save before a successful load could clobber recorded hearings) — the
  // error is surfaced in the panel's place, not just the console.
  useEffect(() => {
    if (!id) return;
    let stale = false;
    dispatch({ type: 'SET_MISSED_HEARINGS', hearings: null }); // reset while fetching
    // New session, fresh save chain: a hung POST from the previous session
    // must not block this one's saves (each save is URL-bound at call time,
    // so the old one can only ever write the old session's file).
    hearingsSaveQueue.current = Promise.resolve();
    getMissedHearings(id)
      .then((hearings) => {
        if (!stale) dispatch({ type: 'SET_MISSED_HEARINGS', hearings });
      })
      .catch((e: Error) => {
        if (!stale) dispatch({ type: 'SET_MISSED_HEARINGS_ERROR', error: e.message });
      });
    return () => {
      stale = true;
    };
  }, [id, dispatch]);

  // Paint/clear the candidate spans on the waveform. state.duration is a
  // dependency so the spans re-apply after the waveform (re)initializes.
  useEffect(() => {
    const spans = state.showMissed && state.missedSpeech ? state.missedSpeech : [];
    waveformRef.current?.setMarkedRegions(spans);
  }, [state.showMissed, state.missedSpeech, state.duration]);

  // 60fps time update handler — imperative DOM manipulation via refs only.
  // Uses segmentsRef (not state) so callback identity is stable.
  const handleTimeUpdate = useCallback(
    (time: number) => {
      const segments = segmentsRef.current;
      if (segments.length === 0) return;

      // Binary search for active segment
      const segIdx = findSegmentIndex(segments, time);
      const seg = segIdx >= 0 ? segments[segIdx] : null;
      const wordIdx = seg ? findWordIndex(seg, time) : -1;

      // Clear previous active segment
      if (activeSegRef.current !== segIdx) {
        if (activeSegRef.current >= 0 && activeSegRef.current < segments.length) {
          const prevCard = segmentRefsMap.current.get(segments[activeSegRef.current].id);
          if (prevCard) {
            prevCard.classList.remove('active');
            // Clear all active words in previous card
            const prevWords = prevCard.querySelectorAll('.word-span.active');
            prevWords.forEach((el) => el.classList.remove('active'));
          }
        }

        // Activate new segment + auto-scroll
        if (seg) {
          const newCard = segmentRefsMap.current.get(seg.id);
          if (newCard) {
            newCard.classList.add('active');
            scrollToSegment(newCard);
          }
        }
        activeSegRef.current = segIdx;
        activeWordRef.current = -1; // Reset word on segment change
      }

      // Update active word within segment
      if (seg && wordIdx !== activeWordRef.current) {
        const card = segmentRefsMap.current.get(seg.id);
        if (card) {
          // Clear previous active word
          if (activeWordRef.current >= 0) {
            const prevWord = card.querySelector(
              `.word-span[data-word="${activeWordRef.current}"]`,
            );
            prevWord?.classList.remove('active');
          }
          // Activate new word
          if (wordIdx >= 0) {
            const newWord = card.querySelector(
              `.word-span[data-word="${wordIdx}"]`,
            );
            newWord?.classList.add('active');
          }
        }
        activeWordRef.current = wordIdx;
      }

      // Throttled state dispatch for time display (~4fps)
      const now = performance.now();
      if (now - lastDispatchRef.current > 250) {
        dispatch({ type: 'SET_TIME', time });
        dispatch({ type: 'SET_ACTIVE_SEGMENT', index: activeSegRef.current });
        lastDispatchRef.current = now;
      }
    },
    [dispatch, scrollToSegment],
  );

  const handleSeek = useCallback(
    (time: number) => {
      waveformRef.current?.setTime(time);
      waveformRef.current?.play();
    },
    [],
  );

  // Step through missed-speech candidates (cyclic). Parks the playhead 1s
  // before the span — the same lead-in the EMP gaps-to-check page uses —
  // WITHOUT autoplay, so the reviewer can orient first; Space plays when
  // ready (Choksi's call at the first visual pass, 2026-07-03).
  const stepMissed = useCallback(
    (dir: 1 | -1) => {
      const cands = state.missedSpeech;
      if (!cands || cands.length === 0) return;
      const next =
        state.missedCursor < 0
          ? dir === 1 ? 0 : cands.length - 1
          : (state.missedCursor + dir + cands.length) % cands.length;
      dispatch({ type: 'SET_MISSED_CURSOR', index: next });
      waveformRef.current?.setTime(Math.max(0, cands[next].start - 1));
    },
    [state.missedSpeech, state.missedCursor, dispatch],
  );

  // Upsert the hear-box entry for the candidate under the stepper.
  // Keyed by the candidate's span; writes the whole array to the sidecar,
  // same optimistic dispatch-then-POST shape as persistNotes below. A failed
  // POST leaves the edit in memory — any later successful save resends the
  // full array, so the failure heals unless the session is left first (the
  // persistent "save failed" notice is what warns against leaving).
  const handleSaveHearing = useCallback(
    (hear: string, verdict: HearingVerdict | null) => {
      const cands = state.missedSpeech;
      const hearings = state.missedHearings;
      if (!id || !cands || hearings === null || state.missedCursor < 0 || state.missedMinGap === null) return;
      const cand = cands[state.missedCursor];
      const now = new Date().toISOString();
      const existing = findHearing(hearings, cand);
      const updated = existing
        ? hearings.map((h) => (h === existing ? { ...h, hear, verdict, updatedAt: now } : h))
        : [
            ...hearings,
            {
              start: cand.start,
              end: cand.end,
              min_gap: state.missedMinGap,
              hear,
              verdict,
              createdAt: now,
              updatedAt: now,
            },
          ];
      dispatch({ type: 'SET_MISSED_HEARINGS', hearings: updated });
      const attempt = hearingsSaveQueue.current.then(() => saveMissedHearings(id, updated));
      hearingsSaveQueue.current = attempt.catch(() => {}); // keep the chain alive past failures
      attempt
        .then(() => {
          setHearingSave({ id, state: 'saved' });
          // Functional update so the transient "saved" can't clear a
          // "failed" that a later attempt set in the meantime.
          window.setTimeout(
            () => setHearingSave((s) => (s?.state === 'saved' ? null : s)),
            1200,
          );
        })
        .catch(() => setHearingSave({ id, state: 'failed' }));
    },
    [id, state.missedSpeech, state.missedHearings, state.missedCursor, state.missedMinGap, dispatch],
  );

  // ⌥-click a word → play just that word (sample-accurate), leaving plain click
  // as seek-and-play-from-here. Stable identity so SegmentCard stays memoized.
  const handlePlayWord = useCallback((start: number, end: number) => {
    waveformRef.current?.playWord(start, end);
  }, []);

  const handleHoverRange = useCallback(
    (start: number, end: number) => {
      waveformRef.current?.highlightRange(start, end);
    },
    [],
  );

  const handleHoverEnd = useCallback(() => {
    waveformRef.current?.clearHighlight();
  }, []);

  const handleToggleAxialLabel = useCallback(
    (segmentId: SegmentId, code: AxialCode) => {
      dispatch({ type: 'TOGGLE_AXIAL_LABEL', segmentId, code });
    },
    [dispatch],
  );

  // Keyboard shortcuts (modal-aware) — placed after handleSeek is defined
  useKeyboardShortcuts({
    waveformRef,
    segments: state.segments,
    activeSegmentIndex: state.activeSegmentIndex,
    filters: state.filters,
    duplicateIds: state.duplicateIds,
    modalOpen: state.noteModal.visible,
    dispatch,
    onSeek: handleSeek,
    onToggleAxialLabel: handleToggleAxialLabel,
  });

  const handleContextMenu = useCallback(
    (e: React.MouseEvent, segmentIndex: number, wordIndex: number) => {
      e.preventDefault();
      const seg = state.segments[segmentIndex];
      if (!seg) return;

      if (wordIndex >= 0 && wordIndex < seg.words.length) {
        const w = seg.words[wordIndex];
        dispatch({
          type: 'SHOW_CONTEXT_MENU',
          menu: {
            visible: true,
            x: e.clientX,
            y: e.clientY,
            target: {
              type: 'word',
              segmentId: seg.id,
              segmentIndex,
              wordIndex,
              wordText: w.word.trim(),
              wordStart: w.start,
              timestamp: w.start,
            },
          },
        });
      } else {
        dispatch({
          type: 'SHOW_CONTEXT_MENU',
          menu: {
            visible: true,
            x: e.clientX,
            y: e.clientY,
            target: {
              type: 'segment',
              segmentId: seg.id,
              segmentIndex,
              timestamp: seg.start,
            },
          },
        });
      }
    },
    [state.segments, dispatch],
  );

  // Create stable callback ref factory for segment cards
  const getCardRef = useCallback(
    (segId: SegmentId) => (el: HTMLDivElement | null) => {
      if (el) {
        segmentRefsMap.current.set(segId, el);
      } else {
        segmentRefsMap.current.delete(segId);
      }
    },
    [],
  );

  // --- Note CRUD ---

  const persistNotes = useCallback(
    (notes: Note[]) => {
      dispatch({ type: 'SET_NOTES', notes });
      if (id) saveNotes(id, notes).catch(console.error);
    },
    [id, dispatch],
  );

  const handleAddNoteFromMenu = useCallback(() => {
    dispatch({
      type: 'SHOW_NOTE_MODAL',
      modal: {
        visible: true,
        target: state.contextMenu.target,
        existingNote: null,
      },
    });
  }, [state.contextMenu.target, dispatch]);

  const handleSaveNote = useCallback(
    (text: string, target: ContextTarget, existingNote: Note | null) => {
      const now = new Date().toISOString();
      let updated: Note[];

      if (existingNote) {
        // Edit existing
        updated = state.notes.map((n) =>
          n.id === existingNote.id ? { ...n, text } : n,
        );
      } else {
        // Create new
        const newNote: Note = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          segmentId: target.segmentId ?? null,
          wordIndex: target.wordIndex ?? null,
          wordText: target.wordText ?? null,
          wordStart: target.wordStart ?? null,
          timestamp: target.timestamp ?? 0,
          text,
          createdAt: now,
        };
        updated = [...state.notes, newNote];
      }

      persistNotes(updated);
      dispatch({ type: 'HIDE_NOTE_MODAL' });
    },
    [state.notes, persistNotes, dispatch],
  );

  const handleDeleteNote = useCallback(
    (noteId: string) => {
      const updated = state.notes.filter((n) => n.id !== noteId);
      persistNotes(updated);
      dispatch({ type: 'HIDE_NOTE_MODAL' });
    },
    [state.notes, persistNotes, dispatch],
  );

  const handleDeleteAllNotes = useCallback(() => {
    persistNotes([]);
  }, [persistNotes]);

  const handleEditNote = useCallback(
    (note: Note) => {
      const target: ContextTarget = note.wordText
        ? { type: 'word', segmentId: note.segmentId ?? undefined, wordText: note.wordText, wordStart: note.wordStart ?? undefined, timestamp: note.timestamp }
        : note.segmentId !== null
          ? { type: 'segment', segmentId: note.segmentId, timestamp: note.timestamp }
          : { type: 'timestamp', timestamp: note.timestamp };

      dispatch({
        type: 'SHOW_NOTE_MODAL',
        modal: { visible: true, target, existingNote: note },
      });
    },
    [dispatch],
  );

  // --- Axial labels ---
  // Debounce label saves. The initial LOAD_SESSION dispatch sets axialLabels;
  // skip saving on that pass to avoid round-tripping a no-op.
  useEffect(() => {
    if (state.loading) return;
    if (initialLabelsRef.current) {
      initialLabelsRef.current = false;
      return;
    }
    if (!id) return;
    const handle = setTimeout(() => {
      saveAxialLabels(id, state.axialLabels).catch(console.error);
    }, 400);
    return () => clearTimeout(handle);
  }, [id, state.axialLabels, state.loading]);

  // Stable dismiss/cancel callbacks for ContextMenu + NoteModal portals
  const handleDismissMenu = useCallback(() => dispatch({ type: 'HIDE_CONTEXT_MENU' }), [dispatch]);
  const handleCancelModal = useCallback(() => dispatch({ type: 'HIDE_NOTE_MODAL' }), [dispatch]);

  if (state.loading) {
    return (
      <div className="validator-page">
        <div className="validator-loading">Loading session...</div>
      </div>
    );
  }

  if (state.error) {
    return (
      <div className="validator-page">
        <div className="validator-error">
          <p>Error: {state.error}</p>
          <Link to="/sessions">Back to sessions</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="validator-page">
      {/* Header */}
      <header className="validator-header">
        <Link to="/sessions" className="validator-back">Sessions</Link>
        {state.sessions.length > 0 ? (
          <select
            className="validator-session-select"
            value={id}
            onChange={(e) => navigate(`/sessions/${e.target.value}/validate`)}
          >
            {state.sessions.map((s) => (
              <option key={s.id} value={s.id}>{s.id}</option>
            ))}
          </select>
        ) : (
          <span className="validator-session-id">{id}</span>
        )}
        <span className="validator-time">
          {formatTime(state.currentTime)} / {formatTime(state.duration)}
        </span>
        <span className="validator-label-progress" title="Segments with an axial-code label">
          {derived.labeledCount} / {state.segments.length} labeled
        </span>
      </header>

      {/* Waveform */}
      {id && (
        <WaveformPlayer
          ref={waveformRef}
          audioUrl={audioURL(id)}
          onTimeUpdate={handleTimeUpdate}
          onReady={(dur) => dispatch({ type: 'SET_DURATION', duration: dur })}
          onPlayPause={(p) => dispatch({ type: 'SET_PLAYING', playing: p })}
          playbackRate={state.playbackRate}
        />
      )}

      {/* Controls bar */}
      <div className="validator-controls">
        <div className="controls-playback">
          <button
            className="btn btn--small"
            onClick={() => waveformRef.current?.playPause()}
          >
            {state.playing ? 'Pause' : 'Play'}
          </button>
          <select
            className="speed-select"
            aria-label="Playback speed"
            value={state.playbackRate}
            onChange={(e) => dispatch({ type: 'SET_PLAYBACK_RATE', rate: Number(e.target.value) })}
          >
            {PLAYBACK_RATES.map((rate) => (
              <option key={rate} value={rate}>{rate}×</option>
            ))}
          </select>
        </div>

        <div className="controls-filters">
          <button
            className={`btn btn--small ${state.filters.silenceGap ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_FILTER', filter: 'silenceGap' })}
            title="Toggle silence gap filter (G)"
          >
            Gaps ({derived.filterCounts.silenceGap})
          </button>
          <button
            className={`btn btn--small ${state.filters.nearZero ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_FILTER', filter: 'nearZero' })}
            title="Toggle near-zero probability filter (Z)"
          >
            Near-zero ({derived.filterCounts.nearZero})
          </button>
          <button
            className={`btn btn--small ${state.filters.duplicates ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_FILTER', filter: 'duplicates' })}
            title="Toggle duplicate filter (D)"
          >
            Dupes ({derived.filterCounts.duplicates})
          </button>
          <button
            className={`btn btn--small ${state.showMissed ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_MISSED' })}
            disabled={state.missedSpeech === null}
            title={
              state.missedError
                ?? 'Show missed-speech spans on the waveform (M) — diarization heard a voice but no segment exists. Counts the whole recording; the published EMP floor is story-scoped, so it can read lower.'
            }
          >
            Missed ({state.missedError ? '—' : state.missedSpeech === null ? '…' : state.missedSpeech.length})
          </button>
          {state.showMissed && state.missedSpeech && state.missedSpeech.length > 0 && (
            <>
              <button
                className="btn btn--small"
                onClick={() => stepMissed(-1)}
                title="Previous missed-speech candidate"
              >
                &lsaquo;
              </button>
              <button
                className="btn btn--small"
                onClick={() => stepMissed(1)}
                title="Next missed-speech candidate"
              >
                &rsaquo;
              </button>
              <span className="missed-cursor">
                {state.missedCursor >= 0
                  ? `${state.missedCursor + 1}/${state.missedSpeech.length}`
                  : `${state.missedSpeech.length} span${state.missedSpeech.length === 1 ? '' : 's'}`}
              </span>
            </>
          )}
        </div>

        <div className="controls-actions">
          <button
            className={`btn btn--small ${state.drawerOpen ? 'btn--active' : ''}`}
            onClick={() => dispatch({ type: 'TOGGLE_DRAWER' })}
            title="Toggle notes drawer (Shift+N)"
          >
            Notes ({state.notes.length})
          </button>
        </div>
      </div>

      {/* Hear-box: appears with the stepper — a missed candidate has
          no segment to anchor a note to, so the ear's observation lands here.
          key= remounts per candidate so the draft text seeds fresh. */}
      {state.showMissed && state.missedSpeech && state.missedCursor >= 0 && state.missedHearings !== null && (
        <MissedHearBox
          key={`${state.missedSpeech[state.missedCursor].start}-${state.missedSpeech[state.missedCursor].end}`}
          candidate={state.missedSpeech[state.missedCursor]}
          cursorLabel={`${state.missedCursor + 1}/${state.missedSpeech.length}`}
          hearing={findHearing(state.missedHearings, state.missedSpeech[state.missedCursor])}
          saveState={hearingSaveState}
          onSave={handleSaveHearing}
        />
      )}

      {/* A sidecar that failed to load must not be a silent no-hear-box:
          recording is disabled (saves could clobber the file), so say why. */}
      {state.showMissed && state.missedHearingsError && (
        <div className="missed-hearbox">
          <span className="missed-hearbox-status missed-hearbox-status--warn">
            hear-box unavailable ({state.missedHearingsError}) — recording disabled until the sidecar loads
          </span>
        </div>
      )}

      {/* Segment list */}
      <div
        ref={segmentsContainerRef}
        className={`validator-segments ${state.drawerOpen ? 'drawer-open' : ''}`}
      >
        {state.segments.map((seg, i) => {
          const activeReasons = derived.segmentFilterReasons.get(seg.id) ?? [];
          const allReasons = derived.segmentAllFilterReasons.get(seg.id) ?? [];
          const label = derived.labelsBySegment.get(seg.id);
          // Show a divider at the first segment of each story (where _story changes).
          const prevStory = i > 0 ? state.segments[i - 1]._story : undefined;
          const story =
            seg._story !== undefined && seg._story !== prevStory
              ? stories[seg._story]
              : undefined;
          return (
            <Fragment key={seg.id}>
              {story && <StoryDivider title={story.title} world={story.world} />}
              <SegmentCard
                segment={seg}
                index={i}
                isFiltered={activeReasons.length > 0}
                filterReasons={activeReasons}
                allFilterReasons={allReasons}
                hasNotes={derived.notesBySegment.has(seg.id)}
                axialCodes={label?.codes ?? []}
                onToggleAxialLabel={handleToggleAxialLabel}
                speakerNames={state.speakerNames}
                speakerColorMap={state.speakerColorMap}
                onSeek={handleSeek}
                onPlayWord={handlePlayWord}
                onContextMenu={handleContextMenu}
                onHoverRange={handleHoverRange}
                onHoverEnd={handleHoverEnd}
                cardRef={getCardRef(seg.id)}
              />
            </Fragment>
          );
        })}
      </div>

      {/* Context menu (portal) */}
      <ContextMenu
        menu={state.contextMenu}
        onAddNote={handleAddNoteFromMenu}
        onDismiss={handleDismissMenu}
      />

      {/* Note modal (portal) */}
      <NoteModal
        modal={state.noteModal}
        onSave={handleSaveNote}
        onDelete={handleDeleteNote}
        onCancel={handleCancelModal}
      />

      {/* Notes drawer */}
      <NotesDrawer
        open={state.drawerOpen}
        tab={state.drawerTab}
        notes={state.notes}
        segments={state.segments}
        onTabChange={(tab) => dispatch({ type: 'SET_DRAWER_TAB', tab })}
        onJump={handleSeek}
        onEditNote={handleEditNote}
        onDeleteNote={handleDeleteNote}
        onDeleteAll={handleDeleteAllNotes}
      />

      {/* Keyboard hints */}
      <div className="keyboard-hints">
        <span><kbd>Space</kbd> play/pause</span>
        <span><kbd>&larr;</kbd><kbd>&rarr;</kbd> seek</span>
        <span><kbd>&uarr;</kbd><kbd>&darr;</kbd> segments</span>
        <span><kbd>1</kbd>&hellip;<kbd>9</kbd> code &middot; M10 click-only</span>
        <span><kbd>0</kbd> N/A</span>
        <span><kbd>N</kbd> note</span>
        <span><kbd>Shift+N</kbd> drawer</span>
        <span><kbd>G</kbd><kbd>Z</kbd><kbd>D</kbd><kbd>M</kbd> filters</span>
      </div>
    </div>
  );
}
