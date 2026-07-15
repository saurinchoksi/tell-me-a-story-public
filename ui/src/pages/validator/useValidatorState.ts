/**
 * useValidatorState — reducer-based state management for the validator page.
 *
 * Centralizes ~15 fields with cross-cutting updates (e.g. filter toggle
 * recomputes which segments are dimmed). Dispatch is used by the page
 * component, keyboard shortcuts, and the notes system.
 */

import { useReducer, useMemo } from 'react';
import type { ValidatorSegment, Note, FilterState, ContextMenuState, NoteModalState, SessionSummary, SegmentId, AxialLabel, AxialCode, MissedSpeechCandidate, MissedHearing } from '../../types';
import { findDuplicateSegmentIds, getFilterReasons, getAllFilterReasons } from '../../utils/filters';

export interface ValidatorState {
  sessions: SessionSummary[];
  segments: ValidatorSegment[];
  notes: Note[];
  axialLabels: AxialLabel[];
  currentTime: number;
  activeSegmentIndex: number;
  duration: number;
  playing: boolean;
  playbackRate: number;
  filters: FilterState;
  duplicateIds: Set<SegmentId>;
  /** Missed-speech candidates from the API (TMAS-48). null = loading or unavailable. */
  missedSpeech: MissedSpeechCandidate[] | null;
  /** Why missed-speech is unavailable (e.g. no diarization.json) — shown on the button. */
  missedError: string | null;
  /** Whether the missed-speech spans are painted on the waveform. */
  showMissed: boolean;
  /** Index of the last-visited missed-speech candidate (-1 = none yet). */
  missedCursor: number;
  /**
   * Hear-box entries (TMAS-75): the ear's verdicts, keyed by candidate span.
   * null until the sidecar load succeeds — saves are full-array replacement,
   * so saving before load would clobber previously recorded hearings.
   */
  missedHearings: MissedHearing[] | null;
  /** Why the hear-box sidecar failed to load (e.g. corrupt file) — surfaced
   *  in place of the panel so a broken sidecar isn't a silent no-hear-box. */
  missedHearingsError: string | null;
  /** Threshold the candidates were queried at — stamped onto new hearings. */
  missedMinGap: number | null;
  drawerOpen: boolean;
  drawerTab: 'notes' | 'low-confidence';
  contextMenu: ContextMenuState;
  noteModal: NoteModalState;
  speakerNames: Map<string, string>;
  speakerColorMap: Map<string, string>;
  loading: boolean;
  error: string | null;
}

export type ValidatorAction =
  | { type: 'LOAD_SESSION'; segments: ValidatorSegment[]; notes: Note[]; axialLabels: AxialLabel[]; speakerNames: Map<string, string>; speakerColorMap: Map<string, string> }
  | { type: 'SET_TIME'; time: number }
  | { type: 'SET_ACTIVE_SEGMENT'; index: number }
  | { type: 'SET_DURATION'; duration: number }
  | { type: 'SET_PLAYING'; playing: boolean }
  | { type: 'SET_PLAYBACK_RATE'; rate: number }
  | { type: 'TOGGLE_FILTER'; filter: keyof FilterState }
  | { type: 'SET_MISSED_SPEECH'; candidates: MissedSpeechCandidate[] | null; minGap: number | null }
  | { type: 'SET_MISSED_ERROR'; error: string }
  | { type: 'TOGGLE_MISSED' }
  | { type: 'SET_MISSED_CURSOR'; index: number }
  | { type: 'SET_MISSED_HEARINGS'; hearings: MissedHearing[] | null }
  | { type: 'SET_MISSED_HEARINGS_ERROR'; error: string }
  | { type: 'TOGGLE_DRAWER' }
  | { type: 'SET_DRAWER_TAB'; tab: 'notes' | 'low-confidence' }
  | { type: 'SHOW_CONTEXT_MENU'; menu: ContextMenuState }
  | { type: 'HIDE_CONTEXT_MENU' }
  | { type: 'SHOW_NOTE_MODAL'; modal: NoteModalState }
  | { type: 'HIDE_NOTE_MODAL' }
  | { type: 'SET_NOTES'; notes: Note[] }
  | { type: 'TOGGLE_AXIAL_LABEL'; segmentId: SegmentId; code: AxialCode }
  | { type: 'SET_ERROR'; error: string }
  | { type: 'SET_LOADING'; loading: boolean }
  | { type: 'SET_SESSIONS'; sessions: SessionSummary[] };

const initialState: ValidatorState = {
  sessions: [],
  segments: [],
  notes: [],
  axialLabels: [],
  currentTime: 0,
  activeSegmentIndex: -1,
  duration: 0,
  playing: false,
  playbackRate: 1,
  filters: { silenceGap: false, nearZero: false, duplicates: false },
  duplicateIds: new Set(),
  missedSpeech: null,
  missedError: null,
  showMissed: false,
  missedCursor: -1,
  missedHearings: null,
  missedHearingsError: null,
  missedMinGap: null,
  drawerOpen: false,
  drawerTab: 'notes',
  contextMenu: { visible: false, x: 0, y: 0, target: null },
  noteModal: { visible: false, target: null, existingNote: null },
  speakerNames: new Map(),
  speakerColorMap: new Map(),
  loading: true,
  error: null,
};

function reducer(state: ValidatorState, action: ValidatorAction): ValidatorState {
  switch (action.type) {
    case 'LOAD_SESSION':
      return {
        ...state,
        segments: action.segments,
        notes: action.notes,
        axialLabels: action.axialLabels,
        speakerNames: action.speakerNames,
        speakerColorMap: action.speakerColorMap,
        duplicateIds: findDuplicateSegmentIds(action.segments),
        loading: false,
        error: null,
      };

    case 'SET_TIME':
      return { ...state, currentTime: action.time };

    case 'SET_ACTIVE_SEGMENT':
      return state.activeSegmentIndex === action.index
        ? state
        : { ...state, activeSegmentIndex: action.index };

    case 'SET_DURATION':
      return { ...state, duration: action.duration };

    case 'SET_PLAYING':
      return { ...state, playing: action.playing };

    case 'SET_PLAYBACK_RATE':
      return { ...state, playbackRate: action.rate };

    case 'TOGGLE_FILTER':
      return {
        ...state,
        filters: {
          ...state.filters,
          [action.filter]: !state.filters[action.filter],
        },
      };

    case 'SET_MISSED_SPEECH':
      // A new candidate list invalidates the stepper position.
      return { ...state, missedSpeech: action.candidates, missedMinGap: action.minGap, missedError: null, missedCursor: -1 };

    case 'SET_MISSED_ERROR':
      return { ...state, missedSpeech: null, missedError: action.error, missedCursor: -1 };

    case 'SET_MISSED_HEARINGS':
      return { ...state, missedHearings: action.hearings, missedHearingsError: null };

    case 'SET_MISSED_HEARINGS_ERROR':
      return { ...state, missedHearings: null, missedHearingsError: action.error };

    case 'TOGGLE_MISSED':
      // No candidates loaded (fetching or errored) → nothing to toggle; keeps
      // the M shortcut honest with the disabled toolbar button.
      if (state.missedSpeech === null) return state;
      return { ...state, showMissed: !state.showMissed };

    case 'SET_MISSED_CURSOR':
      return { ...state, missedCursor: action.index };

    case 'TOGGLE_DRAWER':
      return { ...state, drawerOpen: !state.drawerOpen };

    case 'SET_DRAWER_TAB':
      return { ...state, drawerTab: action.tab };

    case 'SHOW_CONTEXT_MENU':
      return { ...state, contextMenu: action.menu };

    case 'HIDE_CONTEXT_MENU':
      return state.contextMenu.visible
        ? { ...state, contextMenu: { ...state.contextMenu, visible: false } }
        : state;

    case 'SHOW_NOTE_MODAL':
      return { ...state, noteModal: action.modal, contextMenu: { ...state.contextMenu, visible: false } };

    case 'HIDE_NOTE_MODAL':
      return state.noteModal.visible
        ? { ...state, noteModal: { ...state.noteModal, visible: false } }
        : state;

    case 'SET_NOTES':
      return { ...state, notes: action.notes };

    case 'TOGGLE_AXIAL_LABEL': {
      const now = new Date().toISOString();
      const existingIdx = state.axialLabels.findIndex((l) => l.segmentId === action.segmentId);
      const existing = existingIdx >= 0 ? state.axialLabels[existingIdx] : null;

      // Compute the next codes array per the toggle rules.
      let nextCodes: AxialCode[];
      if (!existing) {
        // No label yet — start with just this code.
        nextCodes = [action.code];
      } else if (action.code === 'NotA') {
        // NotA toggles itself; selecting NotA always wipes M codes.
        nextCodes = existing.codes.includes('NotA') ? [] : ['NotA'];
      } else {
        // Any M code clears NotA first, then toggles itself.
        const without = existing.codes.filter((c) => c !== 'NotA');
        nextCodes = without.includes(action.code)
          ? without.filter((c) => c !== action.code)
          : [...without, action.code];
      }

      // Empty codes → remove the row (segment becomes unlabeled).
      if (nextCodes.length === 0) {
        if (existingIdx < 0) return state;
        const next = [...state.axialLabels];
        next.splice(existingIdx, 1);
        return { ...state, axialLabels: next };
      }

      // Update existing row.
      if (existingIdx >= 0) {
        const next = [...state.axialLabels];
        next[existingIdx] = {
          ...next[existingIdx],
          codes: nextCodes,
          updatedAt: now,
        };
        return { ...state, axialLabels: next };
      }

      // New row.
      return {
        ...state,
        axialLabels: [
          ...state.axialLabels,
          {
            segmentId: action.segmentId,
            codes: nextCodes,
            createdAt: now,
            updatedAt: now,
          },
        ],
      };
    }

    case 'SET_ERROR':
      return { ...state, error: action.error, loading: false };

    case 'SET_LOADING':
      return { ...state, loading: action.loading };

    case 'SET_SESSIONS':
      return { ...state, sessions: action.sessions };

    default:
      return state;
  }
}

export interface DerivedState {
  filterCounts: { silenceGap: number; nearZero: number; duplicates: number };
  segmentFilterReasons: Map<SegmentId, string[]>;
  segmentAllFilterReasons: Map<SegmentId, string[]>;
  notesBySegment: Map<SegmentId, Note[]>;
  labelsBySegment: Map<SegmentId, AxialLabel>;
  labeledCount: number;
}

export function useValidatorState() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const derived = useMemo<DerivedState>(() => {
    const filterCounts = { silenceGap: 0, nearZero: 0, duplicates: 0 };
    const segmentFilterReasons = new Map<SegmentId, string[]>();
    const segmentAllFilterReasons = new Map<SegmentId, string[]>();

    for (const seg of state.segments) {
      const active = getFilterReasons(seg, state.duplicateIds, state.filters);
      const all = getAllFilterReasons(seg, state.duplicateIds);

      segmentFilterReasons.set(seg.id, active);
      segmentAllFilterReasons.set(seg.id, all);

      for (const r of all) {
        if (r === 'silence-gap') filterCounts.silenceGap++;
        else if (r === 'near-zero') filterCounts.nearZero++;
        else if (r === 'duplicate') filterCounts.duplicates++;
      }
    }

    const notesBySegment = new Map<SegmentId, Note[]>();
    for (const note of state.notes) {
      if (note.segmentId !== null) {
        const existing = notesBySegment.get(note.segmentId) ?? [];
        existing.push(note);
        notesBySegment.set(note.segmentId, existing);
      }
    }

    const labelsBySegment = new Map<SegmentId, AxialLabel>();
    for (const label of state.axialLabels) {
      labelsBySegment.set(label.segmentId, label);
    }

    return {
      filterCounts,
      segmentFilterReasons,
      segmentAllFilterReasons,
      notesBySegment,
      labelsBySegment,
      labeledCount: state.axialLabels.length,
    };
  }, [state.segments, state.duplicateIds, state.filters, state.notes, state.axialLabels]);

  return { state, dispatch, derived };
}
