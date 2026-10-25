/**
 * NamePassPage — build a per-session gold name key by ear, Opus-assisted (#4).
 *
 * The transcript reads clean, with Opus's best-guess names shown IN PLACE. Clicking a
 * word plays the audio from there and SELECTS it (never writes); every action is on the right:
 * fix all occurrences of a spelling, or override just this one (the per-occurrence
 * split), mark it not-a-name, clear it, or hear it. A copyable marker on the selection
 * lets Choksi hand one exact occurrence back to Claude in chat. His ear is the gate;
 * drafts live in a separate file; each answer is tagged how it got there.
 *
 * Word spans are closure-free (data-attrs) + memoized; a delegated click reads them.
 * The 60fps play-along toggles `.np-active` on the DOM node directly — so after a rule
 * re-renders the transcript (and React reconciles that class away), we reset
 * `activeOccRef` and let the next frame restore it.
 */

import { useCallback, useEffect, useMemo, useRef, useState, Fragment } from 'react';
import { useParams, Link } from 'react-router-dom';
import WaveformPlayer, { type WaveformPlayerHandle } from '../components/WaveformPlayer';
import { getSession, getNamePass, commitNameTruth, saveNameTruthField, audioURL } from '../api/client';
import { formatTime } from '../utils/time';
import type {
  SessionDetail,
  NamePassPayload,
  NamePassCard,
  NameTruthItem,
  NamePassSuggestion,
  TranscriptSegment,
} from '../types';
import './NamePass.css';

const EMPTY_ITEM: NameTruthItem = { category: '', true_spelling: '', notes: '', provenance: '' };

interface SelectedWord {
  occId: string;
  cleaned: string;
  wordText: string;
  start: number | null;
  window: [number, number] | null;
}

// Mirror of detectors/phonetics.clean() — a clicked word keys the same item the backend would.
function cleanTok(tok: string): string {
  let t = tok.trim().toLowerCase();
  t = t.replace(/[^a-z'’]+$/g, '');
  t = t.replace(/^[^a-z'’]+/g, '');
  t = t.replace(/[’']s?$/g, '');
  t = t.replace(/[^a-z]/g, '');
  return t;
}

// '<segId>:<wi>' -> [segId, wi]. Split on the LAST colon: a segment id may be an int OR a gap
// id ('gap_782.052'), which contains no colon but is not a number either. Mirrors the scorer's
// split_occ and the API's _split_occ_id — one idiom, one place, so a change to the id format
// can't be half-applied.
function splitOcc(occId: string): [string, number] {
  const cut = occId.lastIndexOf(':');
  return [occId.slice(0, cut), Number(occId.slice(cut + 1))];
}

// Trailing possessive + punctuation of a raw token (the "'s," in "Marthie's,") — reattached
// when we swap the garble for a name. The ruling names the character; the grammar around
// it (possessive, sentence boundary) belongs to the sentence and must survive the read —
// "Marthie's head" corrects to "Marta's head", never "Marta head".
function trailingSuffix(raw: string): string {
  const punct = raw.match(/[^A-Za-z]+$/)?.[0] ?? '';
  const core = punct ? raw.slice(0, -punct.length) : raw;
  const poss = core.match(/['’]s?$/)?.[0] ?? '';
  return poss + punct;
}

const fmtT = (s: number | null): string => (s === null || s === undefined ? '—' : formatTime(s));

function isGap(seg: TranscriptSegment): boolean {
  return seg._source === 'diarization_gap' || !(seg.words && seg.words.length);
}

export default function NamePassPage() {
  const { id } = useParams<{ id: string }>();
  const sessionId = id ?? '';

  const [session, setSession] = useState<SessionDetail | null>(null);
  const [payload, setPayload] = useState<NamePassPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [keyState, setKeyState] = useState<Record<string, NameTruthItem>>({});
  const [occState, setOccState] = useState<Record<string, string>>({}); // occId -> true_name override
  const [occScope, setOccScope] = useState<Record<string, string>>({}); // occId -> 'name'|'speech'|'tail'|''
  const [selected, setSelected] = useState<SelectedWord | null>(null);
  const [saveStatus, setSaveStatus] = useState('');

  const playerRef = useRef<WaveformPlayerHandle>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const activeOccRef = useRef<string | null>(null);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const playedRef = useRef<Set<number>>(new Set());
  const [coveredSecs, setCoveredSecs] = useState(0);
  const saveTimers = useRef<Record<string, number>>({});

  useEffect(() => {
    if (!sessionId) return;
    let alive = true;
    Promise.all([getSession(sessionId), getNamePass(sessionId)])
      .then(([s, p]) => {
        if (!alive) return;
        setError(null);
        setSession(s);
        setPayload(p);
        setKeyState(p.seed.items ?? {});
        const occs = Object.entries(p.seed.occurrences ?? {});
        setOccState(Object.fromEntries(occs.map(([k, v]) => [k, v.true_name])));
        setOccScope(Object.fromEntries(occs.map(([k, v]) => [k, v.scope ?? ''])));
      })
      .catch((e) => { if (alive) setError(e instanceof Error ? e.message : String(e)); });
    return () => { alive = false; };
  }, [sessionId]);

  const segments = useMemo(
    () => (session?.transcript?.segments ?? []) as TranscriptSegment[],
    [session],
  );
  const stories = useMemo(() => session?.transcript?._stories ?? [], [session]);
  const candidateIndex = useMemo(() => payload?.candidate_index ?? {}, [payload]);
  const suggestions = useMemo<Record<string, NamePassSuggestion>>(() => payload?.suggestions ?? {}, [payload]);
  const categories = payload?.categories ?? [];

  const cardByCleaned = useMemo(() => {
    const m = new Map<string, NamePassCard>();
    for (const c of payload?.cards ?? []) m.set(c.cleaned, c);
    return m;
  }, [payload]);

  // SPANS: which words do NOT draw themselves, because they belong to a name that starts at
  // an earlier word. The head draws the whole span, so an absorbed word must not draw again
  // ("Cruel Baby baby"). Two sources, one map — mirroring the scorer's _span_map exactly, so
  // the tool and the ruler agree by construction.
  //
  //   PHRASE  a bigram card lists only its HEAD; the next word is the tail it absorbed —
  //           unless that tail heads a card of its own (chained bigrams, 'new york' +
  //           'york city'), which keeps its render and leaves its head out of `phraseHeads`.
  //   TAIL    a word HE marked "part of the name before this" (scope 'tail'). Any length,
  //           and it reaches heads no sweep could ever card: the head of "you dished there."
  //           -> Yudhishthira is the function word "you" (#5).
  //
  // A marked tail walks BACK past anything already absorbed, so a span is always a
  // contiguous run of words — mark the 2nd and 3rd words of a three-word name and both land
  // on the 1st.
  const spanFold = useMemo(() => {
    const tails = new Map<string, string>();          // absorbed occId -> the head of its span
    const phraseTail = new Set<string>();
    for (const c of payload?.cards ?? []) {
      if (!c.phrase) continue;
      for (const o of c.occurrences) {
        const tailId = `${o.seg_id}:${o.wi + 1}`;
        if (!(tailId in candidateIndex)) { tails.set(tailId, o.occ_id); phraseTail.add(tailId); }
      }
    }
    const phraseHeads = new Set([...phraseTail].map((t) => tails.get(t)!));

    const marked = new Set(Object.entries(occScope).filter(([, s]) => s === 'tail').map(([k]) => k));
    // A tail with nothing before it in its segment has nothing to be a tail OF. The button
    // can't create one (it hides on a first word), but a re-run transcript can shift indices
    // under an existing key. The scorer calls that a BROKEN TAIL and refuses to print any
    // number — so the tool must not go on showing a clean transcript over a dead ruler.
    const orphans = new Set<string>();
    for (const oid of marked) {
      const [seg, wi] = splitOcc(oid);
      let head: string | null = null;
      for (let j = wi - 1; j >= 0; j -= 1) {
        const prev = `${seg}:${j}`;
        if (!marked.has(prev) && !phraseTail.has(prev)) { head = prev; break; }
      }
      if (head) tails.set(oid, head);
      else orphans.add(oid);
    }

    const heads = new Map<string, string[]>();        // head occId -> its tails, in word order
    for (const [t, h] of tails) heads.set(h, [...(heads.get(h) ?? []), t]);
    for (const ts of heads.values()) ts.sort((a, b) => splitOcc(a)[1] - splitOcc(b)[1]);
    return { tails, heads, phraseHeads, marked, orphans };
  }, [payload, candidateIndex, occScope]);

  // The nearest word BEFORE this one that isn't already absorbed — i.e. the head this word
  // would join if he marked it a tail. Null at the start of a segment (nothing to be a tail
  // OF, which the scorer refuses to resolve quietly).
  const headOf = useCallback(
    (occId: string): string | null => {
      const [seg, wi] = splitOcc(occId);
      for (let j = wi - 1; j >= 0; j -= 1) {
        const prev = `${seg}:${j}`;
        if (!spanFold.tails.has(prev)) return prev;
      }
      return null;
    },
    [spanFold],
  );

  const segById = useMemo(
    () => new Map(segments.map((s) => [String(s.id), s])),
    [segments],
  );

  const wordAt = useCallback(
    (occId: string): string => {
      const [seg, wi] = splitOcc(occId);
      const w = segById.get(seg)?.words?.[wi];
      return w ? w.word.trim() : occId;
    },
    [segById],
  );

  // What a click on one spot selects. A HEAD stands for its whole span — the pair/triple
  // text and a span-wide hear window — which is what makes an absorbed word reachable
  // through the word that swallowed it.
  const describeOcc = useCallback(
    (occId: string): SelectedWord | null => {
      const [segId, wi] = splitOcc(occId);
      const seg = segById.get(segId);
      const w = seg?.words?.[wi];
      if (!w) return null;
      const cleaned = candidateIndex[occId] ?? cleanTok(w.word);
      if (!cleaned) return null;
      const tailWs = (seg.words ?? []).slice(wi + 1, wi + 1 + (spanFold.heads.get(occId)?.length ?? 0));
      const wordText = tailWs.length ? [w, ...tailWs].map((x) => x.word.trim()).join(' ') : w.word;
      const start = typeof w.start === 'number' ? w.start : null;
      const last = tailWs[tailWs.length - 1] ?? w;
      const end = typeof last.end === 'number' ? last.end : null;
      return { occId, cleaned, wordText, start,
               window: start !== null && end !== null ? [start, end] : null };
    },
    [segById, candidateIndex, spanFold],
  );

  // Flat, time-sorted word list for the play-along binary search.
  const timedWords = useMemo(() => {
    const out: { occ: string; start: number; end: number }[] = [];
    for (const seg of segments) {
      (seg.words ?? []).forEach((w, wi) => {
        if (typeof w.start === 'number' && typeof w.end === 'number') {
          const occId = `${seg.id}:${wi}`;
          // A hidden tail extends its head's interval — bridging the breath between the
          // words so the drawn span stays lit through the whole name. A span is contiguous,
          // so each successive tail keeps extending the same (previous) entry.
          const head = spanFold.tails.get(occId);
          const prev = out[out.length - 1];
          if (head && prev?.occ === head) prev.end = w.end;
          else out.push({ occ: head ?? occId, start: w.start, end: w.end });
        }
      });
    }
    return out;
  }, [segments, spanFold]);

  const findActiveOcc = useCallback(
    (t: number): string | null => {
      let lo = 0, hi = timedWords.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const w = timedWords[mid];
        if (t < w.start) hi = mid - 1;
        else if (t > w.end) lo = mid + 1;
        else return w.occ;
      }
      return null;
    },
    [timedWords],
  );

  const handleTimeUpdate = useCallback(
    (time: number) => {
      const sec = Math.floor(time);
      if (!playedRef.current.has(sec)) {
        playedRef.current.add(sec);
        setCoveredSecs(playedRef.current.size);
      }
      const occ = findActiveOcc(time);
      if (occ !== activeOccRef.current) {
        const root = transcriptRef.current;
        if (root) {
          if (activeOccRef.current) {
            root.querySelector(`[data-occ="${CSS.escape(activeOccRef.current)}"]`)?.classList.remove('np-active');
          }
          if (occ) {
            const el = root.querySelector(`[data-occ="${CSS.escape(occ)}"]`);
            el?.classList.add('np-active');
            el?.scrollIntoView({ block: 'nearest' });
          }
        }
        activeOccRef.current = occ;
      }
    },
    [findActiveOcc],
  );

  // A rule/selection re-renders the memoized transcript, but React only strips the
  // imperative `.np-active` from spans whose className prop actually changed — on every
  // other span it survives reconciliation. Sweep it off everywhere and forget the cached
  // occ so the next 60fps tick re-adds exactly the current one.
  useEffect(() => {
    transcriptRef.current?.querySelectorAll('.np-active').forEach((el) => el.classList.remove('np-active'));
    activeOccRef.current = null;
  }, [keyState, occState, occScope, selected]);

  // --- saving ---
  const debouncedSave = useCallback((key: string, fn: () => Promise<unknown>) => {
    window.clearTimeout(saveTimers.current[key]);
    setSaveStatus('saving…');
    saveTimers.current[key] = window.setTimeout(() => {
      fn().then(() => setSaveStatus('saved ✓')).catch((e) => setSaveStatus(e instanceof Error ? e.message : 'save failed'));
    }, 500);
  }, []);

  const provenanceFor = useCallback(
    (cleaned: string, value: string): string => {
      const sug = suggestions[cleaned]?.suggested;
      if (!sug) return 'ear-cold';
      return value.trim() === sug.trim() ? 'suggestion-confirmed' : 'ear-corrected';
    },
    [suggestions],
  );

  // Rule ALL occurrences of a spelling (the item's true_spelling + provenance).
  //
  // A spelling with no category is `unjudged` — the scorer drops it and the ear-work counts
  // for nothing. So when he accepts a DRAFT, the draft's name-kind rides along —
  // but only to fill a blank: a category he set himself is never overwritten, and the
  // dropdown can still change it afterwards. His ear stays the gate; this only decides what
  // an un-touched default lands as, countable instead of uncountable.
  const ruleSpelling = useCallback(
    (cleaned: string, value: string) => {
      // A spelling on a NOT-A-NAME is contradictory — "not a name" means leave this word
      // alone — and both the render and the scorer discard it. It used to be accepted and
      // silently thrown away, which cost him the "Jai Sri Krishna" blessing. The
      // box is disabled for these; this refuses the write even if something else reaches it.
      if (keyState[cleaned]?.category === 'not-a-name') return;
      const provenance = value ? provenanceFor(cleaned, value) : '';
      const fields: Partial<NameTruthItem> = { true_spelling: value, provenance };
      const draftCat = suggestions[cleaned]?.category;
      if (draftCat && provenance === 'suggestion-confirmed' && !keyState[cleaned]?.category) {
        fields.category = draftCat;
      }
      setKeyState((prev) => ({ ...prev, [cleaned]: { ...(prev[cleaned] ?? EMPTY_ITEM), ...fields } as NameTruthItem }));
      debouncedSave(`item:${cleaned}`, () => commitNameTruth(sessionId, { [cleaned]: fields }));
    },
    [sessionId, provenanceFor, debouncedSave, suggestions, keyState],
  );

  // WHICH JOB a per-spot ruling records — stamped, never left for the scorer to guess from
  // punctuation. On a NOT-A-NAME item it is the true SPEECH here: "not a name" means leave
  // this word alone, so a NAME ruling there would contradict it. He heard "Cheshi Kresha
  // Martie" = "Jai Sri Krishna" — the words are true, the tokens are not names, and that is
  // exactly how the Goodnight key already records the same blessing.
  const scopeForOverride = useCallback(
    (cleaned: string) => (keyState[cleaned]?.category === 'not-a-name' ? 'speech' : 'name'),
    [keyState],
  );

  // Override JUST this occurrence (occurrences[occId].true_name) — the per-occurrence split.
  const ruleOccurrence = useCallback(
    (occId: string, cleaned: string, value: string) => {
      const scope = value ? scopeForOverride(cleaned) : '';
      setOccState((prev) => ({ ...prev, [occId]: value }));
      setOccScope((prev) => ({ ...prev, [occId]: scope }));
      debouncedSave(`occ:${occId}`, () => Promise.all([
        saveNameTruthField(sessionId, { kind: 'occ', id: occId, field: 'true_name', value }),
        saveNameTruthField(sessionId, { kind: 'occ', id: occId, field: 'scope', value: scope }),
      ]));
    },
    [sessionId, debouncedSave, scopeForOverride],
  );

  // Mark this word as part of the name that starts BEFORE it (scope 'tail'). Whisper splits
  // a name across several word-tokens — "you dished there." is Yudhishthira — and without
  // this the key could only rule the first of them, so a fixer that wrote the name there and
  // left the leftovers standing scored FIXED (#5).
  //
  // The word vanishes into its head's span, so we select the head: that is where the name
  // gets typed, and it is the only way back to release a mis-tap.
  const markTail = useCallback(
    (occId: string) => {
      const head = headOf(occId);
      setOccScope((prev) => ({ ...prev, [occId]: 'tail' }));
      setOccState((prev) => ({ ...prev, [occId]: '' }));   // a tail owes no name of its own
      setSaveStatus('joining…');
      Promise.all([
        saveNameTruthField(sessionId, { kind: 'occ', id: occId, field: 'scope', value: 'tail' }),
        saveNameTruthField(sessionId, { kind: 'occ', id: occId, field: 'true_name', value: '' }),
      ])
        .then(() => setSaveStatus('joined ✓'))
        .catch((e) => setSaveStatus(e instanceof Error ? e.message : 'save failed'));
      setSelected(head ? describeOcc(head) : null);
    },
    [sessionId, headOf, describeOcc],
  );

  // Give an absorbed word back its own decision — the undo for a mis-tapped tail.
  const releaseTail = useCallback(
    (occId: string) => {
      setOccScope((prev) => ({ ...prev, [occId]: '' }));
      setSaveStatus('releasing…');
      saveNameTruthField(sessionId, { kind: 'occ', id: occId, field: 'scope', value: '' })
        .then(() => setSaveStatus('released ✓'))
        .catch((e) => setSaveStatus(e instanceof Error ? e.message : 'release failed'));
    },
    [sessionId],
  );

  const setCategory = useCallback(
    (cleaned: string, value: string, occId?: string) => {
      setKeyState((prev) => ({ ...prev, [cleaned]: { ...(prev[cleaned] ?? EMPTY_ITEM), category: value } }));
      setSaveStatus('saving…');
      const saves: Promise<unknown>[] = [commitNameTruth(sessionId, { [cleaned]: { category: value } })];
      // A per-spot ruling's JOB depends on the item's kind (see scopeForOverride), so
      // changing the kind has to re-stamp the scope — otherwise ruling a word not-a-name
      // AFTER typing its true words would leave them scored as a name the fixer must write.
      const held = (occState[occId ?? ''] ?? '').trim();
      if (occId && held && occScope[occId] !== 'tail') {
        const scope = value === 'not-a-name' ? 'speech' : 'name';
        setOccScope((prev) => ({ ...prev, [occId]: scope }));
        saves.push(saveNameTruthField(sessionId, { kind: 'occ', id: occId, field: 'scope', value: scope }));
      }
      Promise.all(saves)
        .then(() => setSaveStatus('saved ✓')).catch((e) => setSaveStatus(e instanceof Error ? e.message : 'save failed'));
    },
    [sessionId, occState, occScope],
  );

  // Revert this word: clear its per-occ override AND its whole ruling — spelling, provenance
  // AND category — back to the suggestion/raw.
  //
  // The category has to go too. Once a bare category renders as "ruled" (it is one), leaving
  // it behind meant Clear only half-reverted: the word stayed green, still counted as set,
  // and nothing said so — while this function's own contract promises "back to the
  // suggestion/raw".
  const clearWord = useCallback(
    (cleaned: string, occId: string) => {
      const blank = { true_spelling: '', provenance: '', category: '' };
      // Clearing a HEAD releases the whole span. Leaving its tails marked would strand them
      // on a word that now rules no name — the scorer refuses to score that (a "LOST TAIL"),
      // and Clear's contract is "back to the suggestion/raw", not "back to half of it".
      const spanTails = (spanFold.heads.get(occId) ?? []).filter((t) => spanFold.marked.has(t));
      setOccState((prev) => ({ ...prev, [occId]: '' }));
      setOccScope((prev) => {
        const next = { ...prev, [occId]: '' };
        for (const t of spanTails) next[t] = '';
        return next;
      });
      setKeyState((prev) => ({ ...prev, [cleaned]: { ...(prev[cleaned] ?? EMPTY_ITEM), ...blank } }));
      setSaveStatus('clearing…');
      Promise.all([
        saveNameTruthField(sessionId, { kind: 'occ', id: occId, field: 'true_name', value: '' }),
        saveNameTruthField(sessionId, { kind: 'occ', id: occId, field: 'scope', value: '' }),
        ...spanTails.map((t) => saveNameTruthField(sessionId, { kind: 'occ', id: t, field: 'scope', value: '' })),
        commitNameTruth(sessionId, { [cleaned]: blank }),
      ]).then(() => setSaveStatus('cleared ✓')).catch((e) => setSaveStatus(e instanceof Error ? e.message : 'clear failed'));
    },
    [sessionId, spanFold],
  );

  const copyMarker = useCallback(() => {
    if (!selected) return;
    const marker = `${selected.wordText.trim()} @${fmtT(selected.start)} ⟨${selected.occId}⟩`;
    navigator.clipboard?.writeText(marker)
      .then(() => setSaveStatus('marker copied'))
      .catch(() => setSaveStatus('copy blocked — select it manually'));
  }, [selected]);

  const selectOcc = useCallback((occId: string, cleaned: string, wordText: string, start: number | null, window: [number, number] | null) => {
    setSelected({ occId, cleaned, wordText, start, window });
  }, []);

  // Click-to-listen: jump the player to a word and play ON from there (continuous —
  // distinct from `hear` below, which plays just the one occurrence's window and stops).
  // Untimed words (no start) just select; there's nothing to seek to.
  const playFrom = useCallback((start: number | null) => {
    const p = playerRef.current;
    if (p && start !== null) { p.setTime(start); p.play(); }
  }, []);

  // Delegated click: SELECTS the word and plays the audio from there. It never mutates —
  // playback isn't a write, and every ruling still happens on the panel.
  const onTranscriptClick = useCallback(
    (e: React.MouseEvent) => {
      const span = (e.target as HTMLElement).closest('.np-word') as HTMLElement | null;
      const occId = span?.getAttribute('data-occ');
      if (!occId) return;
      const d = describeOcc(occId);
      if (!d) return;
      setSelected(d);
      playFrom(d.start);
    },
    [describeOcc, playFrom],
  );

  const hear = useCallback((win: [number, number] | null, start: number | null) => {
    const p = playerRef.current;
    if (!p) return;
    if (win) p.playWord(win[0], win[1]);
    else if (start !== null) { p.setTime(start); p.play(); }
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.code === 'Space' && !/^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) {
        e.preventDefault();
        playerRef.current?.playPause();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // --- "Mark heard" ---
  // Only names he has NOT ruled at all. A draft may fill a blank; it may never write over
  // his ear.
  //
  // The old test was `category !== 'not-a-name'`, which let a bulk-accept overwrite a
  // CATEGORY-ONLY ruling — the shape that means "this name is correct exactly as spoken"
  // (michael -> improvised, no spelling; foxy, biffy). Whisper's word stands, only the
  // bucket is his. Under the old test those were still "un-accepted", so Mark heard would
  // stamp the draft's spelling on top of them: a machine guess silently replacing a human
  // ruling in the gold key. Any category at all now means hands off.
  const remainingSuggested = useMemo(
    () =>
      (payload?.cards ?? [])
        .map((c) => c.cleaned)
        .filter((c) => suggestions[c] && !keyState[c]?.true_spelling && !keyState[c]?.category),
    [payload, suggestions, keyState],
  );

  // With nothing left to accept this still stamps `_heard_confirmed` — a fully hand-ruled
  // key would otherwise have NO way to get its ear-gate stamp (the button used to disable).
  const markHeard = useCallback(() => {
    const items: Record<string, Partial<NameTruthItem>> = {};
    for (const c of remainingSuggested) {
      items[c] = { true_spelling: suggestions[c].suggested, provenance: 'suggestion-confirmed' };
      // The draft's name-kind rides along, or this bulk-accept commits a pile of `unjudged`
      // items the scorer silently drops — 18 of 49 on the 20260122 key. Never
      // overwrites a category he set himself.
      const draftCat = suggestions[c].category;
      if (draftCat && !keyState[c]?.category) items[c].category = draftCat;
    }
    setKeyState((prev) => {
      const next = { ...prev };
      for (const [c, f] of Object.entries(items)) next[c] = { ...(next[c] ?? EMPTY_ITEM), ...f } as NameTruthItem;
      return next;
    });
    setSaveStatus('committing…');
    commitNameTruth(sessionId, items, true)
      .then((r) => setSaveStatus(`heard ✓ — accepted ${r.committed}`))
      .catch((e) => setSaveStatus(e instanceof Error ? e.message : 'commit failed'));
  }, [sessionId, remainingSuggested, suggestions, keyState]);

  // --- tallies ---
  // "Set" means COUNTABLE: a name is only really ruled once it has a category, because the
  // scorer drops a name with no bucket. Counting a bare true_spelling as "set"
  // over-reported progress and hid exactly the items that were about to be thrown away.
  const totalCandidates = payload?.cards.length ?? 0;
  const confirmed = (payload?.cards ?? []).filter((c) => keyState[c.cleaned]?.category).length;
  const needsBucket = (payload?.cards ?? []).filter(
    (c) => keyState[c.cleaned]?.true_spelling && !keyState[c.cleaned]?.category,
  ).length;
  const correctedCount = Object.values(keyState).filter((v) => v.provenance === 'ear-corrected').length;
  const coveragePct = duration > 0 ? Math.min(100, Math.round((coveredSecs / duration) * 100)) : 0;

  // --- memoized transcript: names shown in place, styled by state ---
  const transcriptEls = useMemo(() => {
    return segments.map((seg, i) => {
      const story = seg._story;
      const prevStory = i > 0 ? segments[i - 1]._story : undefined;
      const showDivider = typeof story === 'number' && story !== prevStory && stories[story];
      const divider = showDivider ? (
        <div className="np-story" key={`d${i}`}>
          {stories[story!].title || 'Story'}
          {stories[story!].world ? <span className="np-story-world"> · {stories[story!].world}</span> : null}
        </div>
      ) : null;

      if (isGap(seg)) {
        return (
          <Fragment key={i}>
            {divider}
            <span className="np-gap">[unintelligible]</span>{' '}
          </Fragment>
        );
      }

      return (
        <Fragment key={i}>
          {divider}
          <span className="np-seg">
            {(seg.words ?? []).map((w, wi) => {
              const occId = `${seg.id}:${wi}`;
              if (spanFold.tails.has(occId)) return null;  // drawn by the head of its span
              const candCleaned = candidateIndex[occId];
              const cleaned = candCleaned ?? cleanTok(w.word);
              const isCand = candCleaned !== undefined;
              const item = cleaned ? keyState[cleaned] : undefined;
              const occOverride = occState[occId];
              const sug = isCand && cleaned ? suggestions[cleaned] : undefined;

              // A head speaks for every word it absorbed: the raw text is the whole SPAN,
              // and a replacement spelling reattaches the LAST word's trailing punctuation
              // (the ruling names the character; the sentence's grammar must survive).
              //
              // A chained phrase head absorbed nothing — its tail heads another card — so it
              // keeps its own raw word: painting the pair spelling into a one-word slot would
              // double the shared word ('Cruel Baby Baby Shark').
              const spanTails = spanFold.heads.get(occId) ?? [];
              const tailWs = (seg.words ?? []).slice(wi + 1, wi + 1 + spanTails.length);
              const keepRaw = isCand && cardByCleaned.get(cleaned)?.phrase === true
                && !spanFold.phraseHeads.has(occId);
              const raw = tailWs.length ? [w, ...tailWs].map((x) => x.word.trim()).join(' ') : w.word;
              const punct = trailingSuffix((tailWs[tailWs.length - 1] ?? w).word);

              let text = raw;
              const cls = ['np-word'];
              // He joined words into one name here. Worth SEEING — it is the one ruling the
              // transcript's own text gives no hint of, and an accidental one is invisible
              // otherwise. Amber when the head rules no name yet: the tail is hanging off
              // nothing, and the scorer will refuse to score it.
              if (spanTails.some((t) => spanFold.marked.has(t))) {
                cls.push(occState[occId] || item?.true_spelling || item?.category
                  ? 'np-joined' : 'np-joined-unruled');
              }
              // A tail with nothing before it to join. It draws itself (it belongs to no
              // span), so without this it would look like an ordinary untouched word while
              // the scorer refused to print a single number over it.
              if (spanFold.orphans.has(occId)) cls.push('np-joined-unruled');
              // A per-spot override is the MOST specific ruling, so it wins even when the
              // item is not-a-name: "Krishna" is not a name, yet at this spot the true words
              // are "Jai Sri Krishna". The item stays not-a-name (name score); the override
              // shows the true words here (transcript truth). Keep this branch first.
              if (occOverride) {
                if (!keepRaw) text = occOverride + punct;
                cls.push('np-cand', 'np-corrected');
              } else if (item?.category === 'not-a-name') {
                cls.push('np-cand', 'np-nota');
              } else if (item?.true_spelling || item?.category) {
                // RULED — by a spelling, by a bucket, or by both.
                //
                // Two bugs met here. (1) "Confirmed" used to key off true_spelling alone, so
                // a name that was ALREADY CORRECT (Thomas, Sodor) — where the only thing to
                // set is the bucket — went on looking exactly like a word nobody had touched.
                // A category IS a decision, and now it shows as one.
                //
                // (2) But a spelling with NO bucket is `unjudged`: the scorer drops it and the
                // ear-work scores as nothing. Painting that green would be the
                // worse lie — the tool calling it done at the moment it's being discarded. So
                // it renders ruled AND flagged, until it has the bucket that makes it count.
                if (!keepRaw && item.true_spelling) text = item.true_spelling + punct;
                cls.push('np-cand', item.provenance === 'ear-corrected' ? 'np-corrected' : 'np-confirmed');
                if (!item.category) cls.push('np-nobucket');
              } else if (sug) {
                if (!keepRaw) text = sug.suggested + punct;
                cls.push('np-cand', 'np-suggested');
                if (sug.confidence === 'low') cls.push('np-lowconf');
              } else if (isCand) {
                cls.push('np-cand', 'np-tocheck');
              }
              if (selected?.occId === occId) cls.push('np-selected');

              const heardOrig = text.trim() !== raw.trim() ? `heard: ${raw.trim()}` : undefined;
              return (
                <span key={wi} className={cls.join(' ')} data-occ={occId} data-si={i} data-wi={wi} title={heardOrig}>
                  {text.trim() === raw.trim() ? `${raw} ` : `${text} `}
                </span>
              );
            })}
          </span>{' '}
        </Fragment>
      );
    });
  }, [segments, stories, candidateIndex, spanFold, cardByCleaned, suggestions, keyState, occState, selected]);

  if (error) return <div className="np-page np-msg">Couldn’t load name pass: {error}</div>;
  if (!session || !payload) return <div className="np-page np-msg">Loading…</div>;

  // panel data for the selected occurrence
  const sel = selected;
  const selItem = sel ? keyState[sel.cleaned] : undefined;
  const selSug = sel ? suggestions[sel.cleaned] : undefined;
  const selOccVal = sel ? (occState[sel.occId] ?? '') : '';
  const selCard = sel ? cardByCleaned.get(sel.cleaned) : undefined;
  const selOccs = selCard?.occurrences
    ?? (sel ? [{ occ_id: sel.occId, start: sel.start, context: sel.wordText, window: sel.window, m9_coded: false }] : []);
  // Not a name = leave this word alone, so a spelling here is thrown away by both the render
  // and the scorer. Block the box and route him to the per-spot one, which records the true
  // WORDS at this spot as speech rather than as a name to fix.
  const selIsNota = selItem?.category === 'not-a-name';
  // The span he built at this word: the ones HE marked can be released; a phrase tail is the
  // sweep's own doing and is not his to undo here.
  const selTails = sel ? (spanFold.heads.get(sel.occId) ?? []).filter((t) => spanFold.marked.has(t)) : [];
  // Is the SELECTED word itself absorbed? (Reachable only right after marking it, before the
  // selection moves to its head — but the render must never offer to mark it twice.)
  const selIsTail = sel ? spanFold.marked.has(sel.occId) : false;
  const selHead = sel && !selIsTail ? headOf(sel.occId) : null;
  const selRuled = !!(selOccVal || selItem?.true_spelling || selItem?.category);

  return (
    <div className="np-page">
      <header className="np-topbar">
        <div className="np-title">
          <Link to={`/sessions/${sessionId}`} className="np-back">← Session</Link>
          <strong>Name pass</strong>
          <span className="np-sid">{sessionId}</span>
          <Link to={`/sessions/${sessionId}/validate`} className="np-back">Validator →</Link>
        </div>
        <div className="np-metrics">
          <div className="np-metric">
            <span className="np-metric-n">{confirmed}/{totalCandidates}</span> set
            {correctedCount > 0 && <span className="np-ear"> · {correctedCount} you corrected</span>}
            {needsBucket > 0 && (
              <span className="np-nobucket-count" title="Ruled, but with no category — the scorer drops a name with no bucket, so these currently score as nothing.">
                {' '}· {needsBucket} need a bucket
              </span>
            )}
          </div>
          <div className="np-cov">
            <div className="np-cov-bar"><div className="np-cov-fill" style={{ width: `${coveragePct}%` }} /></div>
            <span className="np-cov-label">{coveragePct}% heard</span>
          </div>
          {payload.seed.heard_confirmed && (
            <span className="np-heardstamp" title={`ear-gate stamped ${payload.seed.heard_confirmed}`}>
              heard ✓ {new Date(payload.seed.heard_confirmed).toLocaleDateString()}
            </span>
          )}
          <button className="np-heardbtn" onClick={markHeard}
                  title="Accept every un-corrected suggestion as heard, and stamp the key (re-stamps if already heard)">
            Mark heard{remainingSuggested.length > 0 ? ` — accept ${remainingSuggested.length}` : ''}
          </button>
          <span className="np-save">{saveStatus}</span>
        </div>
      </header>

      <div className="np-player">
        <button className="np-play" onClick={() => playerRef.current?.playPause()}>{playing ? '❚❚' : '▶'}</button>
        <div className="np-rates">
          {[0.5, 1, 1.5, 2].map((r) => (
            <button key={r} className={`np-rate${rate === r ? ' np-rate-on' : ''}`} onClick={() => setRate(r)}>{r}×</button>
          ))}
        </div>
        {session.has_audio ? (
          <WaveformPlayer ref={playerRef} audioUrl={audioURL(sessionId)} playbackRate={rate}
            onTimeUpdate={handleTimeUpdate} onReady={setDuration} onPlayPause={setPlaying} />
        ) : (
          <div className="np-noaudio">No audio for this session — play disabled.</div>
        )}
      </div>

      <div className="np-body">
        <main className="np-transcript" ref={transcriptRef} onClick={onTranscriptClick}>
          {transcriptEls}
        </main>

        <aside className="np-inspector">
          {sel ? (
            <div className="np-card">
              <div className="np-card-head">
                <span className="np-heard">{sel.wordText.trim()}</span>
                <button className="np-copy" onClick={copyMarker} title="Copy a reference to this exact word for chat">
                  ⧉ {sel.occId}
                </button>
              </div>
              {selSug && (
                <div className={`np-sugbox${selSug.confidence === 'low' ? ' np-sugbox-low' : ''}`}>
                  <span className="np-sug-label">Opus suggests</span> <b>{selSug.suggested}</b>
                  <span className="np-sug-conf"> ({selSug.confidence})</span>
                  {selSug.note && <div className="np-sug-note">{selSug.note}</div>}
                </div>
              )}
              {selTails.length > 0 && (
                <div className="np-span">
                  <div className="np-span-label">
                    One name across {selTails.length + 1} words{selRuled ? '' : ' — but this word rules no name yet'}
                  </div>
                  {selTails.map((t) => (
                    <div className="np-span-tail" key={t}>
                      <span className="np-span-word">{wordAt(t)}</span>
                      <button className="np-release" onClick={() => releaseTail(t)}
                        title="Give this word back its own decision">⊖ release</button>
                    </div>
                  ))}
                </div>
              )}
              <label className="np-field">
                <span>Just this one <em>({selIsNota ? 'the true WORDS here — recorded as speech' : 'this occurrence only'})</em></span>
                <input className="np-truth" value={selOccVal} placeholder="override for this spot only"
                  onChange={(e) => ruleOccurrence(sel.occId, sel.cleaned, e.target.value)} />
              </label>
              <label className="np-field">
                <span>All “{sel.cleaned}” <em>(every occurrence)</em></span>
                <input className="np-truth" value={selItem?.true_spelling ?? ''} disabled={selIsNota}
                  placeholder={selIsNota ? 'not a name — nothing to spell'
                    : selSug ? `suggested: ${selSug.suggested}` : 'true spelling'}
                  onChange={(e) => ruleSpelling(sel.cleaned, e.target.value)} />
              </label>
              {selIsNota && (
                <div className="np-route">
                  <b>Not a name</b> means <b>leave this word alone</b>, so a spelling here would be
                  thrown away. If you heard the true <i>words</i> at this spot — a blessing, a phrase
                  Whisper mangled — type them in <b>Just this one</b> above. They’re recorded as what
                  was <i>said</i>, not as a name the fixer has to produce.
                </div>
              )}
              <div className="np-actions">
                {selSug && !selItem?.true_spelling && !selIsNota && (
                  <button className="np-accept" onClick={() => ruleSpelling(sel.cleaned, selSug.suggested)}>✓ Accept “{selSug.suggested}”</button>
                )}
                {selHead && (
                  <button className="np-tail" onClick={() => markTail(sel.occId)}
                    title={`Whisper split one name across several words — join this one to “${wordAt(selHead)}”`}>
                    ⤺ Part of the name before
                  </button>
                )}
                <button className="np-notaname" onClick={() => setCategory(sel.cleaned, 'not-a-name', sel.occId)}>Not a name</button>
                <button className="np-clear" onClick={() => clearWord(sel.cleaned, sel.occId)}>Clear</button>
              </div>
              <div className="np-catrow">
                <select className="np-cat" value={selItem?.category ?? ''} onChange={(e) => setCategory(sel.cleaned, e.target.value, sel.occId)}>
                  {categories.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
                {selItem?.provenance && <span className="np-prov">{selItem.provenance}</span>}
              </div>
              <div className="np-occs">
                {selOccs.map((o) => (
                  <div className={`np-occ${o.occ_id === sel.occId ? ' np-occ-sel' : ''}`} key={o.occ_id}
                    onClick={() => selectOcc(o.occ_id, sel.cleaned, sel.wordText, o.start ?? null, o.window ?? null)}>
                    <button className="np-occ-play" onClick={(e) => { e.stopPropagation(); hear(o.window ?? null, o.start ?? null); }} title="Hear this">▶</button>
                    <span className="np-occ-ts">{fmtT(o.start ?? null)}</span>
                    <span className="np-occ-ctx">{o.context}</span>
                    {o.m9_coded && <span className="np-occ-badge">M9</span>}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="np-empty">
              The transcript shows Opus’s best guesses in place. Play through — <b>click any word</b> to hear it
              from there and work on it here (clicking plays, never writes). Fix all of a spelling, or just one occurrence; then
              <b> Mark heard</b> once you’ve listened.
            </div>
          )}

          <div className="np-legend">
            <span className="np-lg np-suggested">suggested</span>
            <span className="np-lg np-lowconf">low-confidence</span>
            <span className="np-lg np-confirmed">heard ✓</span>
            <span className="np-lg np-corrected">you fixed</span>
            <span className="np-lg np-nobucket" title="Ruled, but it has no category — the scorer drops a name with no bucket. Pick one and it counts.">needs a bucket</span>
            <span className="np-lg np-joined" title="Several words joined into one name — click it to see what it absorbed, or to release one.">one name, joined</span>
            <span className="np-lg np-nota">not a name</span>
          </div>
        </aside>
      </div>
    </div>
  );
}
