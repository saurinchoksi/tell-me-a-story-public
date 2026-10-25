import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useRef } from 'react';
import {
  getNameCorrections,
  blessNameCorrection,
  rejectNameCorrection,
  unrejectName,
  unpickSpelling,
  unruleName,
  audioURL,
} from '../api/client';
import type {
  CoveringVerdict,
  FamilyFixerStatus,
  NameCorrectionOccurrence,
  NameCorrectionSessionEntry,
  NameCorrectionsWorld,
  RuledVerdict,
} from '../types';
import { formatTime } from '../utils/time';
import type { NameCorrectionsRollup, NameCorrectionItem } from '../types';
import './NameReview.css';

/** Plain-language headers for the labeled sections — why the machine believed each group.
 *  Order matters: strongest machine belief first. */
const LABEL_COPY: Record<string, { title: string; sub: string }> = {
  'sounds-like': {
    title: 'Would fix automatically — sounds like the fix',
    sub: 'Not an English word, and it sounds like the character it re-heard. The machine’s strongest class.',
  },
  'audio-only': {
    title: 'Would fix automatically — audio only (shakiest)',
    sub: 'Not an English word, the re-listen heard a cast name, but the two don’t sound alike. This class holds both real recoveries and invented-name mistakes — your verdicts here are the calibration data.',
  },
  'ordinary-word': {
    title: 'Ordinary words — always your call',
    sub: 'The transcript word is a real English word, so the machine never trusts itself here.',
  },
  'inconsistent-spelling': {
    title: 'Spelled more than one way — pick one',
    sub: 'An invented character (no storybook canon) spelled several ways in one night. The machine proposes the spelling used most often, so the character can be followed across the transcript. Yes = use this spelling everywhere (edit it first if you prefer another). No = don’t unify — leave every spot as heard.',
  },
};

/** The occurrence's sentence with the judged word highlighted — several names can fly
 *  by in one span, and this is what tells the reviewer which word the verdict is about. */
function OccurrenceSentence({ o }: { o: NameCorrectionOccurrence }) {
  if (!o.segment_text || o.word_offset == null || o.word_len == null) return null;
  const before = o.segment_text.slice(0, o.word_offset);
  const word = o.segment_text.slice(o.word_offset, o.word_offset + o.word_len);
  const after = o.segment_text.slice(o.word_offset + o.word_len);
  return (
    <div className="name-occ-sentence">
      “{before}
      <mark className="name-occ-target">{word}</mark>
      {after}”
    </div>
  );
}

/** The warning chips — the suspicion signals we're testing during calibration. */
function FlagChips({ item }: { item: NameCorrectionItem }) {
  const nOcc = item.sessions.reduce((n, s) => n + s.occurrences.length, 0);
  const stable = (item.n_sibling_spellings ?? 0) === 0 && nOcc > 1;
  // the Marley signal: the suspect is said MORE often than the suggested name —
  // garbles are usually the minority form, so this smells like its own name
  const outnumbers =
    item.n_heard_in_transcript != null &&
    item.n_suggested_in_transcript != null &&
    item.n_heard_in_transcript > item.n_suggested_in_transcript;
  if (!item.echo_in_sentence && !stable && !outnumbers) return null;
  return (
    <span className="name-flags">
      {outnumbers && (
        <span
          className="name-flag"
          title="Garbles are usually the minority form. A word said far more often than the name it supposedly garbles may just be its own name. Logged for calibration — the machine never acts on it."
        >
          ⚠ said {item.n_heard_in_transcript}×, the suggestion {item.n_suggested_in_transcript}×
        </span>
      )}
      {item.echo_in_sentence && (
        <span
          className="name-flag"
          title="The suggested name already appears, correctly spelled, in the same sentence — the re-listen may just be echoing it."
        >
          ⚠ suggestion already in the sentence
        </span>
      )}
      {stable && (
        <span
          className="name-flag"
          title="Every occurrence is spelled exactly the same. Garbles of a real name tend to wobble; an invented name holds steady."
        >
          ⚠ spelled the same every time
        </span>
      )}
    </span>
  );
}

/** Plain words for the skip/error tokens in queue metadata — mirrors the Python-side
 *  SKIP_REASON_COPY in src/namefix.py (same tokens, this surface's words); an unknown
 *  token falls through raw rather than being hidden. */
const SKIP_COPY: Record<string, string> = {
  no_roster: 'no roster yet',
  no_audio_path: 'no audio file',
  no_stories: 'no stories found on the transcript',
  no_cache_dir: 'no session directory',
  deferred: 'deferred — runs later',
  error: 'the check failed partway',
};

/** The "did the family side run?" strip. Before this, a fresh instance with
 *  no roster looked pixel-identical to "family names checked, all clean" — the same
 *  hole class as the friend-name discovery gap. One line says checked / not run (and
 *  why) / stale / never stamped, plus how many family questions are open below. */
function FamilyStrip({ status }: { status: FamilyFixerStatus }) {
  const { checked, skipped, stale, unknown, questions } = status;
  if (checked === 0 && skipped === 0 && stale === 0 && unknown === 0) return null;
  const parts: string[] = [];
  if (checked > 0) {
    parts.push(`checked on ${checked} recording${checked === 1 ? '' : 's'}`);
    if (questions === 0) parts.push('no open questions');
  }
  if (questions > 0) {
    parts.push(`${questions} open question${questions === 1 ? '' : 's'} below`);
  }
  if (skipped > 0) {
    const reasons = status.skip_reasons.map((r) => SKIP_COPY[r] ?? r).join(', ');
    parts.push(
      `NOT checked on ${skipped} recording${skipped === 1 ? '' : 's'} — ${reasons}`,
    );
  }
  if (stale > 0) {
    parts.push(
      `${stale} recording${stale === 1 ? '' : 's'} changed since the last check — re-run the fixers`,
    );
  }
  if (unknown > 0) {
    parts.push(
      `${unknown} recording${unknown === 1 ? '' : 's'} never stamped — re-run the fixers (fix_session.py --write) to know`,
    );
  }
  const attention = skipped > 0 || stale > 0 || unknown > 0;
  return (
    <p
      className={`name-family-strip${attention ? ' name-family-strip-warn' : ''}`}
      title="Family-name checking is separate from story-universe names: it needs the roster. This line always says whether it actually ran."
    >
      <strong>Family names:</strong> {parts.join(' · ')}
      {status.skip_reasons.includes('no_roster') && (
        <> — register your family’s names on the Roster page, then re-run the fixers.</>
      )}
    </p>
  );
}

/**
 * The name-review queue — the human half of the SHADOW namefix stage (calibration).
 *
 * Nothing is applied to any transcript: the machine's opinions arrive as labeled
 * questions, your yes/no is RECORDED (Name Book + audit — the calibration dataset),
 * and everything applies later in one deliberate apply pass when classes graduate.
 */
export default function NameReview() {
  const [rollup, setRollup] = useState<NameCorrectionsRollup | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null); // heard_cleaned being acted on
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  // occurrence keys the user un-checked (same-sound-two-referents: rule on some spots only)
  const [excluded, setExcluded] = useState<Record<string, boolean>>({});

  const occKey = (sid: string, o: NameCorrectionOccurrence) =>
    `${sid}|${o.segment_id}|${o.word_index}`;

  // Shared clip player (the SessionDetections pattern): one <audio> element; the source
  // swaps to whichever session the clicked occurrence belongs to, seeks just before the
  // word, and auto-stops a few seconds later.
  const PREROLL = 1.5;
  const WINDOW = 2.5;
  const audioRef = useRef<HTMLAudioElement>(null);
  const srcSessionRef = useRef<string | null>(null);
  const stopAtRef = useRef<number | null>(null);
  const [playingKey, setPlayingKey] = useState<string | null>(null);

  function playClip(sessionId: string, key: string, start: number) {
    const audio = audioRef.current;
    if (!audio) return;
    if (playingKey === key) {
      audio.pause();
      return;
    }
    if (srcSessionRef.current !== sessionId) {
      audio.src = audioURL(sessionId);
      srcSessionRef.current = sessionId;
    }
    stopAtRef.current = start + WINDOW;
    audio.currentTime = Math.max(0, start - PREROLL);
    audio.play().then(() => setPlayingKey(key)).catch(() => setPlayingKey(null));
  }

  function handleTimeUpdate() {
    const audio = audioRef.current;
    if (!audio || stopAtRef.current == null) return;
    if (audio.currentTime >= stopAtRef.current) audio.pause();
  }

  function handleStopped() {
    stopAtRef.current = null;
    setPlayingKey(null);
  }

  function refresh() {
    getNameCorrections()
      .then(setRollup)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }
  useEffect(refresh, []);

  // identity for card keys / overrides / busy: world + heard + suggestion — matches how
  // the server rolls names up (one heard word can carry two suggestions: distinct cards)
  const itemKey = (world: string, item: NameCorrectionItem) =>
    `${world}|${item.heard_cleaned}→${item.canonical}`;

  async function actOnAll(
    key: string,
    item: NameCorrectionItem,
    act: (
      sessionId: string,
      gid: string,
      canonical?: string,
      occurrences?: { segment_id: number | string; word_index: number }[],
    ) => Promise<unknown>,
    canonical?: string,
  ) {
    setBusy(key);
    setError(null);
    try {
      // one human verdict covers every session holding this name; un-checked spots
      // stay queued (a partial yes — the API keeps the remainder as a question)
      for (const s of item.sessions) {
        if (s.stale) continue; // verdicts against a stale queue are refused server-side
        const kept = s.occurrences.filter((o) => !excluded[occKey(s.session_id, o)]);
        if (kept.length === 0) continue;
        const partial = kept.length < s.occurrences.length;
        await act(s.session_id, s.gid, canonical,
          partial ? kept.map((o) => ({ segment_id: o.segment_id, word_index: o.word_index })) : undefined);
      }
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  // Collapse-only cluster verdict: one Yes/No covers every member spelling of
  // an invented-name cluster. The backend already records ONE pick per cluster, so we
  // send the same verdict for each member's gid and the view-time fold hides the rest.
  // No per-spot exclusion here — a cluster decision is all-or-nothing (the split waits
  // for the apply pass).
  async function actOnCluster(
    key: string,
    members: NameCorrectionItem[],
    act: (
      sessionId: string,
      gid: string,
      canonical?: string,
      occurrences?: { segment_id: number | string; word_index: number }[],
    ) => Promise<unknown>,
    canonical?: string,
  ) {
    setBusy(key);
    setError(null);
    try {
      for (const item of members) {
        for (const s of item.sessions) {
          if (s.stale) continue; // stale queues are refused server-side
          await act(s.session_id, s.gid, canonical, undefined);
        }
      }
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function liftNo(key: string, world: string, item: NameCorrectionItem) {
    setBusy(`no:${key}`);
    setError(null);
    try {
      // folded_by says which standing ruling folded this card: a spelling pick lifts
      // the cluster pick; anything else lifts the word-level no
      if (item.folded_by === 'pick-leave' || item.folded_by === 'pick-unify') {
        await unpickSpelling(world, item.cluster_id!);
      } else {
        await unrejectName(world, item.heard_cleaned);
      }
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  async function undoVerdict(
    r: Pick<RuledVerdict, 'session_id' | 'verdict' | 'gid' | 'at'>,
  ) {
    setBusy(`undo:${r.session_id}|${r.gid}|${r.at}`);
    setError(null);
    try {
      await unruleName(r.session_id, r.verdict, r.gid, r.at);
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }

  function occurrenceRows(item: NameCorrectionItem, withCheckboxes: boolean) {
    return (
      <ul className="name-occurrences">
        {item.sessions.map((s) =>
          s.occurrences.map((o, i) => {
            const k = occKey(s.session_id, o);
            return (
              <li key={`${k}-${i}`}>
                <button
                  className="name-occ-play"
                  onClick={() => playClip(s.session_id, k, o.start)}
                  title="Play the audio around this word"
                  aria-label={playingKey === k ? 'Pause' : 'Play'}
                >
                  {playingKey === k ? '❚❚' : '▶'}
                </button>{' '}
                {withCheckboxes && (
                  <>
                    <label className="name-occ-check">
                      <input
                        type="checkbox"
                        checked={!excluded[k]}
                        onChange={(e) =>
                          setExcluded({ ...excluded, [k]: !e.target.checked })
                        }
                      />
                    </label>{' '}
                  </>
                )}
                <Link to={`/sessions/${s.session_id}/detections`}>
                  {s.session_id}
                </Link>{' '}
                · seg {o.segment_id} · {formatTime(o.start)} ·{' '}
                <span className="name-token">“{o.token}”</span>
                {o.corrected_by_dictionary && o.now_reads && (
                  <span
                    className="name-corrected-note"
                    title="The opt-in dictionary pass rewrote this word in place after the question was filed. The highlighted sentence shows the corrected form; the question is still about the word as first heard."
                  >
                    {' '}now reads{' '}
                    <span className="name-corrected-now">“{o.now_reads}”</span> — corrected
                    by dictionary
                  </span>
                )}
                <OccurrenceSentence o={o} />
              </li>
            );
          }),
        )}
      </ul>
    );
  }

  /** The covering-verdict rows on a spot-answered card: the answers that already
   *  settled every spot, each with the existing Undo (undoing frees the spots live). */
  function coveringRows(s: NameCorrectionSessionEntry) {
    return (s.covering ?? []).map((c: CoveringVerdict) => {
      const k = `undo:${s.session_id}|${c.gid}|${c.at}`;
      return (
        <li key={k} className="name-verdict-row">
          <span className={c.verdict === 'yes' ? 'name-verdict-yes' : 'name-verdict-no'}>
            {c.verdict === 'yes' ? '✓ yes' : '✕ no'}
          </span>
          {c.verdict === 'yes' && c.canonical && (
            <>
              <span className="name-arrow"> → </span>
              <span className="name-canonical-quiet">{c.canonical}</span>
            </>
          )}
          <span className="name-count">
            {' '}· {c.n_spots} spot{c.n_spots === 1 ? '' : 's'} · {s.session_id}
          </span>{' '}
          <button
            className="name-unreject"
            disabled={busy === k}
            onClick={() =>
              undoVerdict({ session_id: s.session_id, verdict: c.verdict, gid: c.gid, at: c.at })
            }
            title="Take this answer back — its spots return to the live queue"
          >
            ↩ Undo
          </button>
        </li>
      );
    });
  }

  // One per-word question card (the original render, now a helper so a collapsed
  // spelling cluster can reuse the occurrence list without duplicating markup).
  function renderNameCard(
    item: NameCorrectionItem,
    kind: 'live' | 'yes-fold' | 'no-fold' | 'spot-fold',
    world: string,
  ) {
    const key = itemKey(world, item);
    const nOcc = item.sessions.reduce((n, s) => n + s.occurrences.length, 0);
    const hasStale = item.sessions.some((s) => s.stale);
    return (
      <div
        key={key}
        className={`name-card${kind !== 'live' ? ' name-card-suppressed' : ''}`}
      >
        <div className="name-card-head">
          <span className="name-heard">{item.heard}</span>
          <span className="name-arrow">→</span>
          {kind === 'live' ? (
            <input
              className="name-canonical"
              value={overrides[key] ?? item.canonical}
              onChange={(e) => setOverrides({ ...overrides, [key]: e.target.value })}
              aria-label="Correct spelling"
            />
          ) : (
            <span
              className={
                kind === 'no-fold' ? 'name-canonical-fixed' : 'name-canonical-quiet'
              }
            >
              {item.canonical}
            </span>
          )}
          <FlagChips item={item} />
          {hasStale && (
            <span
              className="name-flag"
              title="The transcript changed since this queue was generated — re-run the fixers (fix_session.py --write) before ruling."
            >
              ⟳ stale — re-run the fixers
            </span>
          )}
          <span className="name-count">
            {nOcc} spot{nOcc === 1 ? '' : 's'} · {item.sessions.length} session
            {item.sessions.length === 1 ? '' : 's'}
          </span>
          {kind === 'live' &&
            (() => {
              const folded = item.sessions.reduce(
                (n, s) => n + (s.n_spot_answered_occ ?? 0),
                0,
              );
              return folded > 0 ? (
                <span
                  className="name-count"
                  title="These spots were already answered spot-by-spot — they sit in the 'Answered spot-by-spot' section below."
                >
                  · {folded} spot{folded === 1 ? '' : 's'} already answered — folded
                </span>
              ) : null;
            })()}
          <div className="name-actions">
            {kind === 'live' && (
              <>
                <button
                  className="name-bless"
                  disabled={busy === key || hasStale}
                  onClick={() =>
                    actOnAll(key, item, blessNameCorrection, overrides[key] ?? item.canonical)
                  }
                  title="Record a yes — nothing changes until the apply pass"
                >
                  ✓ Yes — record
                </button>
                <button
                  className="name-reject"
                  disabled={busy === key || hasStale}
                  onClick={() => actOnAll(key, item, rejectNameCorrection)}
                  title={
                    item.label === 'inconsistent-spelling'
                      ? 'Record a no — don’t unify these spellings; every spot stays as heard'
                      : 'Record a no — this word is never auto-rewritten or asked about again in this universe'
                  }
                >
                  ✕ No — record
                </button>
              </>
            )}
            {kind === 'no-fold' && (
              <button
                className="name-unreject"
                disabled={busy === `no:${key}`}
                onClick={() => liftNo(key, world, item)}
                title="Bring this question back to the live queue"
              >
                {item.folded_by === 'pick-leave'
                  ? '↩ Remove this don’t-unify'
                  : '↩ Remove this no'}
              </button>
            )}
          </div>
        </div>
        {occurrenceRows(item, kind === 'live')}
        {kind === 'spot-fold' && (
          <ul className="name-verdict-list">
            {item.sessions.map((s) => coveringRows(s))}
          </ul>
        )}
      </div>
    );
  }

  // One collapsed card for an invented-name spelling cluster. Its several
  // member spellings share ONE machine verdict, so they get ONE card with ONE Yes/No —
  // every member spelling listed so contamination (a different character or an ordinary
  // word swept in) is obvious at a glance, which is exactly when you say No.
  function renderClusterCard(
    cluster: { clusterId: string; members: NameCorrectionItem[] },
    kind: 'live' | 'yes-fold' | 'no-fold' | 'spot-fold',
    world: string,
  ) {
    const { clusterId, members } = cluster;
    const key = `${world}|cluster:${clusterId}`;
    const canonical = members[0].canonical;
    const nOcc = members.reduce(
      (n, m) => n + m.sessions.reduce((k, s) => k + s.occurrences.length, 0),
      0,
    );
    const nSessions = new Set(
      members.flatMap((m) => m.sessions.map((s) => s.session_id)),
    ).size;
    const hasStale = members.some((m) => m.sessions.some((s) => s.stale));
    return (
      <div
        key={key}
        className={`name-card name-cluster-card${kind !== 'live' ? ' name-card-suppressed' : ''}`}
      >
        <div className="name-card-head">
          <span
            className="name-cluster-tag"
            title="These spellings were grouped as one invented character, so the machine records a single verdict for the whole group. One Yes unifies them all to the spelling on the right; one No leaves every spot exactly as heard."
          >
            spelled several ways
          </span>
          <span className="name-arrow">→</span>
          {kind === 'live' ? (
            <input
              className="name-canonical"
              value={overrides[key] ?? canonical}
              onChange={(e) => setOverrides({ ...overrides, [key]: e.target.value })}
              aria-label="One spelling for the whole cluster"
            />
          ) : (
            <span
              className={
                kind === 'no-fold' ? 'name-canonical-fixed' : 'name-canonical-quiet'
              }
            >
              {canonical}
            </span>
          )}
          {hasStale && (
            <span
              className="name-flag"
              title="The transcript changed since this queue was generated — re-run the fixers (fix_session.py --write) before ruling."
            >
              ⟳ stale — re-run the fixers
            </span>
          )}
          <span className="name-count">
            {members.length} spelling{members.length === 1 ? '' : 's'} · {nOcc} spot
            {nOcc === 1 ? '' : 's'} · {nSessions} session{nSessions === 1 ? '' : 's'}
          </span>
          <div className="name-actions">
            {kind === 'live' && (
              <>
                <button
                  className="name-bless"
                  disabled={busy === key || hasStale}
                  onClick={() =>
                    actOnCluster(key, members, blessNameCorrection, overrides[key] ?? canonical)
                  }
                  title="Record a yes — unify every spelling below to the one on the right; nothing changes until the apply pass"
                >
                  ✓ Yes — unify all
                </button>
                <button
                  className="name-reject"
                  disabled={busy === key || hasStale}
                  onClick={() => actOnCluster(key, members, rejectNameCorrection)}
                  title="Record a no — don’t unify these; every spot stays exactly as heard"
                >
                  ✕ No — leave as heard
                </button>
              </>
            )}
            {kind === 'no-fold' && (
              <button
                className="name-unreject"
                disabled={busy === `no:${key}`}
                onClick={() => liftNo(key, world, members[0])}
                title="Bring this cluster’s questions back to the live queue"
              >
                ↩ Remove this don’t-unify
              </button>
            )}
          </div>
        </div>
        <ul className="name-cluster-members">
          {members.map((m) => (
            <li key={itemKey(world, m)} className="name-cluster-member">
              <span className="name-heard">{m.heard}</span>
              <FlagChips item={m} />
              {occurrenceRows(m, false)}
            </li>
          ))}
        </ul>
      </div>
    );
  }

  function worldSections(
    worlds: NameCorrectionsWorld[],
    kind: 'live' | 'yes-fold' | 'no-fold' | 'spot-fold',
  ) {
    // Collapse invented-name spelling questions that share a cluster into ONE card
    // — the machine records one verdict per cluster, so N per-spelling cards
    // were a display lie. LIVE only: once folded, a cluster's members can split across
    // the yes/no fold sections with mixed reasons (a word-level yes/no pre-empts the
    // cluster pick — see _partition_rulings), and each folded member needs its own lift.
    const collapse = kind === 'live';
    return worlds.map((w) => {
      const clusters = new Map<string, NameCorrectionItem[]>();
      const singles: NameCorrectionItem[] = [];
      for (const item of w.names) {
        if (collapse && item.label === 'inconsistent-spelling' && item.cluster_id) {
          const arr = clusters.get(item.cluster_id);
          if (arr) arr.push(item);
          else clusters.set(item.cluster_id, [item]);
        } else {
          singles.push(item);
        }
      }
      const groups = Array.from(clusters.entries());
      const clusterCards = groups
        .filter(([, members]) => members.length > 1)
        .map(([clusterId, members]) =>
          renderClusterCard({ clusterId, members }, kind, w.world),
        );
      // a lone spelling is just a normal card — collapse only real multi-spelling groups
      const loneSpellings = groups
        .filter(([, members]) => members.length === 1)
        .map(([, members]) => members[0]);
      const allSingles = [...singles, ...loneSpellings];
      return (
        <section key={w.world} className="name-review-world">
          <h3>{w.world}</h3>
          {allSingles.map((item) => renderNameCard(item, kind, w.world))}
          {clusterCards}
        </section>
      );
    });
  }

  if (loading) return <p>Loading name questions...</p>;
  if (error) return <p className="name-review-error">{error}</p>;
  const empty =
    !rollup ||
    (rollup.n_pending_groups === 0 &&
      rollup.n_suppressed_groups === 0 &&
      rollup.n_covered_by_yes === 0 &&
      (rollup.n_spot_answered ?? 0) === 0 &&
      rollup.n_ruled === 0);
  if (empty) {
    return (
      <div className="name-review">
        <h1>Name review</h1>
        <p className="name-review-empty">
          Nothing waiting — no recording has open name questions.
        </p>
        {rollup && <FamilyStrip status={rollup.family_fixer} />}
      </div>
    );
  }

  return (
    <div className="name-review">
      <audio
        ref={audioRef}
        preload="none"
        onTimeUpdate={handleTimeUpdate}
        onPause={handleStopped}
        onEnded={handleStopped}
      />
      <h1>Name review</h1>
      <p className="name-review-sub">
        Calibration: nothing has been changed in any transcript. These are the machine&apos;s
        opinions, grouped by why it believed them. Your yes/no is recorded — in the Name
        Book and as calibration data — and everything applies later, together, once a group
        has earned trust. <strong>Mixed card?</strong> Un-check the spots your verdict
        shouldn&apos;t cover — they stay queued for a different answer. A forever-rule is
        only written when a word&apos;s whole story agrees; ambiguous words (like a name in
        one spot and a real word in another) stay per-spot decisions.
      </p>
      <FamilyStrip status={rollup.family_fixer} />
      {rollup.labels.map((section) => {
        const copy = LABEL_COPY[section.label] ?? {
          title: section.label,
          sub: '',
        };
        const n = section.worlds.reduce((acc, w) => acc + w.names.length, 0);
        return (
          <section key={section.label} className="name-label-section">
            <h2>
              {copy.title} <span className="name-label-count">({n})</span>
            </h2>
            {copy.sub && <p className="name-review-sub">{copy.sub}</p>}
            {worldSections(section.worlds, 'live')}
          </section>
        );
      })}
      {rollup.n_covered_by_yes > 0 && (
        <details className="name-suppressed">
          <summary>
            Covered by your earlier yes ({rollup.n_covered_by_yes}) — will apply at the
            apply pass
          </summary>
          {worldSections(rollup.covered_by_yes, 'yes-fold')}
        </details>
      )}
      {rollup.n_suppressed_groups > 0 && (
        <details className="name-suppressed">
          <summary>
            Covered by your earlier no ({rollup.n_suppressed_groups}) — quiet, but nothing
            is hidden
          </summary>
          <p className="name-review-sub">
            You already said no to these words, so they won&apos;t ask again. If a night
            comes where she really does say the name this way, listen here and remove the
            no to bring the question back.
          </p>
          {worldSections(rollup.suppressed, 'no-fold')}
        </details>
      )}
      {(rollup.n_spot_answered ?? 0) > 0 && (
        <details className="name-suppressed">
          <summary>
            Answered spot-by-spot ({rollup.n_spot_answered}) — no standing rule
          </summary>
          <p className="name-review-sub">
            You answered these where a single rule would be wrong (a word that means two
            things in one night) or where no rule book exists yet (a story universe the
            system doesn&apos;t recognize). A re-run brought the same question back, and
            your recorded answers cover every spot — so nothing is asked again. Undo an
            answer to bring its spots back to the live queue.
          </p>
          {worldSections(rollup.spot_answered, 'spot-fold')}
        </details>
      )}
      {rollup.n_ruled > 0 && (
        <details className="name-suppressed">
          <summary>Your verdicts ({rollup.n_ruled}) — everything you&apos;ve ruled on</summary>
          <p className="name-review-sub">
            Newest first. Undo pops the verdict, reverses its Name Book entry, and puts
            the question back in the queue.
          </p>
          <ul className="name-verdict-list">
            {rollup.ruled.map((r) => {
              const k = `undo:${r.session_id}|${r.gid}|${r.at}`;
              return (
                <li key={k} className="name-verdict-row">
                  <span className={r.verdict === 'yes' ? 'name-verdict-yes' : 'name-verdict-no'}>
                    {r.verdict === 'yes' ? '✓ yes' : '✕ no'}
                  </span>{' '}
                  <span className="name-heard">{r.heard}</span>
                  <span className="name-arrow"> → </span>
                  <span className="name-canonical-quiet">{r.canonical}</span>
                  {r.partial && <span className="name-count"> (some spots only)</span>}
                  <span className="name-count">
                    {' '}· {r.world} · {r.session_id} ·{' '}
                    {new Date(r.at).toLocaleString()}
                  </span>{' '}
                  {r.undoable && (
                    <button
                      className="name-unreject"
                      disabled={busy === k}
                      onClick={() => undoVerdict(r)}
                      title="Take this verdict back — the question returns to the queue"
                    >
                      ↩ Undo
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        </details>
      )}
    </div>
  );
}
