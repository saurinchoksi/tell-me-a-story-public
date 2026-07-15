import { useEffect, useState } from 'react';
import {
  addHomeCharacter,
  addRosterAlias,
  addRosterNickname,
  addRosterPerson,
  getHomeCanon,
  getRoster,
} from '../api/client';
import type { HomeCanon, Roster, RosterPerson } from '../types';
import './Roster.css';

/**
 * The roster page — the one deliberate place "who exists" gets edited (TMAS-56).
 *
 * Add-only by design: people (family or friend), nicknames (legitimate name forms —
 * correct as heard, matchable, correction targets of their own), and aliases (known
 * distortions to catch exactly). Every write is validated server-side before saving;
 * removals stay rare hand edits because recorded verdicts reference these forms.
 */
export default function RosterPage() {
  const [roster, setRoster] = useState<Roster | null>(null);
  const [canon, setCanon] = useState<HomeCanon | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [canonical, setCanonical] = useState('');
  const [circle, setCircle] = useState<'family' | 'friend'>('family');
  const [character, setCharacter] = useState('');
  // per-person form inputs, keyed by person id
  const [forms, setForms] = useState<Record<string, { nickname?: string; alias?: string }>>({});

  function refresh() {
    getRoster()
      .then(setRoster)
      .catch((e) => setError(e.message));
    getHomeCanon()
      .then(setCanon)
      .catch((e) => setError(e.message));
  }
  useEffect(refresh, []);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function aliasesOf(p: RosterPerson): string[] {
    return (roster?.aliases ?? [])
      .filter((a) => a.person_id === p.id)
      .map((a) => a.token);
  }

  if (!roster) return <p>{error ?? 'Loading roster...'}</p>;

  return (
    <div className="roster-page">
      <h1>Roster</h1>
      <p className="roster-sub">
        The real people the system knows, and every legitimate spelling of their names.
        A <strong>nickname</strong> is a real name form: never flagged, never rewritten,
        and a garble of it is corrected toward the nickname as spoken. An{' '}
        <strong>alias</strong> is the opposite: a known mis-spelling to catch exactly.
        Adding here is deliberate and validated; nothing can be removed from this page.
      </p>
      {error && <p className="roster-error">{error}</p>}

      {roster.people.map((p) => (
        <div key={p.id} className="roster-person">
          <div className="roster-person-head">
            <span className="roster-canonical">{p.canonical}</span>
            <span className={`roster-circle roster-circle-${p.circle ?? 'family'}`}>
              {p.circle ?? 'family'}
            </span>
            {p.role && <span className="roster-role">{p.role}</span>}
          </div>
          <div className="roster-forms">
            <div className="roster-form-row">
              <span className="roster-form-label">Nicknames</span>
              {(p.nicknames ?? []).map((n) => (
                <span key={n} className="roster-chip">{n}</span>
              ))}
              <input
                placeholder="add a nickname"
                value={forms[p.id]?.nickname ?? ''}
                onChange={(e) =>
                  setForms({ ...forms, [p.id]: { ...forms[p.id], nickname: e.target.value } })
                }
              />
              <button
                disabled={busy || !(forms[p.id]?.nickname ?? '').trim()}
                onClick={() =>
                  act(async () => {
                    await addRosterNickname(p.id, forms[p.id]!.nickname!.trim());
                    setForms({ ...forms, [p.id]: { ...forms[p.id], nickname: '' } });
                  })
                }
                title="A legitimate name form — correct as heard, everywhere"
              >
                + Add
              </button>
            </div>
            <div className="roster-form-row">
              <span className="roster-form-label">Aliases</span>
              {aliasesOf(p).map((t) => (
                <span key={t} className="roster-chip roster-chip-alias">{t}</span>
              ))}
              <input
                placeholder="add a known mis-spelling"
                value={forms[p.id]?.alias ?? ''}
                onChange={(e) =>
                  setForms({ ...forms, [p.id]: { ...forms[p.id], alias: e.target.value } })
                }
              />
              <button
                disabled={busy || !(forms[p.id]?.alias ?? '').trim()}
                onClick={() =>
                  act(async () => {
                    await addRosterAlias(p.id, forms[p.id]!.alias!.trim());
                    setForms({ ...forms, [p.id]: { ...forms[p.id], alias: '' } });
                  })
                }
                title="A distortion to catch exactly (rejected if the sound-match already covers it)"
              >
                + Add
              </button>
            </div>
          </div>
        </div>
      ))}

      <div className="roster-add-person">
        <h2>Add a person</h2>
        <div className="roster-form-row">
          <input
            placeholder="name as it should be spelled"
            value={canonical}
            onChange={(e) => setCanonical(e.target.value)}
          />
          <select value={circle} onChange={(e) => setCircle(e.target.value as 'family' | 'friend')}>
            <option value="family">family</option>
            <option value="friend">friend</option>
          </select>
          <button
            disabled={busy || !canonical.trim()}
            onClick={() =>
              act(async () => {
                await addRosterPerson(canonical.trim(), circle);
                setCanonical('');
              })
            }
          >
            + Add person
          </button>
        </div>
      </div>

      <div className="roster-characters">
        <h2>Our characters</h2>
        <p className="roster-sub">
          The family&apos;s own recurring characters — Our cast. A registered
          character&apos;s spelling is never second-guessed by any fixer, and when its
          name comes out several ways in one night, the registered spelling is what
          gets proposed. Rulings about these live in the &quot;Home&quot; universe.
        </p>
        <div className="roster-form-row roster-character-chips">
          {(canon?.characters ?? []).map((c) => (
            <span key={c.id} className="roster-chip roster-chip-character">
              {c.canonical}
            </span>
          ))}
          <input
            placeholder="character name as it should be spelled"
            value={character}
            onChange={(e) => setCharacter(e.target.value)}
          />
          <button
            disabled={busy || !character.trim()}
            onClick={() =>
              act(async () => {
                await addHomeCharacter(character.trim());
                setCharacter('');
              })
            }
            title="Register one of the family's own characters — refused if it collides with a person's name"
          >
            + Register
          </button>
        </div>
      </div>

      {/* What registration actually triggers (TMAS-72) — before this, the fixer
          re-run lived in an undocumented CLI and nothing on this page said so. */}
      <div className="roster-hint">
        <h2>What happens after you register</h2>
        <p className="roster-sub">
          New recordings are checked against this roster automatically as they come in.
          Recordings processed <em>before</em> a registration are not re-checked by
          themselves — re-run the fixers on them:{' '}
          <code>python src/fix_session.py &lt;session-id&gt; --write</code>. Either way,
          the results land on the <strong>Name review</strong> page, which always says
          whether the family check ran — even when it found nothing.
        </p>
      </div>
    </div>
  );
}
