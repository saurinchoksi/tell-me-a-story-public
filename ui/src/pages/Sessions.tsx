import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { listSessions } from '../api/client';
import { formatSessionDate, formatTime } from '../utils/time';
import { storyTitleLine } from '../utils/stories';
import type { SessionSummary, ValidationStatus as VStatus } from '../types';
import SessionNote from '../components/SessionNote';
import ValidationStatus from '../components/ValidationStatus';
import './Sessions.css';

const PIPELINE_STAGES = [
  { key: 'has_audio' as const, label: 'Recorded' },
  { key: 'has_transcript' as const, label: 'Transcribed' },
  { key: 'has_diarization' as const, label: 'Diarized' },
  { key: 'has_embeddings' as const, label: 'Embeddings' },
  { key: 'has_identifications' as const, label: 'Identified' },
];

/** Human-readable labels for transcript-rich.json _processing stage keys. */
const STAGE_LABELS: Record<string, string> = {
  transcription: 'Transcription',
  diarization: 'Diarization',
  diarization_enrichment: 'Diarization enrichment',
  gap_detection: 'Gap detection',
  llm_normalization: 'LLM normalization',
  dictionary_normalization: 'Dictionary normalization',
  embedding_extraction: 'Embedding extraction',
};

function stageLabel(key: string): string {
  return STAGE_LABELS[key] ?? key.replace(/_/g, ' ');
}

/** Sortable column keys. Time isn't listed — it sorts with Date via session id. */
type SortKey = 'date' | 'length' | 'validation' | 'notes';
type SortDir = 'asc' | 'desc';

/** Validation progress order — not_started (0) sorts first ascending. */
const VALIDATION_ORDER: Record<VStatus, number> = {
  not_started: 0,
  in_progress: 1,
  done: 2,
};

/**
 * Returns a comparator value for two sessions on the given key.
 * Nulls in `duration_seconds` are treated as smallest so they cluster together
 * (top in asc, bottom in desc) rather than scattering.
 */
function compareSessions(a: SessionSummary, b: SessionSummary, key: SortKey): number {
  switch (key) {
    case 'date':
      return a.id.localeCompare(b.id);
    case 'length': {
      const av = a.duration_seconds ?? -1;
      const bv = b.duration_seconds ?? -1;
      return av - bv;
    }
    case 'validation':
      return VALIDATION_ORDER[a.validation_status] - VALIDATION_ORDER[b.validation_status];
    case 'notes':
      return a.note_count - b.note_count;
  }
}

export default function Sessions() {
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>('date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  useEffect(() => {
    listSessions()
      .then(setSessions)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const sortedSessions = useMemo(() => {
    const copy = [...sessions];
    copy.sort((a, b) => {
      const cmp = compareSessions(a, b, sortKey);
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return copy;
  }, [sessions, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('asc');
    }
  }

  if (loading) return <p>Loading sessions...</p>;
  if (error) return <p className="error">Error: {error}</p>;

  if (sessions.length === 0) {
    return (
      <div className="sessions-page">
        <div className="sessions-empty">
          <h2>No stories yet</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="sessions-page">
      <div className="sessions-header">
        <h1>Story Sessions</h1>
        <p className="sessions-subtitle">
          {sessions.length} session{sessions.length !== 1 ? 's' : ''} recorded
        </p>
      </div>

      <div className="session-list-header">
        <SortHeader label="Date" sortKey="date" active={sortKey} dir={sortDir} onSort={handleSort} />
        <span>Time</span>
        <SortHeader label="Length" sortKey="length" active={sortKey} dir={sortDir} onSort={handleSort} />
        <span>Pipeline</span>
        <SortHeader label="Validation" sortKey="validation" active={sortKey} dir={sortDir} onSort={handleSort} />
        <SortHeader label="Notes" sortKey="notes" active={sortKey} dir={sortDir} onSort={handleSort} />
      </div>

      <div className="sessions-list">
        {sortedSessions.map((s) => {
          const { date, time } = formatSessionDate(s.id);
          const failedLabel = s.failed_stages.map(stageLabel).join(', ');

          return (
            <div key={s.id} className="session-row">
              {s.has_transcript ? (
                <Link
                  to={`/sessions/${s.id}`}
                  className="session-row-day session-row-day--link"
                  title="Open reading view"
                >
                  {date}
                </Link>
              ) : (
                <span className="session-row-day">{date}</span>
              )}
              <span className="session-row-time">{time}</span>
              <span className="session-row-length">
                {s.duration_seconds != null ? formatTime(s.duration_seconds) : '—'}
              </span>

              <div className="session-row-pipeline">
                {PIPELINE_STAGES.map((stage) => (
                  <div
                    key={stage.key}
                    className={`pipeline-dot ${s[stage.key] ? 'pipeline-dot--done' : 'pipeline-dot--pending'}`}
                    data-label={stage.label}
                    role="img"
                    aria-label={stage.label}
                  />
                ))}
                {s.failed_stages.length > 0 && (
                  <span
                    className="pipeline-warning"
                    role="img"
                    data-label={`${failedLabel} failed`}
                    aria-label={`Pipeline error: ${failedLabel} failed`}
                  >
                    ⚠
                  </span>
                )}
              </div>

              <ValidationStatus
                sessionId={s.id}
                initialStatus={s.validation_status}
              />

              <span
                className="session-row-notes"
                title={`${s.note_count} validation note${s.note_count === 1 ? '' : 's'}`}
              >
                {s.note_count}
              </span>

              <Link
                to={`/sessions/${s.id}/speakers`}
                className="session-row-speakers"
                title="Open speakers identification"
                aria-label="Open speakers identification"
              >
                👥
              </Link>

              {s.stories && (
                <div className="session-row-stories">
                  <span className="session-row-stories-icon" aria-hidden="true">📖</span>
                  {s.stories.worlds.map((w) => (
                    <span key={w} className="story-chip">{w}</span>
                  ))}
                  {storyTitleLine(s.stories) && (
                    <span className="session-row-stories-title">
                      {storyTitleLine(s.stories)}
                    </span>
                  )}
                </div>
              )}

              <SessionNote sessionId={s.id} initialNote={s.note} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

interface SortHeaderProps {
  label: string;
  sortKey: SortKey;
  active: SortKey;
  dir: SortDir;
  onSort: (key: SortKey) => void;
}

function SortHeader({ label, sortKey, active, dir, onSort }: SortHeaderProps) {
  const isActive = active === sortKey;
  const arrow = isActive ? (dir === 'asc' ? '▲' : '▼') : '';
  return (
    <button
      type="button"
      className={`session-sort-header${isActive ? ' session-sort-header--active' : ''}`}
      aria-sort={isActive ? (dir === 'asc' ? 'ascending' : 'descending') : 'none'}
      onClick={() => onSort(sortKey)}
    >
      {label}
      <span className="session-sort-arrow" aria-hidden="true">{arrow}</span>
    </button>
  );
}
