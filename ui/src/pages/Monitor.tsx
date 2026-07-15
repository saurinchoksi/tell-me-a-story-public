import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getDetectionsRollup, scanAllDetections, getNameCorrections } from '../api/client';
import { formatSessionDate, formatTime } from '../utils/time';
import type { DetectionRunSummary, DetectionsRollup } from '../types';
import './Monitor.css';

/** Per-session badge: distinguishes flags / scanned-clean / never-scanned. */
function FlagBadge({ summary }: { summary: DetectionRunSummary | undefined }) {
  if (!summary) {
    return (
      <span className="monitor-badge monitor-badge--unscanned" title="Not scanned">
        —
      </span>
    );
  }
  const scannedAt = new Date(summary.run_at).toLocaleString();
  if (summary.n_flags === 0) {
    return (
      <span className="monitor-badge monitor-badge--clean" title={`Clean — scanned ${scannedAt}`}>
        0
      </span>
    );
  }
  return (
    <span
      className="monitor-badge monitor-badge--flagged"
      title={`${summary.n_flags} flags — scanned ${scannedAt}`}
    >
      {summary.n_flags}
    </span>
  );
}

export default function Monitor() {
  const [rollup, setRollup] = useState<DetectionsRollup | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [scanning, setScanning] = useState(false);
  const [pendingNames, setPendingNames] = useState(0);

  useEffect(() => {
    getDetectionsRollup()
      .then(setRollup)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
    // the name-review queue count is additive info — a failure here never blocks the monitor
    getNameCorrections()
      .then((nc) => setPendingNames(nc.n_pending_groups))
      .catch(() => {});
  }, []);

  async function handleRescanAll() {
    setScanning(true);
    setError(null);
    try {
      setRollup(await scanAllDetections());
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setScanning(false);
    }
  }

  if (loading) return <p>Loading monitor...</p>;
  if (error) return <p className="error">Error: {error}</p>;
  if (!rollup) return null;

  const { detectors, sessions, totals } = rollup;
  const anyStale = sessions.some((s) => s.stale);

  // Date | Time | Length | one column per detector — shared by header and rows
  const gridStyle = {
    gridTemplateColumns: `152px 68px 52px repeat(${detectors.length}, minmax(80px, 1fr))`,
  };

  return (
    <div className="monitor-page">
      <div className="monitor-header">
        <div className="monitor-header-row">
          <h1>Monitor</h1>
          <button
            className="monitor-rescan"
            onClick={handleRescanAll}
            disabled={scanning}
            title="Re-scan sessions that are missing or stale (full pass, incl. the LLM judge)"
          >
            {scanning ? 'Scanning…' : anyStale ? 'Re-scan stale' : 'Re-scan all'}
          </button>
        </div>
        <p className="monitor-subtitle">
          Failure-mode detectors over {sessions.length} transcribed session
          {sessions.length !== 1 ? 's' : ''} — runs after each transcription; viewing only
          reads results, transcripts are never modified
        </p>
        {rollup.warning && <p className="monitor-warning">⚠ {rollup.warning}</p>}
        {pendingNames > 0 && (
          <p className="monitor-name-banner">
            <Link to="/name-review">
              {pendingNames} name{pendingNames === 1 ? '' : 's'} need your review →
            </Link>
          </p>
        )}
      </div>

      <div className="monitor-cards">
        {detectors.map((d) => {
          const scanned = sessions.filter((s) => s.results[d.id] !== undefined).length;
          const total = totals[d.id] ?? 0;
          return (
            <div key={d.id} className="monitor-card">
              <div className="monitor-card-top">
                <span className="monitor-card-mode">{d.failure_mode}</span>
                <span className="monitor-card-version">v{d.version}</span>
              </div>
              <h2 className="monitor-card-label">{d.label}</h2>
              <div className="monitor-card-stats">
                <span className="monitor-card-count">{total}</span>
                <span className="monitor-card-count-label">
                  flag{total !== 1 ? 's' : ''} · {scanned}/{sessions.length} sessions scanned
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {sessions.length === 0 ? (
        <div className="monitor-empty">
          <h2>No transcribed sessions yet</h2>
        </div>
      ) : (
        <>
          <div className="monitor-list-header" style={gridStyle}>
            <span>Date</span>
            <span>Time</span>
            <span>Length</span>
            {detectors.map((d) => (
              <span key={d.id} title={d.label}>
                {d.failure_mode}
              </span>
            ))}
          </div>
          <div className="monitor-list">
            {sessions.map((s) => {
              const { date, time } = formatSessionDate(s.session_id);
              return (
                <div key={s.session_id} className="monitor-row" style={gridStyle}>
                  <span className="monitor-row-day-cell">
                    <Link
                      to={`/sessions/${s.session_id}/detections`}
                      className="monitor-row-day"
                      title="Open detection detail"
                    >
                      {date}
                    </Link>
                    {s.stale && (
                      <span className="monitor-stale" title="Transcript changed since last scan — re-scan">
                        ⟳
                      </span>
                    )}
                  </span>
                  <span className="monitor-row-time">{time}</span>
                  <span className="monitor-row-length">
                    {s.duration_seconds != null ? formatTime(s.duration_seconds) : '—'}
                  </span>
                  {detectors.map((d) => (
                    <FlagBadge key={d.id} summary={s.results[d.id]} />
                  ))}
                  {s.stories && (
                    <span className="monitor-row-story" title={s.stories.label}>
                      📖 {s.stories.label}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
