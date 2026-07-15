import { useState } from 'react';
import { setValidationStatus } from '../api/client';
import type { ValidationStatus as Status } from '../types';

interface ValidationStatusProps {
  sessionId: string;
  initialStatus: Status;
}

/** A click advances the status in this order. */
const NEXT: Record<Status, Status> = {
  not_started: 'in_progress',
  in_progress: 'done',
  done: 'not_started',
};

const LABELS: Record<Status, string> = {
  not_started: 'Not started',
  in_progress: 'In progress',
  done: 'Done',
};

/** Glyphs progress empty → partial → full, mirroring the status. */
const ICONS: Record<Status, string> = {
  not_started: '○',
  in_progress: '◐',
  done: '●',
};

/**
 * Validation-status indicator for a Sessions-list row.
 * A compact button that cycles not started → in progress → done on click,
 * with an optimistic update that reverts if the save fails.
 */
export default function ValidationStatus({
  sessionId,
  initialStatus,
}: ValidationStatusProps) {
  const [status, setStatus] = useState<Status>(initialStatus);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function cycle() {
    if (saving) return;
    const previous = status;
    const next = NEXT[status];
    setStatus(next); // optimistic
    setSaving(true);
    setError(null);
    try {
      const result = await setValidationStatus(sessionId, next);
      setStatus(result.validationStatus);
    } catch (e) {
      setStatus(previous); // revert
      setError(e instanceof Error ? e.message : 'Failed to update status');
    } finally {
      setSaving(false);
    }
  }

  const label = LABELS[status];

  return (
    <button
      className={`validation-status validation-status--${status}`}
      onClick={() => void cycle()}
      disabled={saving}
      title={error ?? `Validation: ${label} — click to change`}
      aria-label={`Validation: ${label}. Click to change.`}
    >
      <span className="validation-status-icon" aria-hidden="true">
        {ICONS[status]}
      </span>
    </button>
  );
}
