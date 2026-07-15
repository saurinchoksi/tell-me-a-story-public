import type { Decision } from '../types';
import './ConfirmBar.css';

export type SaveState = 'idle' | 'saving' | 'success' | 'error';

interface ConfirmBarProps {
  decisions: Record<string, Decision>;
  saveState: SaveState;
  errorMessage: string | null;
  onSave: () => void;
}

export default function ConfirmBar({
  decisions,
  saveState,
  errorMessage,
  onSave,
}: ConfirmBarProps) {
  const values = Object.values(decisions);
  const confirmed = values.filter(
    (d) => d.action === 'confirm' || d.action === 'reassign' || d.action === 'confirm_variant',
  ).length;
  const created = values.filter((d) => d.action === 'create').length;
  const skipped = values.filter((d) => d.action === 'skip').length;

  // Disable save when: no non-skip decisions, or any "create" has empty fields
  const hasWork = confirmed + created > 0;
  const createDecisions = values.filter((d) => d.action === 'create');
  const createValid = createDecisions.every(
    (d) => (d.new_name ?? '').trim() && (d.new_role ?? '').trim(),
  );
  const canSave = hasWork && createValid && saveState !== 'saving';

  const parts: string[] = [];
  if (confirmed > 0) parts.push(`${confirmed} confirmed`);
  if (created > 0) parts.push(`${created} new`);
  if (skipped > 0) parts.push(`${skipped} skipped`);
  const summary = parts.join(', ') || 'No decisions yet';

  const btnClass =
    saveState === 'success'
      ? 'confirm-bar-save confirm-bar-save--success'
      : 'confirm-bar-save';

  const btnLabel =
    saveState === 'saving'
      ? 'Saving...'
      : saveState === 'success'
        ? 'Saved!'
        : 'Confirm & Save';

  return (
    <div className="confirm-bar">
      <span className="confirm-bar-summary">{summary}</span>
      <div className="confirm-bar-actions">
        {saveState === 'error' && errorMessage && (
          <span className="confirm-bar-error">{errorMessage}</span>
        )}
        <button
          className={btnClass}
          onClick={onSave}
          disabled={!canSave}
        >
          {btnLabel}
        </button>
      </div>
    </div>
  );
}
