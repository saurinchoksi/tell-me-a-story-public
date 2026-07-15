import { useState } from 'react';
import { saveSessionNote } from '../api/client';

interface SessionNoteProps {
  sessionId: string;
  initialNote: string;
}

/**
 * Session-level free-text note shown on a Sessions-list row.
 * Read-only by default; click to edit in place.
 */
export default function SessionNote({ sessionId, initialNote }: SessionNoteProps) {
  const [note, setNote] = useState(initialNote);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(initialNote);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function startEditing() {
    setDraft(note);
    setError(null);
    setEditing(true);
  }

  function cancelEditing() {
    setEditing(false);
    setError(null);
  }

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const result = await saveSessionNote(sessionId, draft.trim());
      setNote(result.note);
      setEditing(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save note');
    } finally {
      setSaving(false);
    }
  }

  if (editing) {
    return (
      <div className="session-note session-note--editing">
        <textarea
          className="session-note-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Escape') {
              cancelEditing();
            } else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
              void save();
            }
          }}
          placeholder="Story type, who's in the recording, observations…"
          rows={2}
          autoFocus
        />
        <div className="session-note-controls">
          <button
            className="session-note-save"
            onClick={() => void save()}
            disabled={saving}
          >
            {saving ? 'Saving…' : 'Save'}
          </button>
          <button
            className="session-note-cancel"
            onClick={cancelEditing}
            disabled={saving}
          >
            Cancel
          </button>
          {error && <span className="session-note-error">{error}</span>}
        </div>
      </div>
    );
  }

  return (
    <div className="session-note">
      {note ? (
        <button className="session-note-display" onClick={startEditing}>
          <span className="session-note-text">{note}</span>
          <span className="session-note-pencil" aria-hidden="true">
            ✎
          </span>
        </button>
      ) : (
        <button className="session-note-add" onClick={startEditing}>
          + add a note
        </button>
      )}
    </div>
  );
}
