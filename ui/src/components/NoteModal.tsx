/**
 * NoteModal — modal for creating/editing validation notes.
 *
 * Target-aware title ("Add Note for Segment 5", "Edit Note for 'Arjuna'").
 * Pre-populates textarea for edits. Escape to cancel, Enter to save
 * (without Shift). Rendered as a portal overlay.
 *
 * Uses a key-based reset: NoteModalContent remounts when the existingNote
 * changes, so initialText is always fresh without needing setState in an effect.
 */

import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import type { NoteModalState, Note, ContextTarget } from '../types';
import './NoteModal.css';

interface NoteModalProps {
  modal: NoteModalState;
  onSave: (text: string, target: ContextTarget, existingNote: Note | null) => void;
  onDelete: (noteId: string) => void;
  onCancel: () => void;
}

function getTitle(target: ContextTarget | null, isEdit: boolean): string {
  const prefix = isEdit ? 'Edit Note' : 'Add Note';
  if (!target) return prefix;

  switch (target.type) {
    case 'word':
      return `${prefix} for "${target.wordText}"`;
    case 'segment':
      return `${prefix} for Segment ${target.segmentId}`;
    case 'timestamp':
      return `${prefix} at ${target.timestamp?.toFixed(1)}s`;
    default:
      return prefix;
  }
}

/** Inner content — remounted via key to reset textarea state cleanly. */
function NoteModalContent({
  initialText,
  title,
  isEdit,
  modal,
  onSave,
  onDelete,
  onCancel,
}: {
  initialText: string;
  title: string;
  isEdit: boolean;
  modal: NoteModalState;
  onSave: NoteModalProps['onSave'];
  onDelete: NoteModalProps['onDelete'];
  onCancel: NoteModalProps['onCancel'];
}) {
  const [text, setText] = useState(initialText);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleSave = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onSave(trimmed, modal.target!, modal.existingNote);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSave();
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      onCancel();
    }
  };

  return (
    <div className="note-modal-overlay" onClick={onCancel}>
      <div className="note-modal" onClick={(e) => e.stopPropagation()}>
        <h3 className="note-modal-title">{title}</h3>
        <textarea
          ref={textareaRef}
          className="note-modal-textarea"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Write your note..."
          rows={3}
        />
        <div className="note-modal-actions">
          {isEdit && (
            <button
              className="btn btn--small btn--danger"
              onClick={() => onDelete(modal.existingNote!.id)}
            >
              Delete
            </button>
          )}
          <div className="note-modal-spacer" />
          <button className="btn btn--small" onClick={onCancel}>
            Cancel
          </button>
          <button
            className="btn btn--small btn--active"
            onClick={handleSave}
            disabled={!text.trim()}
          >
            {isEdit ? 'Update' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default function NoteModal({ modal, onSave, onDelete, onCancel }: NoteModalProps) {
  if (!modal.visible || !modal.target) return null;

  const isEdit = modal.existingNote !== null;
  const title = getTitle(modal.target, isEdit);
  // Key includes note ID + timestamp so remount happens on each new modal open
  const contentKey = modal.existingNote?.id ?? `new-${modal.target.timestamp}`;

  return createPortal(
    <NoteModalContent
      key={contentKey}
      initialText={modal.existingNote?.text ?? ''}
      title={title}
      isEdit={isEdit}
      modal={modal}
      onSave={onSave}
      onDelete={onDelete}
      onCancel={onCancel}
    />,
    document.body,
  );
}
