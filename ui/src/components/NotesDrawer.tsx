/**
 * NotesDrawer — fixed right panel with two tabs: Notes and Low Confidence.
 *
 * Notes tab: sorted by timestamp, with jump/edit/delete per note.
 * Low Confidence tab: words with prob < 0.5 grouped by segment, click-to-seek.
 */

import { useMemo } from 'react';
import type { Note, ValidatorSegment, SegmentId } from '../types';
import { formatTime } from '../utils/time';
import './NotesDrawer.css';

interface NotesDrawerProps {
  open: boolean;
  tab: 'notes' | 'low-confidence';
  notes: Note[];
  segments: ValidatorSegment[];
  onTabChange: (tab: 'notes' | 'low-confidence') => void;
  onJump: (time: number) => void;
  onEditNote: (note: Note) => void;
  onDeleteNote: (noteId: string) => void;
  onDeleteAll: () => void;
}

interface LowConfidenceGroup {
  segmentId: SegmentId;
  segmentIndex: number;
  words: Array<{ word: string; start: number; probability: number; wordIndex: number }>;
}

export default function NotesDrawer({
  open,
  tab,
  notes,
  segments,
  onTabChange,
  onJump,
  onEditNote,
  onDeleteNote,
  onDeleteAll,
}: NotesDrawerProps) {
  const sortedNotes = useMemo(
    () => [...notes].sort((a, b) => a.timestamp - b.timestamp),
    [notes],
  );

  const lowConfidenceGroups = useMemo(() => {
    const groups: LowConfidenceGroup[] = [];
    segments.forEach((seg, segIdx) => {
      const lowWords = seg.words
        .map((w, wi) => ({ word: w.word, start: w.start, probability: w.probability, wordIndex: wi }))
        .filter((w) => w.probability < 0.5);
      if (lowWords.length > 0) {
        groups.push({ segmentId: seg.id, segmentIndex: segIdx, words: lowWords });
      }
    });
    return groups;
  }, [segments]);

  return (
    <div className={`notes-drawer ${open ? 'notes-drawer-open' : ''}`}>
      {/* Tab bar */}
      <div className="drawer-tabs">
        <button
          className={`drawer-tab ${tab === 'notes' ? 'drawer-tab-active' : ''}`}
          onClick={() => onTabChange('notes')}
        >
          Notes ({notes.length})
        </button>
        <button
          className={`drawer-tab ${tab === 'low-confidence' ? 'drawer-tab-active' : ''}`}
          onClick={() => onTabChange('low-confidence')}
        >
          Low Confidence
        </button>
      </div>

      {/* Tab content */}
      <div className="drawer-content">
        {tab === 'notes' && (
          <>
            {sortedNotes.length === 0 ? (
              <p className="drawer-empty">No notes yet. Right-click a segment or word, or press N.</p>
            ) : (
              <>
                <div className="drawer-actions-top">
                  <button className="btn btn--small btn--danger" onClick={onDeleteAll}>
                    Delete All
                  </button>
                </div>
                {sortedNotes.map((note) => (
                  <div key={note.id} className="note-card">
                    <div className="note-card-header">
                      <button className="note-jump" onClick={() => onJump(note.timestamp)}>
                        {formatTime(note.timestamp)}
                      </button>
                      {note.wordText && (
                        <span className="note-target">"{note.wordText}"</span>
                      )}
                      {note.segmentId !== null && !note.wordText && (
                        <span className="note-target">Seg {note.segmentId}</span>
                      )}
                    </div>
                    <p className="note-card-text">{note.text}</p>
                    <div className="note-card-actions">
                      <button className="btn btn--small" onClick={() => onEditNote(note)}>
                        Edit
                      </button>
                      <button className="btn btn--small btn--danger" onClick={() => onDeleteNote(note.id)}>
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </>
            )}
          </>
        )}

        {tab === 'low-confidence' && (
          <>
            {lowConfidenceGroups.length === 0 ? (
              <p className="drawer-empty">No low-confidence words found.</p>
            ) : (
              lowConfidenceGroups.map((group) => (
                <div key={group.segmentId} className="lc-group">
                  <div className="lc-group-header">Segment {group.segmentId}</div>
                  <div className="lc-chips">
                    {group.words.map((w) => (
                      <button
                        key={w.wordIndex}
                        className="lc-chip"
                        onClick={() => onJump(w.start)}
                        title={`prob: ${w.probability.toFixed(3)}`}
                      >
                        {w.word.trim()}
                        <span className="lc-chip-prob">{(w.probability * 100).toFixed(0)}%</span>
                      </button>
                    ))}
                  </div>
                </div>
              ))
            )}
          </>
        )}
      </div>
    </div>
  );
}
