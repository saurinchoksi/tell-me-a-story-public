/**
 * ContextMenu — right-click menu rendered via portal.
 *
 * Auto-adjusts position near viewport edges. Dismissed on click-outside,
 * scroll, or Escape key (handled by keyboard shortcuts hook).
 */

import { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import type { ContextMenuState } from '../types';
import './ContextMenu.css';

interface ContextMenuProps {
  menu: ContextMenuState;
  onAddNote: () => void;
  onDismiss: () => void;
}

export default function ContextMenu({ menu, onAddNote, onDismiss }: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!menu.visible) return;

    // Adjust position if too close to viewport edges
    const el = menuRef.current;
    if (el) {
      const rect = el.getBoundingClientRect();
      if (rect.right > window.innerWidth) {
        el.style.left = `${menu.x - rect.width}px`;
      }
      if (rect.bottom > window.innerHeight) {
        el.style.top = `${menu.y - rect.height}px`;
      }
    }

    // Dismiss on click-outside or scroll (next tick to avoid catching the opener)
    const dismiss = () => onDismiss();
    const timerId = setTimeout(() => {
      window.addEventListener('mousedown', dismiss);
      window.addEventListener('scroll', dismiss, true);
    }, 0);

    return () => {
      clearTimeout(timerId);
      window.removeEventListener('mousedown', dismiss);
      window.removeEventListener('scroll', dismiss, true);
    };
  }, [menu.visible, menu.x, menu.y, onDismiss]);

  if (!menu.visible || !menu.target) return null;

  const label =
    menu.target.type === 'word'
      ? `Add note for "${menu.target.wordText}"`
      : menu.target.type === 'segment'
        ? `Add note for Segment ${menu.target.segmentId}`
        : `Add note at ${menu.target.timestamp?.toFixed(1)}s`;

  return createPortal(
    <div
      ref={menuRef}
      className="context-menu"
      style={{ left: menu.x, top: menu.y }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <button className="context-menu-item" onClick={onAddNote}>
        {label}
      </button>
    </div>,
    document.body,
  );
}
