/**
 * useAutoScroll — scroll-to-active-segment with user-scroll-away detection.
 *
 * When audio is playing, auto-scrolls to keep the active segment visible.
 * If the user scrolls manually, auto-scroll pauses for 5 seconds, then resumes.
 * Programmatic scrolls (from scrollToSegment) are distinguished from user scrolls
 * via a flag so they don't trigger the pause.
 */

import { useRef, useEffect, useCallback } from 'react';

const PAUSE_DURATION_MS = 5000;

export function useAutoScroll(containerRef: React.RefObject<HTMLDivElement | null>) {
  const userScrolledAwayRef = useRef(false);
  const programmaticScrollUntilRef = useRef(0);
  const resumeTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // Scroll listener — detects user-initiated scrolls
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (Date.now() < programmaticScrollUntilRef.current) {
        // Within programmatic scroll window, ignore
        return;
      }

      // User scrolled away — pause auto-scroll
      userScrolledAwayRef.current = true;

      // Reset after 5 seconds
      clearTimeout(resumeTimerRef.current);
      resumeTimerRef.current = setTimeout(() => {
        userScrolledAwayRef.current = false;
      }, PAUSE_DURATION_MS);
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      container.removeEventListener('scroll', handleScroll);
      clearTimeout(resumeTimerRef.current);
    };
  }, [containerRef]);

  /** Scroll a segment card into view. Call from the timeupdate handler. */
  const scrollToSegment = useCallback(
    (element: HTMLElement) => {
      if (userScrolledAwayRef.current) return;

      programmaticScrollUntilRef.current = Date.now() + 500;
      element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    },
    [],
  );

  return { scrollToSegment, userScrolledAwayRef };
}
