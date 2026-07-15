/** Shared time formatting utility. */

export function formatTime(seconds: number): string {
  if (!isFinite(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

/** Parse session ID (YYYYMMDD-HHMMSS) into human-readable date and time. */
export function formatSessionDate(sessionId: string): { date: string; time: string } {
  const match = sessionId.match(/^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/);
  if (!match) return { date: sessionId, time: '' };
  const [, year, month, day, hour, min, sec] = match;
  const d = new Date(+year, +month - 1, +day, +hour, +min, +sec);
  return {
    date: d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }),
    time: d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' }),
  };
}
