import type { StorySummary } from '../types';

/**
 * Lead-story line for a session: the first title plus "+N more", or a bare
 * count when the stories carry no titles (any recognized worlds already show as
 * chips). Lets a row or header say what it holds instead of just its date.
 * Lives here so the Sessions list and the StoryHeading derive it identically.
 */
export function storyTitleLine(st: StorySummary): string {
  if (st.titles.length === 0) {
    return st.n_stories > 1 ? `${st.n_stories} stories` : '';
  }
  const lead = st.titles[0];
  return st.n_stories > 1 ? `${lead} + ${st.n_stories - 1} more` : lead;
}
