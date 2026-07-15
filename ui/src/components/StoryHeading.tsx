import type { StorySummary } from '../types';
import { storyTitleLine } from '../utils/stories';
import './StoryHeading.css';

/**
 * Rich story identity for a per-session detail header: 📖 + a chip per
 * recognized world + the lead story title. Renders nothing when the session has
 * no stories. Mirrors the Sessions-list line so a drilled-in session stays
 * recognizable by its content, not just its date.
 */
export default function StoryHeading({ stories }: { stories: StorySummary | null }) {
  if (!stories) return null;
  const title = storyTitleLine(stories);
  return (
    <div className="story-heading">
      <span className="story-heading-icon" aria-hidden="true">📖</span>
      {stories.worlds.map((w) => (
        <span key={w} className="story-chip">{w}</span>
      ))}
      {title && <span className="story-heading-title">{title}</span>}
    </div>
  );
}
