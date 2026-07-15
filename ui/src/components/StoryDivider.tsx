/**
 * StoryDivider — a labelled break between bedtime stories in the validator transcript.
 *
 * The pipeline splits each recording into its stories and tags every in-story segment
 * with a `_story` index (see src/stories.py). The validator renders one of these dividers
 * wherever that index changes, showing the story's title and inferred world.
 */
import './StoryDivider.css';

interface StoryDividerProps {
  title: string;
  world: string;
}

export default function StoryDivider({ title, world }: StoryDividerProps) {
  return (
    <div className="story-divider" role="separator">
      <span className="story-divider__title">{title || 'Story'}</span>
      {world && <span className="story-divider__world">{world}</span>}
    </div>
  );
}
