import { useCallback } from 'react';
import type {
  Decision,
  SpeakerEmbedding,
  SpeakerIdentification,
  DiarizationSegment,
  ProfileSummary,
} from '../types';
import './SpeakerCard.css';

const STATUS_ICON: Record<string, string> = {
  identified: '\u2713',
  suggested: '~',
  unknown: '?',
};

const STATUS_CLASS: Record<string, string> = {
  identified: 'speaker-card-status--identified',
  suggested: 'speaker-card-status--suggested',
  unknown: 'speaker-card-status--unknown',
};

interface SpeakerCardProps {
  speakerKey: string;
  speakerIndex: number;
  embedding: SpeakerEmbedding;
  identification: SpeakerIdentification | undefined;
  segments: DiarizationSegment[];
  decision: Decision;
  onDecisionChange: (decision: Decision) => void;
  profiles: ProfileSummary[];
  onPlay: (timestamp: number) => void;
}

/** Find a representative segment — prefer >= 3s, near the midpoint of total speech time. */
function findRepresentativeSegment(segments: DiarizationSegment[]): DiarizationSegment | null {
  if (segments.length === 0) return null;

  const totalDuration = segments.reduce((sum, s) => sum + (s.end - s.start), 0);
  const midTarget = totalDuration / 2;

  let cumulative = 0;
  let midSegment = segments[0];
  for (const seg of segments) {
    cumulative += seg.end - seg.start;
    if (cumulative >= midTarget) {
      midSegment = seg;
      break;
    }
  }

  // Prefer a segment >= 3s near the midpoint
  const longSegments = segments.filter(s => (s.end - s.start) >= 3);
  if (longSegments.length > 0) {
    // Pick the long segment closest to midSegment
    longSegments.sort(
      (a, b) => Math.abs(a.start - midSegment.start) - Math.abs(b.start - midSegment.start)
    );
    return longSegments[0];
  }

  return midSegment;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export default function SpeakerCard({
  speakerKey,
  speakerIndex,
  embedding,
  identification,
  segments,
  decision,
  onDecisionChange,
  profiles,
  onPlay,
}: SpeakerCardProps) {
  const status = identification?.status ?? 'unknown';
  const totalDuration = segments.reduce((sum, s) => sum + (s.end - s.start), 0);
  const repSegment = findRepresentativeSegment(segments);

  const handlePlay = useCallback(() => {
    if (repSegment) {
      onPlay(repSegment.start);
    }
  }, [repSegment, onPlay]);

  const handleProfileSelect = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value;
      if (value === '__skip__') {
        onDecisionChange({ speaker_key: speakerKey, action: 'skip' });
      } else if (value === '__create__') {
        onDecisionChange({
          speaker_key: speakerKey,
          action: 'create',
          new_name: '',
          new_role: '',
        });
      } else {
        onDecisionChange({
          speaker_key: speakerKey,
          action: 'confirm',
          profile_id: value,
        });
      }
    },
    [speakerKey, onDecisionChange],
  );

  const handleVariantToggle = useCallback(() => {
    if (decision.action === 'confirm_variant') {
      // Toggle back to confirm
      onDecisionChange({
        speaker_key: speakerKey,
        action: 'confirm',
        profile_id: decision.profile_id,
      });
    } else if (decision.action === 'confirm' && decision.profile_id) {
      onDecisionChange({
        speaker_key: speakerKey,
        action: 'confirm_variant',
        profile_id: decision.profile_id,
      });
    }
  }, [speakerKey, decision, onDecisionChange]);

  const handleNameChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onDecisionChange({ ...decision, new_name: e.target.value });
    },
    [decision, onDecisionChange],
  );

  const handleRoleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onDecisionChange({ ...decision, new_role: e.target.value });
    },
    [decision, onDecisionChange],
  );

  // Determine the current dropdown value
  let selectValue = '__skip__';
  if (decision.action === 'create') {
    selectValue = '__create__';
  } else if (decision.action === 'confirm' || decision.action === 'confirm_variant' || decision.action === 'reassign') {
    selectValue = decision.profile_id ?? '__skip__';
  }

  const speakerColorVar = { '--speaker-color': `var(--speaker-${speakerIndex % 5})` } as React.CSSProperties;

  return (
    <div className="speaker-card" style={speakerColorVar}>
      <div className="speaker-card-info">
        <div className="speaker-card-label">
          <span className={`speaker-card-status ${STATUS_CLASS[status]}`}>
            {STATUS_ICON[status]}
          </span>
          {speakerKey}
        </div>
        <div className="speaker-card-meta">
          {formatDuration(totalDuration)} · {segments.length} segment{segments.length !== 1 ? 's' : ''}
          {embedding.num_segments !== undefined && ` · ${embedding.num_segments} emb`}
        </div>
        {identification?.confidence != null && (
          <div className="speaker-card-confidence">
            {identification.profile_name} ({(identification.confidence * 100).toFixed(0)}%)
          </div>
        )}
      </div>

      <button
        className="speaker-card-play"
        onClick={handlePlay}
        disabled={!repSegment}
        title={repSegment ? `Play from ${formatDuration(repSegment.start)}` : 'No segments'}
      >
        &#9654;
      </button>

      <div className="speaker-card-controls">
        <select
          className="speaker-card-select"
          value={selectValue}
          onChange={handleProfileSelect}
        >
          <option value="__skip__">-- Skip --</option>
          <option value="__create__">+ New profile</option>
          {profiles.map((p) => (
            <option key={p.id} value={p.id}>
              {p.name} ({p.role})
            </option>
          ))}
        </select>

        {(decision.action === 'confirm' || decision.action === 'confirm_variant') && decision.profile_id && (
          <button
            className="speaker-card-variant-toggle"
            onClick={handleVariantToggle}
            title="Voice variants are preserved but don't affect speaker identity"
          >
            {decision.action === 'confirm_variant' ? 'variant (undo)' : 'mark as variant'}
          </button>
        )}

        {decision.action === 'create' && (
          <div className="speaker-card-create">
            <input
              type="text"
              placeholder="Name"
              value={decision.new_name ?? ''}
              onChange={handleNameChange}
            />
            <input
              type="text"
              placeholder="Role"
              value={decision.new_role ?? ''}
              onChange={handleRoleChange}
            />
          </div>
        )}
      </div>
    </div>
  );
}
