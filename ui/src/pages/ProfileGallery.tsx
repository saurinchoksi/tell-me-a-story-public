import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import type { ProfileSummary } from '../types';
import { listProfiles } from '../api/client';
import { healthDotClass } from '../utils/health';
import './ProfileGallery.css';

function formatSessionId(sessionId: string): string {
  // "20260101-120000" -> "2026-01-01"
  const d = sessionId.slice(0, 8);
  return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`;
}

export default function ProfileGallery() {
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listProfiles()
      .then(setProfiles)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading...</p>;
  if (error) return <p className="error">Error: {error}</p>;

  if (profiles.length === 0) {
    return (
      <div>
        <div className="profile-gallery-header">
          <h1>Speaker Profiles</h1>
        </div>
        <div className="profile-gallery-cold-start">
          No speaker profiles yet. Process a session and use{' '}
          <Link to="/sessions">Speaker Review</Link> to create profiles.
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="profile-gallery-header">
        <h1>Speaker Profiles</h1>
      </div>

      <div className="profile-gallery-grid">
        {profiles.map((p) => (
          <Link
            key={p.id}
            to={`/profiles/${p.id}`}
            className="profile-card"
          >
            <div className="profile-card-header">
              <span className={`health-dot ${healthDotClass(p.latest_match_score)}`} />
              <span className="profile-card-name">{p.name}</span>
              <span className="profile-card-role">{p.role}</span>
            </div>

            <div className="profile-card-stats">
              <span className="profile-card-stat">
                {p.embeddings} enrollment{p.embeddings !== 1 ? 's' : ''}
              </span>
              {p.voice_variants > 0 && (
                <span className="profile-card-stat">
                  {p.voice_variants} variant{p.voice_variants !== 1 ? 's' : ''}
                </span>
              )}
              {p.latest_match_score != null && (
                <span className="profile-card-stat">
                  {(p.latest_match_score * 100).toFixed(0)}% match
                </span>
              )}
            </div>

            {p.last_seen && (
              <div className="profile-card-last-seen">
                Last seen: {formatSessionId(p.last_seen)}
              </div>
            )}
          </Link>
        ))}
      </div>
    </div>
  );
}
