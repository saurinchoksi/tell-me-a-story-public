import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import type { ProfileDetail as ProfileDetailType } from '../types';
import {
  getProfile,
  updateProfile,
  deleteProfile,
  refreshCentroid,
  removeEmbedding,
  audioURL,
} from '../api/client';
import AudioPlayer from '../components/AudioPlayer';
import { healthDotClass } from '../utils/health';
import './ProfileDetail.css';

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '-';
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return iso;
  }
}

function formatDuration(seconds: number | null): string {
  if (seconds == null) return '-';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

export default function ProfileDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [profile, setProfile] = useState<ProfileDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Inline editing state
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editRole, setEditRole] = useState('');

  const [refreshCounter, setRefreshCounter] = useState(0);

  useEffect(() => {
    if (!id) return;
    getProfile(id)
      .then((p) => {
        setProfile(p);
        setEditName(p.name);
        setEditRole(p.role);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id, refreshCounter]);

  // --- Handlers ---

  const handleSaveEdit = async () => {
    if (!id) return;
    try {
      await updateProfile(id, { name: editName, role: editRole });
      setEditing(false);
      setRefreshCounter(c => c + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Update failed');
    }
  };

  const handleCancelEdit = () => {
    if (profile) {
      setEditName(profile.name);
      setEditRole(profile.role);
    }
    setEditing(false);
  };

  const handleDelete = async () => {
    if (!id) return;
    if (!window.confirm(`Delete profile "${profile?.name}"? This cannot be undone.`)) return;
    try {
      await deleteProfile(id);
      navigate('/profiles');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed');
    }
  };

  const handleRefreshCentroid = async () => {
    if (!id) return;
    try {
      await refreshCentroid(id);
      setRefreshCounter(c => c + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Refresh failed');
    }
  };

  const handleRemoveEmbedding = async (sessionId: string) => {
    if (!id) return;
    if (!window.confirm(`Remove enrollment from session ${sessionId}?`)) return;
    try {
      await removeEmbedding(id, sessionId);
      setRefreshCounter(c => c + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Remove failed');
    }
  };

  // --- Render guards ---

  if (loading) return <p>Loading...</p>;
  if (error) return <p className="error">Error: {error}</p>;
  if (!profile) return <p className="error">Profile not found</p>;

  const matchScore = profile.latest_match?.confidence ?? null;

  return (
    <div className="profile-detail">
      <Link to="/profiles" className="profile-detail-back">
        &larr; All profiles
      </Link>

      {/* Header */}
      {editing ? (
        <div className="profile-detail-edit-form">
          <input
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            placeholder="Name"
          />
          <input
            value={editRole}
            onChange={(e) => setEditRole(e.target.value)}
            placeholder="Role"
          />
          <button className="btn" onClick={handleSaveEdit}>Save</button>
          <button className="btn" onClick={handleCancelEdit}>Cancel</button>
        </div>
      ) : (
        <>
          <div className="profile-detail-header">
            <span className={`health-dot ${healthDotClass(matchScore)}`} />
            <h1>{profile.name}</h1>
            <span className="profile-detail-role">{profile.role}</span>
            <button className="btn btn--small" onClick={() => setEditing(true)}>
              Edit
            </button>
          </div>
          <div className="profile-detail-meta">
            Created {formatDate(profile.created)} &middot; Updated {formatDate(profile.updated)}
          </div>
        </>
      )}

      {/* Audio sample */}
      {profile.audio_sample && (
        <div className="profile-detail-audio">
          <div className="profile-detail-audio-label">Voice sample</div>
          <AudioPlayer
            src={audioURL(profile.audio_sample.session_id)}
            seekTo={profile.audio_sample.start}
          />
        </div>
      )}

      {/* Actions */}
      <div className="profile-detail-actions">
        <button className="btn" onClick={handleRefreshCentroid}>
          Refresh Centroid
        </button>
        <button className="btn btn--danger" onClick={handleDelete}>
          Delete Profile
        </button>
      </div>

      {/* Enrollment sources */}
      <div className="profile-detail-section">
        <h2>Enrollment Sources</h2>
        {profile.embeddings.length === 0 ? (
          <p className="profile-detail-placeholder">
            No enrollments yet. Use Speaker Review to enroll this profile.
          </p>
        ) : (
          <table className="profile-detail-table">
            <thead>
              <tr>
                <th>Session</th>
                <th>Speaker</th>
                <th>Duration</th>
                <th>Segments</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {profile.embeddings.map((emb) => (
                <tr key={emb.session_id ?? 'unknown'}>
                  <td>
                    {emb.session_id ? (
                      <Link to={`/sessions/${emb.session_id}/speakers`}>
                        {emb.session_id}
                      </Link>
                    ) : (
                      '-'
                    )}
                  </td>
                  <td>{emb.source_speaker_key ?? '-'}</td>
                  <td>{formatDuration(emb.total_duration_s)}</td>
                  <td>{emb.num_segments ?? '-'}</td>
                  <td>
                    {emb.session_id && (
                      <button
                        className="btn btn--small btn--danger"
                        onClick={() => handleRemoveEmbedding(emb.session_id!)}
                      >
                        Remove
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Voice variants */}
      {profile.voice_variants.length > 0 && (
        <div className="profile-detail-section">
          <h2>Voice Variants</h2>
          <table className="profile-detail-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Session</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {profile.voice_variants.map((v) => (
                <tr key={v.id}>
                  <td><code>{v.id}</code></td>
                  <td>
                    {v.session_id ? (
                      <Link to={`/sessions/${v.session_id}/speakers`}>
                        {v.session_id}
                      </Link>
                    ) : (
                      '-'
                    )}
                  </td>
                  <td>{formatDate(v.created)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Match history placeholder */}
      <div className="profile-detail-section">
        <h2>Match History</h2>
        {profile.latest_match ? (
          <p className="profile-detail-placeholder">
            Latest: {(profile.latest_match.confidence * 100).toFixed(0)}% confidence
            in session {profile.latest_match.session_id}.
            More sessions needed for match history.
          </p>
        ) : (
          <p className="profile-detail-placeholder">
            No match data yet. Run speaker identification on a session to see results.
          </p>
        )}
      </div>
    </div>
  );
}
