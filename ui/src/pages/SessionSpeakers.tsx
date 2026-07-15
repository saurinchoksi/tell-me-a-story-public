import { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import type {
  SessionDetail,
  ProfileSummary,
  Decision,
  DiarizationSegment,
  SpeakerIdentification,
  IdentificationData,
} from '../types';
import { getSession, listProfiles, confirmSpeakers, audioURL } from '../api/client';
import AudioPlayer from '../components/AudioPlayer';
import SpeakerCard from '../components/SpeakerCard';
import ConfirmBar, { type SaveState } from '../components/ConfirmBar';
import './SessionSpeakers.css';

/** Group diarization segments by speaker key. */
function groupSegments(
  segments: DiarizationSegment[],
): Record<string, DiarizationSegment[]> {
  const groups: Record<string, DiarizationSegment[]> = {};
  for (const seg of segments) {
    (groups[seg.speaker] ??= []).push(seg);
  }
  return groups;
}

/** Build initial decisions from identification data. */
function initDecisions(
  speakerKeys: string[],
  identifications: IdentificationData | null,
): Record<string, Decision> {
  const idMap = new Map<string, SpeakerIdentification>();
  if (identifications) {
    for (const ident of identifications.identifications) {
      idMap.set(ident.speaker_key, ident);
    }
  }

  const decisions: Record<string, Decision> = {};
  for (const key of speakerKeys) {
    const ident = idMap.get(key);
    if (ident?.confirmed && ident.confirmed_profile_id) {
      // Normalize one-time actions: 'create' and 'reassign' already happened,
      // so on reload they become regular confirmations.
      const action = ident.confirmed_action === 'confirm_variant'
        ? 'confirm_variant'
        : 'confirm';
      decisions[key] = {
        speaker_key: key,
        action,
        profile_id: ident.confirmed_profile_id,
      };
    } else if (ident?.status === 'identified' && ident.profile_id) {
      decisions[key] = {
        speaker_key: key,
        action: 'confirm',
        profile_id: ident.profile_id,
      };
    } else {
      decisions[key] = { speaker_key: key, action: 'skip' };
    }
  }
  return decisions;
}

export default function SessionSpeakers() {
  const { id } = useParams<{ id: string }>();

  const [session, setSession] = useState<SessionDetail | null>(null);
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [identifications, setIdentifications] = useState<IdentificationData | null>(null);
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [seekTo, setSeekTo] = useState<number | undefined>(undefined);
  const [saveState, setSaveState] = useState<SaveState>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // --- Data fetching ---
  useEffect(() => {
    if (!id) return;
    Promise.all([getSession(id), listProfiles()])
      .then(([sess, profs]) => {
        setSession(sess);
        setProfiles(profs);
        setIdentifications(sess.identifications);

        const speakerKeys = sess.embeddings
          ? Object.keys(sess.embeddings.speakers).sort()
          : [];
        setDecisions(initDecisions(speakerKeys, sess.identifications));
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [id]);

  // --- Handlers ---
  const handleDecisionChange = useCallback((decision: Decision) => {
    setDecisions((prev) => ({ ...prev, [decision.speaker_key]: decision }));
  }, []);

  const handlePlay = useCallback((timestamp: number) => {
    // Tiny random delta ensures re-seek to the same timestamp triggers the effect
    setSeekTo(timestamp + Math.random() * 0.001);
  }, []);

  const handleSave = useCallback(async () => {
    if (!id) return;
    setSaveState('saving');
    setErrorMessage(null);

    try {
      const decisionList = Object.values(decisions);
      const result = await confirmSpeakers(id, decisionList);

      // Update identifications from response (for display badges)
      setIdentifications(result.identifications);

      // Re-fetch profiles (counts may have changed, new profiles created)
      const freshProfiles = await listProfiles();
      setProfiles(freshProfiles);

      // Rebuild decisions from fresh identifications (confirmed fields now persist)
      const speakerKeys = session?.embeddings
        ? Object.keys(session.embeddings.speakers).sort()
        : [];
      setDecisions(initDecisions(speakerKeys, result.identifications));

      setSaveState('success');
      setTimeout(() => setSaveState('idle'), 2000);
    } catch (e) {
      setSaveState('error');
      setErrorMessage(e instanceof Error ? e.message : 'Save failed');
    }
  }, [id, decisions, session]);

  // --- Render guards ---
  if (loading) return <p>Loading...</p>;
  if (error) return <p className="error">Error: {error}</p>;
  if (!session) return <p className="error">Session not found</p>;
  if (!session.embeddings) {
    return (
      <div>
        <h1>Speaker Review</h1>
        <p>No embeddings available for session <code>{id}</code>.</p>
        <p>Run the pipeline first to extract speaker embeddings.</p>
        <Link to="/sessions">&larr; Back to sessions</Link>
      </div>
    );
  }

  // --- Speaker data ---
  const speakerKeys = Object.keys(session.embeddings.speakers).sort();
  const segmentsBySpk = groupSegments(session.diarization?.segments ?? []);

  const idMap = new Map<string, SpeakerIdentification>();
  if (identifications) {
    for (const ident of identifications.identifications) {
      idMap.set(ident.speaker_key, ident);
    }
  }

  // Group speakers into sections
  const identified: string[] = [];
  const suggested: string[] = [];
  const unknown: string[] = [];

  for (const key of speakerKeys) {
    const ident = idMap.get(key);
    if (ident?.confirmed) identified.push(key);
    else if (ident?.status === 'identified') identified.push(key);
    else if (ident?.status === 'suggested') suggested.push(key);
    else unknown.push(key);
  }

  const isColdStart = profiles.length === 0 && unknown.length === speakerKeys.length;

  const renderCards = (keys: string[]) =>
    keys.map((key) => {
      const globalIndex = speakerKeys.indexOf(key);
      return (
        <SpeakerCard
          key={key}
          speakerKey={key}
          speakerIndex={globalIndex}
          embedding={session.embeddings!.speakers[key]}
          identification={idMap.get(key)}
          segments={segmentsBySpk[key] ?? []}
          decision={decisions[key] ?? { speaker_key: key, action: 'skip' }}
          onDecisionChange={handleDecisionChange}
          profiles={profiles}
          onPlay={handlePlay}
        />
      );
    });

  return (
    <div className="session-speakers">
      <Link to="/sessions" className="session-speakers-back">
        &larr; Sessions
      </Link>
      <div className="session-speakers-header">
        <h1>Speaker Review</h1>
        <code>{id}</code>
      </div>

      {session.has_audio && (
        <div className="session-speakers-audio">
          <AudioPlayer src={audioURL(id!)} seekTo={seekTo} />
        </div>
      )}

      {isColdStart && (
        <div className="session-speakers-guide">
          <strong>First session!</strong> Create profiles for each person so we can
          recognize them next time. Select &ldquo;+ New profile&rdquo; from the
          dropdown, enter a name and role, then save.
        </div>
      )}

      {identified.length > 0 && (
        <div className="session-speakers-group">
          <div className="session-speakers-group-label session-speakers-group-label--identified">
            Confirmed ({identified.length})
          </div>
          {renderCards(identified)}
        </div>
      )}

      {suggested.length > 0 && (
        <div className="session-speakers-group">
          <div className="session-speakers-group-label session-speakers-group-label--suggested">
            Review ({suggested.length})
          </div>
          {renderCards(suggested)}
        </div>
      )}

      {unknown.length > 0 && (
        <div className="session-speakers-group">
          <div className="session-speakers-group-label session-speakers-group-label--unknown">
            Unknown ({unknown.length})
          </div>
          {renderCards(unknown)}
        </div>
      )}

      <ConfirmBar
        decisions={decisions}
        saveState={saveState}
        errorMessage={errorMessage}
        onSave={handleSave}
      />
    </div>
  );
}
