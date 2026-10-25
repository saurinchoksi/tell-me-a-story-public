/**
 * Typed fetch wrapper for the tell-me-a-story API.
 *
 * All endpoints go through Vite's dev proxy (/api → localhost:5002),
 * so no absolute URLs or CORS configuration needed.
 */

import type {
  SessionSummary,
  SessionDetail,
  ValidationStatus,
  IdentificationData,
  ProfileSummary,
  ProfileDetail,
  Decision,
  ConfirmSpeakersResponse,
  Note,
  AxialLabel,
  DetectionsRollup,
  SessionDetectionsData,
  NameVerdict,
  MissedSpeechData,
  MissedHearing,
} from '../types';

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init);
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(body.error || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// --- Sessions ---

export async function listSessions(): Promise<SessionSummary[]> {
  const data = await fetchJSON<{ sessions: SessionSummary[] }>('/api/sessions');
  return data.sessions;
}

export async function getSession(id: string): Promise<SessionDetail> {
  return fetchJSON<SessionDetail>(`/api/sessions/${id}`);
}

export async function identifySpeakers(sessionId: string): Promise<IdentificationData> {
  return fetchJSON<IdentificationData>(`/api/sessions/${sessionId}/identify`, {
    method: 'POST',
  });
}

export function audioURL(sessionId: string): string {
  return `/api/sessions/${sessionId}/audio`;
}

export async function saveSessionNote(
  sessionId: string,
  note: string,
): Promise<{ note: string; updatedAt: string }> {
  return fetchJSON<{ note: string; updatedAt: string }>(
    `/api/sessions/${sessionId}/note`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ note }),
    },
  );
}

export async function setValidationStatus(
  sessionId: string,
  status: ValidationStatus,
): Promise<{ validationStatus: ValidationStatus; updatedAt: string }> {
  return fetchJSON<{ validationStatus: ValidationStatus; updatedAt: string }>(
    `/api/sessions/${sessionId}/validation-status`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    },
  );
}

// --- Speakers ---

export async function confirmSpeakers(
  sessionId: string,
  decisions: Decision[],
): Promise<ConfirmSpeakersResponse> {
  return fetchJSON<ConfirmSpeakersResponse>(
    `/api/sessions/${sessionId}/confirm-speakers`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decisions }),
    },
  );
}

// --- Profiles ---

export async function listProfiles(): Promise<ProfileSummary[]> {
  const data = await fetchJSON<{ profiles: ProfileSummary[] }>('/api/profiles');
  return data.profiles;
}

export async function createProfile(name: string, role: string): Promise<string> {
  const data = await fetchJSON<{ profile_id: string }>('/api/profiles', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, role }),
  });
  return data.profile_id;
}

export async function updateProfile(
  profileId: string,
  updates: { name?: string; role?: string },
): Promise<void> {
  await fetchJSON(`/api/profiles/${profileId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(updates),
  });
}

export async function getProfile(id: string): Promise<ProfileDetail> {
  return fetchJSON<ProfileDetail>(`/api/profiles/${id}`);
}

export async function deleteProfile(id: string): Promise<void> {
  await fetchJSON(`/api/profiles/${id}`, { method: 'DELETE' });
}

export async function refreshCentroid(id: string): Promise<void> {
  await fetchJSON(`/api/profiles/${id}/refresh-centroid`, { method: 'POST' });
}

export async function removeEmbedding(
  profileId: string,
  sessionId: string,
): Promise<{ embeddings_remaining: number }> {
  return fetchJSON(`/api/profiles/${profileId}/embeddings/${sessionId}`, {
    method: 'DELETE',
  });
}

// --- Detections (Monitor) ---

export async function getDetectionsRollup(): Promise<DetectionsRollup> {
  return fetchJSON<DetectionsRollup>('/api/detections');
}

export async function getSessionDetections(
  sessionId: string,
): Promise<SessionDetectionsData> {
  return fetchJSON<SessionDetectionsData>(`/api/sessions/${sessionId}/detections`);
}

/** Full re-scan of one session (code detectors + the M9b LLM judge). Slow. */
export async function scanSession(sessionId: string): Promise<SessionDetectionsData> {
  return fetchJSON<SessionDetectionsData>(
    `/api/sessions/${sessionId}/detections/scan`, { method: 'POST' });
}

/** Toggle a human name verdict (re-sending an identical one removes it). Returns fresh detail. */
export async function saveNameVerdict(
  sessionId: string,
  verdict: NameVerdict,
): Promise<SessionDetectionsData> {
  return fetchJSON<SessionDetectionsData>(
    `/api/sessions/${sessionId}/name-verdicts`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(verdict),
    },
  );
}

/** Re-scan every session whose results are missing or stale (full pass). */
export async function scanAllDetections(): Promise<DetectionsRollup> {
  return fetchJSON<DetectionsRollup>('/api/detections/scan', { method: 'POST' });
}

// --- Missed speech ---

/**
 * Missed-speech candidate spans for the validator waveform, computed live
 * server-side from transcript + diarization by the canonical EMP gap
 * detection (src/gaps.py). Read-only; default threshold is the >=1.0s
 * high-confidence floor.
 */
export async function getMissedSpeech(sessionId: string): Promise<MissedSpeechData> {
  return fetchJSON<MissedSpeechData>(`/api/sessions/${sessionId}/missed-speech`);
}

/**
 * Hear-box entries for a session's missed-speech candidates —
 * "what I hear" + verdict, keyed by candidate span. Human-authored sidecar
 * (missed-speech-verdicts.json); saving replaces the full array, like notes.
 */
export async function getMissedHearings(sessionId: string): Promise<MissedHearing[]> {
  const data = await fetchJSON<{ hearings: MissedHearing[] }>(
    `/api/sessions/${sessionId}/missed-speech/hearings`,
  );
  return data.hearings;
}

export async function saveMissedHearings(
  sessionId: string,
  hearings: MissedHearing[],
): Promise<{ saved: number }> {
  return fetchJSON<{ saved: number }>(
    `/api/sessions/${sessionId}/missed-speech/hearings`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hearings }),
    },
  );
}

// --- Notes ---

export async function getNotes(sessionId: string): Promise<Note[]> {
  const data = await fetchJSON<{ notes: Note[] }>(`/api/sessions/${sessionId}/notes`);
  return data.notes;
}

export async function saveNotes(sessionId: string, notes: Note[]): Promise<{ saved: number }> {
  return fetchJSON<{ saved: number }>(`/api/sessions/${sessionId}/notes`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes }),
  });
}

// --- Axial labels ---

/**
 * Server returns labels in one of two shapes (legacy single-code or current
 * multi-code). This shape captures both for the normalizer below.
 */
type RawAxialLabel = {
  segmentId: AxialLabel['segmentId'];
  createdAt: string;
  updatedAt: string;
} & ({ code: AxialLabel['codes'][number]; codes?: never } | { codes: AxialLabel['codes']; code?: never });

/**
 * Normalize a label entry into the multi-code shape. Single-code legacy entries
 * (`code: 'M3'`) become `codes: ['M3']`. Entries with neither field are corrupt
 * and we fail loud — silently dropping a label would lose data.
 */
function normalizeAxialLabel(raw: RawAxialLabel): AxialLabel {
  if ('codes' in raw && Array.isArray(raw.codes)) {
    return {
      segmentId: raw.segmentId,
      codes: raw.codes,
      createdAt: raw.createdAt,
      updatedAt: raw.updatedAt,
    };
  }
  if ('code' in raw && typeof raw.code === 'string') {
    return {
      segmentId: raw.segmentId,
      codes: [raw.code],
      createdAt: raw.createdAt,
      updatedAt: raw.updatedAt,
    };
  }
  throw new Error(
    `Corrupt axial label entry: missing both 'code' and 'codes' (segmentId=${raw.segmentId})`,
  );
}

export async function getAxialLabels(sessionId: string): Promise<AxialLabel[]> {
  const data = await fetchJSON<{ labels: RawAxialLabel[] }>(
    `/api/sessions/${sessionId}/axial-labels`,
  );
  return data.labels.map(normalizeAxialLabel);
}

export async function saveAxialLabels(
  sessionId: string,
  labels: AxialLabel[],
): Promise<{ saved: number }> {
  return fetchJSON<{ saved: number }>(`/api/sessions/${sessionId}/axial-labels`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ labels }),
  });
}

// --- Name-correction queue (namefix bless loop) ---

import type { NameCorrectionsRollup } from '../types';

export async function getNameCorrections(): Promise<NameCorrectionsRollup> {
  return fetchJSON<NameCorrectionsRollup>('/api/name-corrections');
}

export async function blessNameCorrection(
  sessionId: string,
  gid: string,
  canonical?: string,
  occurrences?: { segment_id: number | string; word_index: number }[],
): Promise<{ status: string; applied_occurrences: number; n_pending: number }> {
  return fetchJSON(`/api/sessions/${sessionId}/name-corrections/bless`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      gid,
      ...(canonical ? { canonical } : {}),
      ...(occurrences ? { occurrences } : {}),
    }),
  });
}

export async function rejectNameCorrection(
  sessionId: string,
  gid: string,
  _canonical?: string, // unused — keeps the verdict-action signature uniform
  occurrences?: { segment_id: number | string; word_index: number }[],
): Promise<{ status: string; n_pending: number }> {
  return fetchJSON(`/api/sessions/${sessionId}/name-corrections/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gid, ...(occurrences ? { occurrences } : {}) }),
  });
}

/** Lift a standing Name Book no (word-level) — matching folded groups return to the queue. */
export async function unrejectName(
  world: string,
  heardCleaned: string,
): Promise<{ status: string }> {
  return fetchJSON('/api/name-book/unreject', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ world, heard_cleaned: heardCleaned }),
  });
}

/** Lift a standing spelling pick (cluster-level) — the invented-name cluster's
 *  questions return to the live queue. */
export async function unpickSpelling(
  world: string,
  cluster: string,
): Promise<{ status: string }> {
  return fetchJSON('/api/name-book/unpick-spelling', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ world, cluster }),
  });
}

/** Undo a recorded verdict: pops the audit entry, reverses its Name Book write, and
 *  restores the question to the live queue. */
export async function unruleName(
  sessionId: string,
  verdict: 'yes' | 'no',
  gid: string,
  at?: string,
): Promise<{ status: string }> {
  return fetchJSON(`/api/sessions/${sessionId}/name-corrections/unrule`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ verdict, gid, ...(at ? { at } : {}) }),
  });
}

// --- Name pass (the by-ear gold-key builder, #4) ---

import type { NamePassPayload, NameTruthFieldSave, NameTruthItem } from '../types';

/** Name candidates for a session + the existing key as a cross-check seed. Model-free,
 *  sub-second — no model loads in the request. */
export async function getNamePass(sessionId: string): Promise<NamePassPayload> {
  return fetchJSON<NamePassPayload>(`/api/sessions/${sessionId}/name-pass`);
}

/** Field-routed save into the key (emp/results/visuals/<id>/name-truth.json). Writes
 *  ONE field so a save never wipes a sibling field or a between-turn edit. */
export async function saveNameTruthField(
  sessionId: string,
  body: NameTruthFieldSave,
): Promise<{ ok: boolean }> {
  return fetchJSON<{ ok: boolean }>(`/api/sessions/${sessionId}/name-truth`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/** Bulk ruling into the key. One item (an inline correction) or many (the "Mark heard"
 *  commit). `heard: true` stamps the ear-gate attestation. Sets only the given fields —
 *  never wipes a sibling. */
export async function commitNameTruth(
  sessionId: string,
  items: Record<string, Partial<NameTruthItem>>,
  heard = false,
): Promise<{ ok: boolean; committed: number }> {
  return fetchJSON<{ ok: boolean; committed: number }>(
    `/api/sessions/${sessionId}/name-truth/commit`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items, heard }),
    },
  );
}

// --- Roster (who exists — the /roster page, add-only) ---

import type { Roster, RosterPerson, RosterAlias } from '../types';

export async function getRoster(): Promise<Roster> {
  return fetchJSON<Roster>('/api/roster');
}

export async function addRosterPerson(
  canonical: string,
  circle: 'family' | 'friend',
): Promise<{ status: string; person: RosterPerson }> {
  return fetchJSON('/api/roster/people', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canonical, circle }),
  });
}

export async function addRosterNickname(
  personId: string,
  nickname: string,
): Promise<{ status: string; person: RosterPerson }> {
  return fetchJSON(`/api/roster/people/${personId}/nicknames`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nickname }),
  });
}

export async function addRosterAlias(
  personId: string,
  token: string,
): Promise<{ status: string; alias: RosterAlias }> {
  return fetchJSON(`/api/roster/people/${personId}/aliases`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
}

// --- Home canon (our characters — the Roster page's second registry) ---

import type { HomeCanon, HomeCharacter } from '../types';

export async function getHomeCanon(): Promise<HomeCanon> {
  return fetchJSON<HomeCanon>('/api/home-canon');
}

export async function addHomeCharacter(
  canonical: string,
): Promise<{ status: string; character: HomeCharacter }> {
  return fetchJSON('/api/home-canon/characters', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ canonical }),
  });
}
