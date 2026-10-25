/** TypeScript interfaces matching API response shapes. */

/**
 * Segment identifier. Integer for real Whisper segments; string for injected
 * gap segments (e.g. "gap_241.680" from speaker.py gap detection). The union
 * is intentional — consumers must accept both, not narrow to `number`.
 */
export type SegmentId = number | string;

/** Human-review status for a session, set from the Sessions list. */
export type ValidationStatus = 'not_started' | 'in_progress' | 'done';

/**
 * Glanceable summary of a session's stories, derived server-side from the
 * transcript's `_stories`. `label` is a compact one-liner (for tight spaces
 * like the Monitor); `worlds` (distinct recognized worlds) and `titles` let a
 * roomier view render chips + the lead title. Null/absent when a session has no
 * stories (untranscribed, or a transcript predating story segmentation).
 */
export interface StorySummary {
  label: string;
  n_stories: number;
  worlds: string[];
  titles: string[];
}

export interface SessionSummary {
  id: string;
  has_audio: boolean;
  has_transcript: boolean;
  has_diarization: boolean;
  has_embeddings: boolean;
  has_identifications: boolean;
  /** Session-level free-text note; '' when none set. */
  note: string;
  /** Recording duration in seconds; null when the session isn't transcribed. */
  duration_seconds: number | null;
  /** Count of timestamped validation notes. */
  note_count: number;
  /** Human-review status. */
  validation_status: ValidationStatus;
  /** Pipeline stages that errored (from transcript-rich.json _processing); [] when healthy. */
  failed_stages: string[];
  /** What this recording holds — derived from the story map; null when none. */
  stories: StorySummary | null;
}

export interface SessionDetail {
  id: string;
  has_audio: boolean;
  transcript: TranscriptData | null;
  diarization: DiarizationData | null;
  embeddings: EmbeddingsData | null;
  identifications: IdentificationData | null;
}

export interface StoryRegion {
  index: number;
  start_id: SegmentId;
  end_id: SegmentId;
  title: string;
  world: string;
}

export interface TranscriptData {
  segments: TranscriptSegment[];
  _stories?: StoryRegion[];
  [key: string]: unknown;
}

export interface TranscriptSegment {
  id: SegmentId;
  start: number;
  end: number;
  text: string;
  words?: TranscriptWord[];
  [key: string]: unknown;
}

export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
  probability?: number;
  _speaker?: { label: string };
  [key: string]: unknown;
}

export interface DiarizationData {
  segments: DiarizationSegment[];
  [key: string]: unknown;
}

export interface DiarizationSegment {
  speaker: string;
  start: number;
  end: number;
}

export interface EmbeddingsData {
  speakers: Record<string, SpeakerEmbedding>;
  _generator_version?: string;
  _dimension?: number;
}

export interface SpeakerEmbedding {
  vector: number[];
  num_segments: number;
  total_duration_s?: number;
}

export interface IdentificationData {
  session_id: string;
  identified_at: string;
  profiles_used: number;
  identifications: SpeakerIdentification[];
}

export interface SpeakerIdentification {
  speaker_key: string;
  status: 'identified' | 'suggested' | 'unknown';
  profile_id: string | null;
  profile_name: string | null;
  confidence: number | null;
  confirmed?: boolean;
  confirmed_action?: DecisionAction;
  confirmed_profile_id?: string;
}

/** Profile as returned by GET /api/profiles (vectors stripped). */
export interface ProfileSummary {
  id: string;
  name: string;
  role: string;
  created: string;
  updated: string;
  embeddings: number;       // count, not the actual vectors
  voice_variants: number;   // count
  latest_match_score?: number | null;
  latest_match_session?: string | null;
  last_seen?: string | null;
}

/** Enriched embedding info from GET /api/profiles/:id. */
export interface ProfileEmbeddingInfo {
  session_id: string | null;
  source_speaker_key: string | null;
  total_duration_s: number | null;
  num_segments: number | null;
}

/** Voice variant metadata from GET /api/profiles/:id. */
export interface ProfileVariantInfo {
  id: string;
  created: string;
  session_id: string | null;
  source_speaker_key: string | null;
}

/** Reference to an audio segment for speaker preview. */
export interface AudioSampleRef {
  session_id: string;
  start: number;
  end: number;
}

/** Most recent identification match for a profile. */
export interface LatestMatch {
  session_id: string;
  confidence: number;
  identified_at: string;
}

/** Full profile detail from GET /api/profiles/:id. */
export interface ProfileDetail {
  id: string;
  name: string;
  role: string;
  created: string;
  updated: string;
  embeddings: ProfileEmbeddingInfo[];
  voice_variants: ProfileVariantInfo[];
  audio_sample: AudioSampleRef | null;
  latest_match: LatestMatch | null;
}

// --- Validator types ---

export interface ValidatorWord {
  word: string;
  start: number;
  end: number;
  probability: number;
  _speaker?: {
    label: string | null;
    coverage: number;
  };
  _original?: string;
  _corrections?: Array<{
    stage: string;
    from: string;
    to: string;
  }>;
}

export interface ValidatorSegment {
  id: SegmentId;
  start: number;
  end: number;
  text: string;
  words: ValidatorWord[];
  temperature?: number;
  compression_ratio?: number;
  no_speech_prob?: number;
  avg_logprob?: number;
  _speaker?: {
    label: string | null;
    coverage: number;
  };
  _source?: string;
  _story?: number;
}

/**
 * One missed-speech candidate span: diarization heard a voice but
 * NO segment of any kind exists there — no transcribed line, not even an
 * [unintelligible] placeholder. Computed live by the API from the canonical
 * EMP gap detection (src/gaps.py); the validator only displays it.
 */
export interface MissedSpeechCandidate {
  start: number;
  end: number;
  dur: number;
  speaker: string;
}

export interface MissedSpeechData {
  candidates: MissedSpeechCandidate[];
  min_gap: number;
}

/**
 * The four things an ear can conclude about a missed-speech candidate:
 * speech the transcript truly missed, noise adjacent to a hallucinated
 * stretch, the uncovered tail of a partially captured word, or nothing there.
 * Mirrors HEARING_VERDICTS (api/routes/missed_speech.py) and the VERDICTS
 * labels (components/MissedHearBox.tsx) — the three move together.
 */
export type HearingVerdict =
  | 'real-speech'
  | 'hallucination-adjacent'
  | 'boundary-tail'
  | 'nothing';

/**
 * One hear-box entry: what the ear heard at a missed-speech
 * candidate, keyed by the candidate's span (start/end) since no transcript
 * segment exists to anchor it. Persists to missed-speech-verdicts.json — a
 * human-authored sidecar, never a transcript.
 */
export interface MissedHearing {
  start: number;
  end: number;
  /**
   * Threshold the candidate was queried at — provenance only, deliberately
   * NOT part of the join key: gap spans are computed threshold-independently
   * (src/gaps.py; min_gap just filters by duration), so the same span at any
   * tier is the same acoustic stretch and shares its hearing.
   */
  min_gap: number;
  /** "What I hear" free text — the by-ear ground truth, verbatim. */
  hear: string;
  /** null while only text has been typed. */
  verdict: HearingVerdict | null;
  createdAt: string;
  updatedAt: string;
}

export interface Note {
  id: string;
  segmentId: SegmentId | null;
  wordIndex: number | null;
  wordText: string | null;
  wordStart: number | null;
  timestamp: number;
  text: string;
  createdAt: string;
}

/** Per-segment axial codes for EMP step 5 counting. Multiple codes allowed per segment. */
export type AxialCode = 'M1' | 'M2' | 'M3' | 'M4' | 'M5' | 'M6' | 'M7' | 'M8' | 'M9' | 'M10' | 'NotA';

export interface AxialLabel {
  segmentId: SegmentId;
  codes: AxialCode[];
  createdAt: string;
  updatedAt: string;
}

export interface FilterState {
  silenceGap: boolean;
  nearZero: boolean;
  duplicates: boolean;
}

export interface ContextTarget {
  type: 'segment' | 'word' | 'timestamp';
  segmentId?: SegmentId;
  segmentIndex?: number;
  wordIndex?: number;
  wordText?: string;
  wordStart?: number;
  timestamp?: number;
}

export interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  target: ContextTarget | null;
}

export interface NoteModalState {
  visible: boolean;
  target: ContextTarget | null;
  existingNote: Note | null;
}

// --- Detections (Monitor) ---

/** Registry metadata for one failure-mode detector. */
export interface DetectorInfo {
  id: string;
  label: string;
  failure_mode: string;
  version: string;
}

/** One detector's run summary on one session (rollup view). */
export interface DetectionRunSummary {
  n_flags: number;
  run_at: string;
  detector_version: string;
}

export interface DetectionsRollupSession {
  session_id: string;
  duration_seconds: number | null;
  /** What this recording holds — derived from the story map; null when none. */
  stories: StorySummary | null;
  /** Keyed by detector id. A missing key means "never scanned" — distinct from 0 flags. */
  results: Record<string, DetectionRunSummary>;
  /** The transcript changed since this session was scanned — re-scan to refresh. */
  stale: boolean;
}

export interface DetectionsRollup {
  detectors: DetectorInfo[];
  sessions: DetectionsRollupSession[];
  totals: Record<string, number>;
  /** Set when a scan ran code-only because the LLM judge's venv was absent. */
  warning?: string;
}

/** Fields every detector's flag carries — the anchor + the server-side join. */
export interface DetectionFlagBase {
  segment_id: SegmentId;
  word_index: number;
  start: number | null;
  end: number | null;
  token: string;
  cleaned: string;
  dm_codes: string[];
  segment_text: string | null;
  segment_start: number | null;
  segment_end: number | null;
  segment_speaker: string | null;
}

/** M9a — a token matched against the family-name roster. */
export interface FamilyNameFlag extends DetectionFlagBase {
  match_type: 'phonetic' | 'alias';
  matched_person_ids: string[];
  matched_canonicals: string[];
}

/** M9b — a token in a name spelled inconsistently across the session. Only the
 *  deviating (non-majority) spellings are flagged; `majority_spelling` is what this
 *  recording mostly used (the odd token differs from it), or null when no spelling
 *  dominates. It is a within-recording reference, NOT a global canonical. */
export interface NameConsistencyFlag extends DetectionFlagBase {
  cluster_id: string;
  cluster_spellings: string[];
  n_cluster_occurrences: number;
  majority_spelling: string | null;
  majority_count: number | null;
}

/** M9c — a sourced-canon name (Thomas / Mahabharata) spelled wrong, from the
 *  per-story reader. Carries the correct spelling and the story's inferred world. */
export interface CanonNameFlag extends DetectionFlagBase {
  case: 'M9c';
  canonical: string;
  card_id: number;
  all_spellings: string[];
  wrong_cleaned: string[];
  evidence: string;
  story_id: number;
  story_world: string;
  /** Confidence in the SUGGESTED spelling: true when the heard token sounds like the
   *  canonical (shared Double-Metaphone code). false = a "best guess" (e.g. Urzi→Urjani).
   *  Optional: pre-0.4.0 flags lack it. */
  suggestion_confident?: boolean;
  /** How the catch was made — informational, not the confidence driver. */
  methods?: string[];
  /** Order-robust judge: rounds (of vote_rounds) that agreed this is a misspelled canon name. */
  vote_count?: number;
  vote_rounds?: number;
  /** View-time confidence tier (api.helpers.canon_tier): 'confident' (sound-alike) and
   *  'best_guess' (judge vote ≥ 4) show by default; 'low' shows only under "Show all". */
  tier?: 'confident' | 'best_guess' | 'low';
}

/** Discriminated by a field unique to each shape: `cluster_spellings` → m9b,
 *  `matched_canonicals` → m9a, `canonical` → m9c. Render code must narrow on one
 *  of these and fall back gracefully, never assuming a single shape. */
export type DetectionFlag = FamilyNameFlag | NameConsistencyFlag | CanonNameFlag;

export interface SessionDetectorResult {
  label: string;
  failure_mode: string;
  detector_version: string;
  run_at: string;
  n_word_tokens: number;
  n_flags: number;
  flags: DetectionFlag[];
  /** The transcript changed since this section was scanned. */
  stale: boolean;
  /** Whether the LLM judge ran on this section (M9b full pass). */
  judge_applied?: boolean;
}

/**
 * A human "set the record straight" verdict on a flagged name, persisted beside the
 * transcript so it survives every re-scan. `not_canon` drops a wrongly-claimed canon
 * name from M9c (releasing it back to M9b); `correct` marks a spelling as right, so it
 * stops being flagged in M9b/M9c. `stale` is set when the name's spellings drifted since.
 */
export type NameVerdict =
  | { type: 'not_canon'; name: string; cleaned_at_review?: string[]; note?: string; at?: string; stale?: boolean }
  | { type: 'correct'; cleaned: string; spelling?: string; note?: string; at?: string; stale?: boolean };

export interface SessionDetectionsData {
  session_id: string;
  /** Whether an audio file exists — gates the per-flag "play clip" control. */
  has_audio: boolean;
  /** What this recording holds — derived from the story map; null when none. */
  stories: StorySummary | null;
  /** Human corrections applied to this session's flags (see NameVerdict). */
  name_verdicts: NameVerdict[];
  /** Keyed by detector id; {} when the session was never scanned. */
  detectors: Record<string, SessionDetectorResult>;
  /** Set when a scan ran code-only because the LLM judge's venv was absent. */
  warning?: string;
}

// --- Speaker confirmation (Task 6) ---

export type DecisionAction = 'confirm' | 'confirm_variant' | 'create' | 'reassign' | 'skip';

export interface Decision {
  speaker_key: string;
  action: DecisionAction;
  profile_id?: string;
  new_name?: string;
  new_role?: string;
}

export interface ConfirmSpeakersResponse {
  applied: number;
  skipped: number;
  created_profiles: string[];
  identifications: IdentificationData;
}

// --- Name-correction queue (namefix bless loop) ---

export interface NameCorrectionOccurrence {
  segment_id: number | string;
  word_index: number;
  start: number;
  token: string;
  /** The occurrence's full sentence + the judged word's char span within it —
   *  joined server-side so the reviewer knows WHICH of several nearby names
   *  they're judging. Absent on pending files older than the enrichment. */
  segment_text?: string;
  word_offset?: number;
  word_len?: number;
  /** the opt-in dictionary pass (Pass 5) rewrote this spot in place AFTER the
   *  fixer filed the question, so the highlighted sentence now shows the corrected
   *  surface, not the word as first heard. A GET-time, read-only annotation — the queue
   *  is never written. `now_reads` is the current display token at the spot. */
  corrected_by_dictionary?: boolean;
  now_reads?: string;
}

/** One verdict covering a spot-answered fold — what the inline Undo needs. */
export interface CoveringVerdict {
  verdict: 'yes' | 'no';
  gid: string;
  canonical?: string;
  at: string;
  n_spots: number;
}

export interface NameCorrectionSessionEntry {
  session_id: string;
  /** namefix: the story's index; familyfix: "family"; spellingfix: null
   *  (clusters can span stories). */
  story_id: number | string | null;
  /** The group's stable identity, minted by namefix — all verdict POSTs key on this. */
  gid: string;
  /** The session's queue is stale (transcript changed since the namefix run) —
   *  verdicts are refused server-side until namefix re-runs. */
  stale?: boolean;
  occurrences: NameCorrectionOccurrence[];
  /** Spot-answered fold only: the recorded verdicts covering this card's spots. */
  covering?: CoveringVerdict[];
  /** Live cards only: how many of this card's spots were already ruled (folded
   *  out of the occurrence list — the strict rule shows just the un-ruled spots). */
  n_spot_answered_occ?: number;
}

export interface NameCorrectionItem {
  heard: string;
  heard_cleaned: string;
  canonical: string;
  method: string;
  /** Calibration label: WHY the machine believed this (sounds-like / audio-only /
   *  ordinary-word / inconsistent-spelling). Shadow mode: nothing is ever applied. */
  label: string;
  /** Which fixer asked (namefix / family / spelling). */
  fixer?: string;
  /** Spelling questions only: the invented-name cluster this question rules on. */
  cluster_id?: string | null;
  /** Folded items only: which standing ruling folded it — 'no' | 'yes' |
   *  'pick-unify' | 'pick-leave' — so the lift button knows what to lift. */
  folded_by?: string | null;
  dm_share?: boolean;
  /** ⚠ the suggested name already appears correctly spelled in the same sentence. */
  echo_in_sentence?: boolean;
  /** How many OTHER spellings in this world sound-alike into the same suggestion —
   *  0 with several occurrences = "spelled the same every time" (smells invented). */
  n_sibling_spellings?: number;
  /** Frequency asymmetry (the Marley lesson): how often the heard word vs the
   *  suggested name appear in the transcript(s). Garbles are usually the minority
   *  form — a suspect outnumbering its suggestion may be its own name. Logged for
   *  calibration; never acted on by code. */
  n_heard_in_transcript?: number;
  n_suggested_in_transcript?: number;
  /** how many of this card's spots the dictionary has since rewritten in place
   *  — so a heard count of 0 isn't a bare contradiction of the card's own premise. */
  n_heard_dictionary_corrected?: number;
  sessions: NameCorrectionSessionEntry[];
}

export interface NameCorrectionsWorld {
  world: string;
  names: NameCorrectionItem[];
}

export interface NameCorrectionsLabelSection {
  label: string;
  worlds: NameCorrectionsWorld[];
}

/** One recorded verdict (the reviewer's own history) — undoable via unrule. */
export interface RuledVerdict {
  session_id: string;
  verdict: 'yes' | 'no';
  gid: string;
  heard: string;
  heard_cleaned: string;
  canonical: string;
  world: string;
  label: string;
  partial?: boolean;
  n_occurrences: number;
  at: string;
  undoable: boolean;
}

/** Did family-name-fix run? — per session: ran ("checked"), recorded a skip
 *  or failed run with a reason ("skipped"), its record is for an older transcript
 *  ("stale"), or it was never stamped at all — no queue file or no family entry
 *  ("unknown"). "checked, none found" and "never ran" must be visibly different. */
export interface FamilyFixerStatus {
  checked: number;
  skipped: number;
  stale: number;
  unknown: number;
  /** LIVE family cards on non-stale sessions, deduped the way the page merges
   *  them (world + heard→canonical) — the number a reviewer can count below. */
  questions: number;
  /** Distinct skip reasons seen (machine tokens, e.g. "no_roster"). */
  skip_reasons: string[];
}

export interface NameCorrectionsRollup {
  /** Live questions awaiting a verdict (the labeled sections). */
  n_pending_groups: number;
  /** Covered by a standing Name Book yes — will apply at the apply pass. */
  n_covered_by_yes: number;
  /** Covered by a standing Name Book no — quiet (never re-asked) but viewable. */
  n_suppressed_groups: number;
  /** Regenerated questions whose every spot was already ruled without a standing
   *  rule (spot-by-spot answers, world-less picks) — the third fold. */
  n_spot_answered: number;
  n_ruled: number;
  labels: NameCorrectionsLabelSection[];
  covered_by_yes: NameCorrectionsWorld[];
  suppressed: NameCorrectionsWorld[];
  spot_answered: NameCorrectionsWorld[];
  /** Every recorded verdict, newest first. */
  ruled: RuledVerdict[];
  /** Sessions whose queue is stale (transcript changed since the namefix run). */
  stale_sessions: string[];
  /** Whether the family side ran at all — the "say what didn't happen" strip. */
  family_fixer: FamilyFixerStatus;
}

// --- Name pass (the by-ear gold-key builder, #4) ---

/** One occurrence of a name candidate in the transcript. `window` is the audio clip
 *  [start,end] for the Hear button; `m9_coded` marks a hand-coded M9 segment. */
export interface NamePassOccurrence {
  occ_id: string;
  seg_id: SegmentId;
  wi: number;
  start: number | null;
  context: string;
  cap: boolean;
  m9_coded: boolean;
  window: [number, number] | null;
}

/** One name candidate the model-free sweep surfaced, as a per-spelling card.
 *  `source` records which net(s) caught it (m9a/m9b/cap-sweep/lowercase-sound/phrase). */
export interface NamePassCard {
  cleaned: string;
  group: string;
  surfaces: string[];
  detectors: string[];
  source: string[];
  watch: boolean;
  phrase: boolean;
  cluster: string | null;
  primary: string;
  occurrences: NamePassOccurrence[];
}

/** One ruled item in the key: category + the correct spelling by ear. */
export interface NameTruthItem {
  category: string;
  true_spelling: string;
  notes: string;
  /** How true_spelling got here: '' | 'ear-cold' | 'suggestion-confirmed' | 'ear-corrected'. */
  provenance?: string;
}

/** Opus's draft spelling for a candidate — machine scaffolding, rendered inline; the ear
 *  confirms or corrects it into the key. `confidence`: 'high' | 'medium' | 'low'. */
export interface NamePassSuggestion {
  suggested: string;
  confidence: string;
  note: string;
  /** The draft's name-kind. Optional — older sidecars predate it. A draft that carries one
   *  lands its category when accepted, so an accepted draft is COUNTABLE: a key item with a
   *  spelling but no category is `unjudged` and the scorer drops it. Scaffolding
   *  like the spelling, never the key — his ear still rules, and can change it. */
  category?: string;
}

/** One per-spot row in the key — the ruling that only this occurrence carries.
 *  `scope` says WHICH JOB it records (emp/src/name_truth.py OCC_SCOPES):
 *    'name'   the true NAME here — a name-fixer must produce it.
 *    'speech' the true SPEECH here — a hallucination, the wrong person, or words Whisper
 *             dropped ("Krishna," -> Jai Sri Krishna). Not a name-fixer's job.
 *    'tail'   NO decision of its own — this word is part of the name that starts before it
 *             ("you" + "dished" + "there." = Yudhishthira). It joins that name's span, so
 *             the fixer only passes if the WHOLE span reads the name (#5).
 *    ''       unstamped — the scorer infers and prints the inference. */
export interface NameTruthOccurrence {
  true_name: string;
  scope?: string;
  kind?: string;
}

/** The existing key, served as a cross-check seed (the prior pass — assumed incomplete). */
export interface NamePassSeed {
  items: Record<string, NameTruthItem>;
  occurrences: Record<string, NameTruthOccurrence>;
  watch: string[];
  /** null when the prior key was never stamped to a transcript version. */
  fingerprint: string | null;
  marked_at: string | null;
  /** UTC ISO when the ear-gate ("Mark heard") was last stamped; null if never. */
  heard_confirmed: string | null;
}

export interface NamePassPayload {
  session_id: string;
  cards: NamePassCard[];
  /** occ_id -> card key (cleaned), for the transcript highlighter. */
  candidate_index: Record<string, string>;
  /** cleaned -> Opus draft spelling (may be empty when no suggestion pass has run). */
  suggestions: Record<string, NamePassSuggestion>;
  m9_meta: { coded: SegmentId[]; uncovered: SegmentId[] };
  fingerprint: string;
  /** [value, label] pairs — the canonical category vocabulary. */
  categories: [string, string][];
  seed: NamePassSeed;
}

/** Field-routed save into the key — one field at a time (never clobbers a sibling). */
export interface NameTruthFieldSave {
  kind: 'item' | 'occ';
  id: string;
  field: string;
  value: string;
}

// --- Roster (who exists — the /roster page) ---

export interface RosterPerson {
  id: string;
  canonical: string;
  role?: string;
  /** Who this person is to the family. Real people beyond the household live in the
   *  same roster; every matcher treats them identically. */
  circle?: 'family' | 'friend';
  /** Legitimate name forms — correct as heard, matchable, correction targets of
   *  their own (never silently promoted to the canonical). */
  nicknames?: string[];
}

export interface RosterAlias {
  /** The CLEANED distortion to catch exactly (lowercased, punctuation stripped). */
  token: string;
  person_id: string;
}

export interface Roster {
  people: RosterPerson[];
  aliases: RosterAlias[];
}

// --- Home canon (the family's own characters) ---

export interface HomeCharacter {
  id: string;
  canonical: string;
}

export interface HomeCanon {
  characters: HomeCharacter[];
}
