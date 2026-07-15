import { useRef, useState, useEffect, useCallback } from 'react';
import { formatTime } from '../utils/time';
import './AudioPlayer.css';

interface AudioPlayerProps {
  src: string;
  seekTo?: number;
  onTimeUpdate?: (currentTime: number) => void;
  playbackRate?: number;
  /**
   * When true, an external seek also starts playback (a clicked line should
   * *play* from there, not just move the playhead). Defaults to false so
   * existing callers — the speaker-review sample preview — keep their
   * seek-only behavior.
   */
  playOnSeek?: boolean;
}

export default function AudioPlayer({
  src,
  seekTo,
  onTimeUpdate,
  playbackRate = 1,
  playOnSeek = false,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // External seek control.
  // Note: if parent needs to re-seek to the same timestamp, it should
  // change the value slightly (e.g. seekTo + 0.001) or switch to a
  // callback pattern — React skips effects when deps are unchanged.
  useEffect(() => {
    if (seekTo !== undefined && audioRef.current) {
      audioRef.current.currentTime = seekTo;
      if (playOnSeek) {
        // User-initiated (click), so autoplay policy permits it; swallow the
        // rejection promise so a blocked play never surfaces as an error.
        // `playing` state follows the element's own play/pause events (below),
        // so a late resolution here can never flip the label out of sync.
        audioRef.current.play().catch(() => {});
      }
    }
  }, [seekTo, playOnSeek]);

  // Playback rate
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  const handleTimeUpdate = useCallback(() => {
    if (!audioRef.current) return;
    const t = audioRef.current.currentTime;
    setCurrentTime(t);
    onTimeUpdate?.(t);
  }, [onTimeUpdate]);

  const handleLoadedMetadata = useCallback(() => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  }, []);

  // Branch on the element's truth, not React state: `playing` mirrors the
  // element via its play/pause events (below) and is display-only, so it can
  // never desync the actual toggle behavior.
  const togglePlay = useCallback(() => {
    if (!audioRef.current) return;
    if (audioRef.current.paused) {
      audioRef.current.play().catch(() => {});
    } else {
      audioRef.current.pause();
    }
  }, []);

  const handleScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return;
    const t = parseFloat(e.target.value);
    audioRef.current.currentTime = t;
    setCurrentTime(t);
  }, []);

  return (
    <div className="audio-player">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        preload="metadata"
      />
      <button className="audio-player-btn" onClick={togglePlay}>
        {playing ? 'Pause' : 'Play'}
      </button>
      <span className="audio-player-time">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
      <input
        className="audio-player-scrub"
        type="range"
        min={0}
        max={duration || 0}
        step={0.1}
        value={currentTime}
        onChange={handleScrub}
      />
    </div>
  );
}

