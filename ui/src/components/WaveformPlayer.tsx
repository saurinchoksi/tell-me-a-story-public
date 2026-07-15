/**
 * WaveformPlayer — WaveSurfer wrapper with minimap plugin.
 *
 * Manages the WaveSurfer lifecycle: creates on mount and fires a 60fps
 * onTimeUpdate via requestAnimationFrame for smooth active segment tracking.
 */

import { useEffect, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import Minimap from 'wavesurfer.js/dist/plugins/minimap.esm.js';
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.esm.js';
import './WaveformPlayer.css';

export interface WaveformPlayerHandle {
  play(): void;
  pause(): void;
  playPause(): void;
  setTime(seconds: number): void;
  getCurrentTime(): number;
  setPlaybackRate(rate: number): void;
  isPlaying(): boolean;
  highlightRange(start: number, end: number): void;
  clearHighlight(): void;
  /** Play a single word slice, sample-accurately, via Web Audio (see playWord below). */
  playWord(start: number, end: number): void;
  /**
   * Paint a persistent set of marker spans (e.g. missed-speech candidates).
   * Independent of the transient hover/word-preview highlight; survives
   * waveform recreation (re-applied on 'ready'). Pass [] to clear.
   */
  setMarkedRegions(spans: { start: number; end: number }[]): void;
}

type RegionsInstance = ReturnType<typeof RegionsPlugin.create>;
type WsRegion = ReturnType<RegionsInstance['addRegion']>;

interface WaveformPlayerProps {
  audioUrl: string;
  onTimeUpdate: (time: number) => void;
  onReady?: (duration: number) => void;
  onPlayPause?: (playing: boolean) => void;
  playbackRate?: number;
}

function getCSSColor(prop: string, fallback: string): string {
  const val = getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
  return val || fallback;
}

const WaveformPlayer = forwardRef<WaveformPlayerHandle, WaveformPlayerProps>(
  function WaveformPlayer({ audioUrl, onTimeUpdate, onReady, onPlayPause, playbackRate = 1 }, ref) {
    const containerRef = useRef<HTMLDivElement>(null);
    const minimapRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WaveSurfer | null>(null);
    const regionsRef = useRef<ReturnType<typeof RegionsPlugin.create> | null>(null);
    const rafRef = useRef<number>(0);
    const savedTimeRef = useRef(0);
    const savedRateRef = useRef(playbackRate);
    // Web Audio single-word preview (⌥-click a word). We reuse WaveSurfer's
    // already-decoded AudioBuffer (getDecodedData) and play a slice through our
    // own context — an AudioBufferSourceNode's duration arg stops on the exact
    // sample, which the <audio> timeupdate cadence can't do for short words.
    const audioCtxRef = useRef<AudioContext | null>(null);
    const wordSourceRef = useRef<AudioBufferSourceNode | null>(null);

    // Keep callback refs stable so WaveSurfer events always call latest versions
    const onTimeUpdateRef = useRef(onTimeUpdate);
    const onReadyRef = useRef(onReady);
    const onPlayPauseRef = useRef(onPlayPause);
    useEffect(() => { onTimeUpdateRef.current = onTimeUpdate; }, [onTimeUpdate]);
    useEffect(() => { onReadyRef.current = onReady; }, [onReady]);
    useEffect(() => { onPlayPauseRef.current = onPlayPause; }, [onPlayPause]);

    const startRAF = useCallback(() => {
      const tick = () => {
        if (wsRef.current) {
          onTimeUpdateRef.current(wsRef.current.getCurrentTime());
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    }, []);

    const stopRAF = useCallback(() => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = 0;
      }
    }, []);

    // Two kinds of regions share the plugin: ONE transient highlight (hover /
    // word preview) tracked in hoverRegionRef, plus a persistent set of marker
    // spans (missed-speech candidates) tracked separately — so clearing the
    // hover never wipes the markers and vice versa.
    const hoverRegionRef = useRef<WsRegion | null>(null);
    const markedSpansRef = useRef<{ start: number; end: number }[]>([]);
    const markedRegionsRef = useRef<WsRegion[]>([]);

    // Transient highlight — one source of truth for hover + word preview.
    const paintRegion = useCallback((start: number, end: number) => {
      const regions = regionsRef.current;
      if (!regions) return;
      hoverRegionRef.current?.remove();
      // Tiny gaps would render as zero-width; show at least 1px-worth of audio
      const safeEnd = end > start ? end : start + 0.01;
      hoverRegionRef.current = regions.addRegion({
        start,
        end: safeEnd,
        color: 'rgba(191, 162, 101, 0.35)',
        drag: false,
        resize: false,
      });
    }, []);

    const clearRegions = useCallback(() => {
      hoverRegionRef.current?.remove();
      hoverRegionRef.current = null;
    }, []);

    // (Re)paint the persistent marker spans from markedSpansRef.
    const applyMarkedRegions = useCallback(() => {
      const regions = regionsRef.current;
      if (!regions) return;
      for (const r of markedRegionsRef.current) r.remove();
      const color = getCSSColor('--waveform-missed', 'rgba(176, 96, 80, 0.4)');
      markedRegionsRef.current = markedSpansRef.current.map((s) =>
        regions.addRegion({
          start: s.start,
          end: s.end > s.start ? s.end : s.start + 0.01,
          color,
          drag: false,
          resize: false,
        }),
      );
    }, []);

    // Stop an in-flight word preview (new click, session switch, or unmount) and
    // clear its highlight. Idempotent — safe to call when nothing is playing.
    const stopWordPreview = useCallback(() => {
      const src = wordSourceRef.current;
      if (src) {
        src.onended = null;
        try {
          src.stop();
        } catch {
          // already stopped/ended — fine
        }
        src.disconnect();
        wordSourceRef.current = null;
      }
      clearRegions();
    }, [clearRegions]);

    // Create/recreate WaveSurfer when audioUrl changes
    useEffect(() => {
      if (!containerRef.current || !minimapRef.current) return;

      // Save position before destroying
      if (wsRef.current) {
        savedTimeRef.current = wsRef.current.getCurrentTime();
        savedRateRef.current = wsRef.current.getPlaybackRate();
        stopRAF();
        wsRef.current.destroy();
        // Regions died with the instance — drop the stale handles.
        hoverRegionRef.current = null;
        markedRegionsRef.current = [];
      }

      const waveColor = getCSSColor('--waveform-color', '#999');
      const progressColor = getCSSColor('--waveform-progress', '#BFA265');
      const cursorColor = getCSSColor('--waveform-cursor', '#6A3843');

      const regions = RegionsPlugin.create();
      regionsRef.current = regions;

      const ws = WaveSurfer.create({
        container: containerRef.current,
        url: audioUrl,
        waveColor,
        progressColor,
        cursorColor,
        cursorWidth: 2,
        height: 100,
        minPxPerSec: 100,
        hideScrollbar: true,
        normalize: true,
        plugins: [
          Minimap.create({
            container: minimapRef.current,
            height: 36,
            waveColor: getCSSColor('--waveform-minimap', '#ccc'),
            progressColor,
            cursorColor,
            normalize: true,
          }),
          regions,
        ],
      });

      ws.on('ready', () => {
        ws.setPlaybackRate(savedRateRef.current);
        if (savedTimeRef.current > 0) {
          ws.setTime(savedTimeRef.current);
        }
        // Re-apply persistent marker spans (they need a decoded duration to
        // position, and any set before/across a recreate would be lost).
        applyMarkedRegions();
        onReadyRef.current?.(ws.getDuration());
      });

      ws.on('play', () => {
        // A main-transport start silences any in-flight word preview — the
        // symmetric half of playWord() pausing the transport before it speaks.
        stopWordPreview();
        startRAF();
        onPlayPauseRef.current?.(true);
      });

      ws.on('pause', () => {
        stopRAF();
        // Fire one last time update at pause position
        onTimeUpdateRef.current(ws.getCurrentTime());
        onPlayPauseRef.current?.(false);
      });

      ws.on('seeking', () => {
        onTimeUpdateRef.current(ws.getCurrentTime());
      });

      wsRef.current = ws;

      return () => {
        stopWordPreview();
        stopRAF();
        ws.destroy();
        wsRef.current = null;
        regionsRef.current = null;
        hoverRegionRef.current = null;
        markedRegionsRef.current = [];
      };
    }, [audioUrl, startRAF, stopRAF, stopWordPreview, applyMarkedRegions]);

    // Update playback rate without recreating
    useEffect(() => {
      if (wsRef.current) {
        wsRef.current.setPlaybackRate(playbackRate);
        savedRateRef.current = playbackRate;
      }
    }, [playbackRate]);

    // Release the Web Audio context on final unmount (reused across sessions until then)
    useEffect(() => {
      return () => {
        stopWordPreview();
        void audioCtxRef.current?.close();
        audioCtxRef.current = null;
      };
    }, [stopWordPreview]);

    useImperativeHandle(ref, () => ({
      play() { wsRef.current?.play(); },
      pause() { wsRef.current?.pause(); },
      playPause() { wsRef.current?.playPause(); },
      setTime(seconds: number) {
        if (wsRef.current) {
          wsRef.current.setTime(seconds);
          onTimeUpdateRef.current(seconds);
        }
      },
      getCurrentTime() { return wsRef.current?.getCurrentTime() ?? 0; },
      setPlaybackRate(rate: number) { wsRef.current?.setPlaybackRate(rate); },
      isPlaying() { return wsRef.current?.isPlaying() ?? false; },
      highlightRange: paintRegion,
      clearHighlight: clearRegions,
      setMarkedRegions(spans: { start: number; end: number }[]) {
        markedSpansRef.current = spans;
        applyMarkedRegions();
      },
      playWord(start: number, end: number) {
        const ws = wsRef.current;
        if (!ws) return;
        const buf = ws.getDecodedData();
        if (!buf) {
          // Buffer not decoded yet — fall back to a plain seek + play
          ws.setTime(start);
          ws.play();
          return;
        }
        // Stop the main transport and any prior preview so nothing overlaps
        if (ws.isPlaying()) ws.pause();
        stopWordPreview();

        let ctx = audioCtxRef.current;
        if (!ctx) {
          ctx = new AudioContext();
          audioCtxRef.current = ctx;
        }
        // Created/resumed inside the click gesture → satisfies autoplay policy
        if (ctx.state === 'suspended') void ctx.resume();

        // ~30ms pad each side so onset/coda isn't clipped, clamped to the buffer
        const PAD = 0.03;
        const from = Math.max(0, start - PAD);
        const to = Math.min(buf.duration, end + PAD);
        const dur = Math.max(0, to - from);

        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(ctx.destination);
        src.onended = () => {
          if (wordSourceRef.current === src) stopWordPreview();
        };
        wordSourceRef.current = src;
        paintRegion(start, end);
        src.start(0, from, dur); // the duration arg is the sample-accurate stop
      },
    }));

    return (
      <div className="waveform-player">
        <div ref={minimapRef} className="waveform-minimap" />
        <div ref={containerRef} className="waveform-main" />
      </div>
    );
  },
);

export default WaveformPlayer;
