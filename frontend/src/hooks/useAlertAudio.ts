import { useRef, useCallback } from 'react';

export function useAlertAudio() {
  const audioCtxRef = useRef<AudioContext | null>(null);

  const playAlarmSound = useCallback((enabled: boolean = true) => {
    if (!enabled) return;

    try {
      if (!audioCtxRef.current) {
        const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        audioCtxRef.current = new AudioContextClass();
      }

      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        ctx.resume();
      }

      // Security Intrusion 2-tone alarm chime (880Hz -> 660Hz)
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, now);
      osc.frequency.setValueAtTime(660, now + 0.12);
      osc.frequency.setValueAtTime(990, now + 0.24);

      gain.gain.setValueAtTime(0.25, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 0.45);
    } catch (e) {
      console.warn('Audio playback not permitted yet (user interaction required):', e);
    }
  }, []);

  return { playAlarmSound };
}
