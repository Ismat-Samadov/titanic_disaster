/**
 * Sound effect utilities using Web Audio API
 * Generates procedural sounds — no external audio files needed
 */

let audioCtx: AudioContext | null = null;
let musicNode: AudioBufferSourceNode | null = null;
let musicGain: GainNode | null = null;
let isMusicPlaying = false;

function getCtx(): AudioContext {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
  }
  return audioCtx;
}

/** Plays a short beep-like tone */
function playTone(
  frequency: number,
  duration: number,
  type: OscillatorType = "sine",
  gainVal = 0.3,
  delay = 0
) {
  try {
    const ctx = getCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.type = type;
    osc.frequency.setValueAtTime(frequency, ctx.currentTime + delay);
    gain.gain.setValueAtTime(gainVal, ctx.currentTime + delay);
    gain.gain.exponentialRampToValueAtTime(
      0.001,
      ctx.currentTime + delay + duration
    );

    osc.start(ctx.currentTime + delay);
    osc.stop(ctx.currentTime + delay + duration);
  } catch {
    // Silently fail if audio unavailable
  }
}

/** Card deal sound — quick tick */
export function playDeal() {
  playTone(800, 0.08, "square", 0.15);
  playTone(600, 0.06, "square", 0.1, 0.05);
}

/** Chip click sound */
export function playChip() {
  playTone(1200, 0.05, "triangle", 0.2);
}

/** Win fanfare */
export function playWin() {
  [0, 0.1, 0.2, 0.35].forEach((delay, i) => {
    const notes = [523, 659, 784, 1047];
    playTone(notes[i], 0.25, "sine", 0.3, delay);
  });
}

/** Loss sound — descending tones */
export function playLose() {
  [0, 0.15, 0.3].forEach((delay, i) => {
    const notes = [400, 320, 240];
    playTone(notes[i], 0.2, "sawtooth", 0.2, delay);
  });
}

/** Tie sound — neutral ping */
export function playTie() {
  playTone(660, 0.3, "sine", 0.25);
  playTone(660, 0.3, "sine", 0.1, 0.15);
}

/** Natural win (8 or 9) — special fanfare */
export function playNatural() {
  [0, 0.08, 0.16, 0.24, 0.32, 0.4].forEach((delay, i) => {
    const notes = [523, 659, 784, 880, 1047, 1175];
    playTone(notes[i], 0.3, "sine", 0.35, delay);
  });
}

/** Card flip sound */
export function playFlip() {
  playTone(440, 0.1, "triangle", 0.15);
}

/** Button click */
export function playClick() {
  playTone(900, 0.05, "square", 0.1);
}

/** Generates looping ambient casino background music */
function generateMusicBuffer(ctx: AudioContext): AudioBuffer {
  const sampleRate = ctx.sampleRate;
  const duration = 8; // 8-second loop
  const buffer = ctx.createBuffer(2, sampleRate * duration, sampleRate);

  // Simple pentatonic melody pattern (C major pentatonic)
  const notes = [261.63, 293.66, 329.63, 392.0, 440.0, 523.25];
  const pattern = [0, 2, 4, 3, 1, 4, 2, 0, 3, 5, 2, 4];

  for (let channel = 0; channel < 2; channel++) {
    const data = buffer.getChannelData(channel);
    const beatDuration = (sampleRate * duration) / pattern.length;

    pattern.forEach((noteIndex, beatIdx) => {
      const freq = notes[noteIndex] * (channel === 1 ? 0.5 : 1);
      const start = Math.floor(beatIdx * beatDuration);
      const end = Math.floor(start + beatDuration * 0.7);

      for (let i = start; i < end && i < data.length; i++) {
        const t = (i - start) / sampleRate;
        const envelope = Math.exp(-t * 3);
        data[i] += Math.sin(2 * Math.PI * freq * t) * 0.04 * envelope;
        // Add harmonics for richer sound
        data[i] += Math.sin(2 * Math.PI * freq * 2 * t) * 0.02 * envelope;
      }
    });
  }

  return buffer;
}

/** Starts looping background music */
export function startMusic() {
  try {
    const ctx = getCtx();
    if (isMusicPlaying) return;

    const buffer = generateMusicBuffer(ctx);
    musicGain = ctx.createGain();
    musicGain.gain.setValueAtTime(0.25, ctx.currentTime);
    musicGain.connect(ctx.destination);

    const playLoop = () => {
      if (!isMusicPlaying) return;
      musicNode = ctx.createBufferSource();
      musicNode.buffer = buffer;
      musicNode.connect(musicGain!);
      musicNode.onended = playLoop;
      musicNode.start();
    };

    isMusicPlaying = true;
    playLoop();
  } catch {
    // Silently fail
  }
}

/** Stops background music */
export function stopMusic() {
  try {
    isMusicPlaying = false;
    if (musicNode) {
      musicNode.onended = null;
      musicNode.stop();
      musicNode = null;
    }
    if (musicGain) {
      musicGain.disconnect();
      musicGain = null;
    }
  } catch {
    // Silently fail
  }
}

/** Resume audio context if suspended (required by browser policy) */
export function resumeAudio() {
  try {
    if (audioCtx?.state === "suspended") {
      audioCtx.resume();
    }
  } catch {
    // Silently fail
  }
}
