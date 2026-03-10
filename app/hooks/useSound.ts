/**
 * Sound toggle hook — persists preference to localStorage
 */
"use client";

import { useCallback, useEffect, useState } from "react";
import { startMusic, stopMusic } from "@/app/lib/sound";

const SOUND_KEY = "baccarat_sound";
const MUSIC_KEY = "baccarat_music";

export function useSound() {
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [musicEnabled, setMusicEnabled] = useState(false);

  // Load preferences on mount
  useEffect(() => {
    const s = localStorage.getItem(SOUND_KEY);
    const m = localStorage.getItem(MUSIC_KEY);
    if (s !== null) setSoundEnabled(s === "true");
    if (m !== null) setMusicEnabled(m === "true");
  }, []);

  // Sync music state
  useEffect(() => {
    if (musicEnabled) {
      startMusic();
    } else {
      stopMusic();
    }
  }, [musicEnabled]);

  const toggleSound = useCallback(() => {
    setSoundEnabled((prev) => {
      const next = !prev;
      localStorage.setItem(SOUND_KEY, String(next));
      return next;
    });
  }, []);

  const toggleMusic = useCallback(() => {
    setMusicEnabled((prev) => {
      const next = !prev;
      localStorage.setItem(MUSIC_KEY, String(next));
      return next;
    });
  }, []);

  return { soundEnabled, musicEnabled, toggleSound, toggleMusic };
}
