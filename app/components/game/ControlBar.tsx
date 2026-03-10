/**
 * Top control bar: title, sound toggles, pause
 */
"use client";

import { motion } from "framer-motion";

interface ControlBarProps {
  soundEnabled: boolean;
  musicEnabled: boolean;
  onToggleSound: () => void;
  onToggleMusic: () => void;
  isPaused: boolean;
  onPause: () => void;
}

export function ControlBar({
  soundEnabled,
  musicEnabled,
  onToggleSound,
  onToggleMusic,
  isPaused,
  onPause,
}: ControlBarProps) {
  return (
    <div className="flex items-center justify-between w-full">
      {/* Logo */}
      <motion.div
        className="flex items-center gap-2"
        initial={{ x: -20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
      >
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-pink-500 flex items-center justify-center text-sm font-black">
          B
        </div>
        <div>
          <div className="text-white font-black text-base sm:text-lg tracking-tight leading-none">
            BACCARAT
          </div>
          <div className="text-[10px] text-violet-400 tracking-widest uppercase">
            Royal Edition
          </div>
        </div>
      </motion.div>

      {/* Controls */}
      <motion.div
        className="flex items-center gap-2"
        initial={{ x: 20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
      >
        <IconButton
          title={soundEnabled ? "Mute sounds" : "Enable sounds"}
          onClick={onToggleSound}
          active={soundEnabled}
        >
          {soundEnabled ? "🔊" : "🔇"}
        </IconButton>

        <IconButton
          title={musicEnabled ? "Stop music" : "Play music"}
          onClick={onToggleMusic}
          active={musicEnabled}
        >
          {musicEnabled ? "🎵" : "🎵"}
        </IconButton>

        <IconButton
          title={isPaused ? "Resume" : "Pause"}
          onClick={onPause}
          active={isPaused}
        >
          {isPaused ? "▶" : "⏸"}
        </IconButton>
      </motion.div>
    </div>
  );
}

interface IconButtonProps {
  children: React.ReactNode;
  onClick: () => void;
  title?: string;
  active?: boolean;
}

function IconButton({ children, onClick, title, active }: IconButtonProps) {
  return (
    <motion.button
      onClick={onClick}
      title={title}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.9 }}
      className={`w-9 h-9 rounded-lg border transition-all duration-200 flex items-center justify-center text-base cursor-pointer ${
        active
          ? "border-violet-400/60 bg-violet-500/20 text-white"
          : "border-white/10 bg-white/5 text-white/40 hover:text-white/70"
      }`}
    >
      {children}
    </motion.button>
  );
}
