/**
 * Bet selection button (Player / Banker / Tie)
 */
"use client";

import { motion } from "framer-motion";
import { BetType } from "@/app/lib/baccarat";

interface BetButtonProps {
  type: BetType;
  selected: boolean;
  disabled: boolean;
  onClick: () => void;
  payout: string;
}

const CONFIG: Record<BetType, { label: string; colors: string; glow: string }> = {
  player: {
    label: "PLAYER",
    colors:
      "from-blue-600 to-cyan-600 border-blue-400/60 hover:border-blue-300",
    glow: "shadow-[0_0_20px_rgba(59,130,246,0.5)]",
  },
  banker: {
    label: "BANKER",
    colors:
      "from-rose-600 to-pink-600 border-rose-400/60 hover:border-rose-300",
    glow: "shadow-[0_0_20px_rgba(244,63,94,0.5)]",
  },
  tie: {
    label: "TIE",
    colors:
      "from-emerald-600 to-teal-600 border-emerald-400/60 hover:border-emerald-300",
    glow: "shadow-[0_0_20px_rgba(16,185,129,0.5)]",
  },
};

export function BetButton({
  type,
  selected,
  disabled,
  onClick,
  payout,
}: BetButtonProps) {
  const cfg = CONFIG[type];

  return (
    <motion.button
      onClick={onClick}
      disabled={disabled}
      whileHover={!disabled ? { scale: 1.05, y: -2 } : {}}
      whileTap={!disabled ? { scale: 0.95 } : {}}
      className={`
        relative flex flex-col items-center justify-center gap-1
        px-4 py-3 sm:px-6 sm:py-4 rounded-xl border-2 font-bold
        bg-gradient-to-br ${cfg.colors} text-white
        transition-all duration-200 cursor-pointer
        disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100
        ${selected ? `${cfg.glow} scale-105 brightness-110 border-opacity-100` : ""}
      `}
    >
      {/* Selected indicator */}
      {selected && (
        <motion.div
          layoutId="bet-selected"
          className="absolute inset-0 rounded-xl bg-white/10"
          transition={{ type: "spring", stiffness: 400, damping: 30 }}
        />
      )}

      <span className="text-xs sm:text-sm tracking-widest z-10">{cfg.label}</span>
      <span className="text-lg sm:text-2xl font-black z-10">
        {type === "tie" ? "8" : "1"}
        <span className="text-xs font-normal ml-0.5">:1</span>
      </span>
      <span className="text-[10px] sm:text-xs text-white/70 z-10">{payout}</span>
    </motion.button>
  );
}
