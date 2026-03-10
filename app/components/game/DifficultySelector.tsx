/**
 * Difficulty selector component
 */
"use client";

import { motion } from "framer-motion";
import { Difficulty } from "@/app/lib/baccarat";

interface DifficultySelectorProps {
  current: Difficulty;
  onChange: (d: Difficulty) => void;
  disabled: boolean;
}

const DIFFICULTIES: {
  value: Difficulty;
  label: string;
  description: string;
  color: string;
}[] = [
  {
    value: "easy",
    label: "Easy",
    description: "2% banker commission",
    color: "text-emerald-400 border-emerald-400/60 bg-emerald-500/10",
  },
  {
    value: "medium",
    label: "Medium",
    description: "4% banker commission",
    color: "text-yellow-400 border-yellow-400/60 bg-yellow-500/10",
  },
  {
    value: "hard",
    label: "Hard",
    description: "5% banker commission",
    color: "text-rose-400 border-rose-400/60 bg-rose-500/10",
  },
];

export function DifficultySelector({
  current,
  onChange,
  disabled,
}: DifficultySelectorProps) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs text-white/40 uppercase tracking-widest text-center">
        Difficulty
      </span>
      <div className="flex gap-2 justify-center">
        {DIFFICULTIES.map((d) => (
          <motion.button
            key={d.value}
            onClick={() => onChange(d.value)}
            disabled={disabled}
            whileHover={!disabled ? { scale: 1.05 } : {}}
            whileTap={!disabled ? { scale: 0.95 } : {}}
            title={d.description}
            className={`
              px-3 py-1.5 rounded-lg border text-xs font-bold uppercase tracking-wider
              transition-all duration-200 cursor-pointer
              disabled:opacity-40 disabled:cursor-not-allowed
              ${current === d.value ? d.color : "border-white/10 text-white/40 bg-white/5 hover:text-white/60"}
            `}
          >
            {d.label}
          </motion.button>
        ))}
      </div>
    </div>
  );
}
