/**
 * Chip selector for bet amounts
 */
"use client";

import { motion } from "framer-motion";

interface ChipSelectorProps {
  currentBet: number;
  balance: number;
  onSelect: (amount: number) => void;
  disabled: boolean;
}

const CHIP_VALUES = [25, 50, 100, 250, 500, 1000];

const CHIP_COLORS: Record<number, { bg: string; border: string; text: string }> = {
  25: { bg: "bg-red-600", border: "border-red-400", text: "text-white" },
  50: { bg: "bg-blue-600", border: "border-blue-400", text: "text-white" },
  100: { bg: "bg-gray-700", border: "border-gray-400", text: "text-white" },
  250: { bg: "bg-green-600", border: "border-green-400", text: "text-white" },
  500: { bg: "bg-violet-600", border: "border-violet-400", text: "text-white" },
  1000: { bg: "bg-yellow-500", border: "border-yellow-300", text: "text-gray-900" },
};

export function ChipSelector({
  currentBet,
  balance,
  onSelect,
  disabled,
}: ChipSelectorProps) {
  return (
    <div className="flex flex-col items-center gap-3">
      <div className="text-xs text-white/50 uppercase tracking-widest">Select Chip</div>

      <div className="flex flex-wrap justify-center gap-2">
        {CHIP_VALUES.map((val) => {
          const colors = CHIP_COLORS[val];
          const isSelected = currentBet === val;
          const isTooExpensive = val > balance;

          return (
            <motion.button
              key={val}
              onClick={() => onSelect(val)}
              disabled={disabled || isTooExpensive}
              whileHover={!disabled && !isTooExpensive ? { scale: 1.15, y: -3 } : {}}
              whileTap={!disabled && !isTooExpensive ? { scale: 0.9 } : {}}
              className={`
                relative w-12 h-12 sm:w-14 sm:h-14 rounded-full
                ${colors.bg} ${colors.border} ${colors.text}
                border-4 font-bold text-xs
                flex items-center justify-center
                shadow-lg cursor-pointer
                disabled:opacity-30 disabled:cursor-not-allowed
                transition-all duration-150
                ${isSelected ? "ring-2 ring-white ring-offset-2 ring-offset-transparent scale-110" : ""}
              `}
            >
              {/* Casino chip inner ring */}
              <div className="absolute inset-1 rounded-full border border-white/30" />
              <span className="z-10 leading-none">
                {val >= 1000 ? "1K" : val}
              </span>
            </motion.button>
          );
        })}
      </div>

      {/* Current bet display */}
      <div className="flex items-center gap-2">
        <span className="text-white/50 text-sm">Bet:</span>
        <motion.span
          key={currentBet}
          initial={{ scale: 1.3, color: "#a78bfa" }}
          animate={{ scale: 1, color: "#ffffff" }}
          className="text-white font-bold text-lg"
        >
          ${currentBet.toLocaleString()}
        </motion.span>
      </div>
    </div>
  );
}
