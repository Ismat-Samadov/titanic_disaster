/**
 * Score badge for player/banker hand totals
 */
"use client";

import { motion, AnimatePresence } from "framer-motion";

interface ScoreDisplayProps {
  score: number;
  label: string;
  isWinner?: boolean;
  show?: boolean;
}

export function ScoreDisplay({
  score,
  label,
  isWinner = false,
  show = true,
}: ScoreDisplayProps) {
  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key={score}
          initial={{ scale: 0.5, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.5, opacity: 0 }}
          className={`flex flex-col items-center gap-1`}
        >
          <span className="text-xs text-white/60 uppercase tracking-widest font-medium">
            {label}
          </span>
          <motion.div
            animate={
              isWinner
                ? {
                    boxShadow: [
                      "0 0 0 0 rgba(167,139,250,0)",
                      "0 0 0 8px rgba(167,139,250,0.4)",
                      "0 0 0 0 rgba(167,139,250,0)",
                    ],
                  }
                : {}
            }
            transition={{ duration: 1, repeat: Infinity }}
            className={`w-12 h-12 rounded-full flex items-center justify-center text-2xl font-bold border-2 transition-all duration-300 ${
              isWinner
                ? "border-violet-400 bg-violet-500/30 text-violet-200"
                : "border-white/20 bg-white/10 text-white"
            }`}
          >
            {score}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
