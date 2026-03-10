/**
 * Renders a hand (Player or Banker) with cards and score
 */
"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Card } from "@/app/lib/baccarat";
import { PlayingCard } from "@/app/components/ui/PlayingCard";
import { ScoreDisplay } from "@/app/components/ui/ScoreDisplay";

interface HandAreaProps {
  label: string;
  cards: Card[];
  score: number;
  isWinner: boolean;
  showScore: boolean;
  side: "left" | "right";
}

export function HandArea({
  label,
  cards,
  score,
  isWinner,
  showScore,
  side,
}: HandAreaProps) {
  return (
    <div
      className={`flex flex-col items-center gap-3 flex-1 ${
        side === "left" ? "items-start sm:items-center" : "items-end sm:items-center"
      }`}
    >
      {/* Label */}
      <motion.div
        className={`text-sm font-bold uppercase tracking-widest px-3 py-1 rounded-full border ${
          isWinner && showScore
            ? "border-violet-400/70 bg-violet-500/20 text-violet-300"
            : "border-white/20 bg-white/5 text-white/60"
        } transition-all duration-300`}
      >
        {label}
        {isWinner && showScore && (
          <motion.span
            initial={{ opacity: 0, x: -5 }}
            animate={{ opacity: 1, x: 0 }}
            className="ml-1"
          >
            ★
          </motion.span>
        )}
      </motion.div>

      {/* Cards */}
      <div className="flex gap-1.5 sm:gap-2 flex-wrap justify-center min-h-[96px]">
        <AnimatePresence>
          {cards.map((card, i) => (
            <PlayingCard
              key={`${card.rank}-${card.suit}-${i}`}
              card={card}
              delay={i * 0.15}
              size="md"
            />
          ))}
        </AnimatePresence>

        {/* Empty placeholders */}
        {cards.length === 0 &&
          [0, 1].map((i) => (
            <div
              key={i}
              className="w-16 h-24 rounded-lg border-2 border-dashed border-white/10 bg-white/5"
            />
          ))}
      </div>

      {/* Score */}
      <ScoreDisplay
        score={score}
        label="Points"
        isWinner={isWinner && showScore}
        show={showScore && cards.length > 0}
      />
    </div>
  );
}
