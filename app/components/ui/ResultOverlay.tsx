/**
 * Animated result overlay shown after each round
 */
"use client";

import { motion, AnimatePresence } from "framer-motion";
import { GameResult, BetType } from "@/app/lib/baccarat";

interface ResultOverlayProps {
  result: GameResult;
  betType: BetType | null;
  winAmount: number;
  balance: number;
  show: boolean;
  onContinue: () => void;
  isGameOver: boolean;
  onReset: () => void;
}

export function ResultOverlay({
  result,
  betType,
  winAmount,
  balance,
  show,
  onContinue,
  isGameOver,
  onReset,
}: ResultOverlayProps) {
  const playerWon =
    (result === "player" && betType === "player") ||
    (result === "banker" && betType === "banker") ||
    (result === "tie" && betType === "tie");

  const isPush = result === "tie" && betType !== "tie";

  const config = isGameOver
    ? {
        title: "GAME OVER",
        subtitle: "You've run out of chips!",
        emoji: "💸",
        colors: "from-gray-800 to-gray-900",
        border: "border-gray-500/50",
        titleColor: "text-gray-300",
      }
    : playerWon
    ? {
        title: winAmount >= 500 ? "BIG WIN!" : "YOU WIN!",
        subtitle: `+$${winAmount.toLocaleString()}`,
        emoji: winAmount >= 1000 ? "🎰" : "💎",
        colors: "from-violet-900/95 to-indigo-900/95",
        border: "border-violet-400/60",
        titleColor: "text-violet-300",
      }
    : isPush
    ? {
        title: "PUSH",
        subtitle: "Tie — bet returned",
        emoji: "🤝",
        colors: "from-emerald-900/95 to-teal-900/95",
        border: "border-emerald-400/60",
        titleColor: "text-emerald-300",
      }
    : {
        title: "YOU LOSE",
        subtitle: result === "tie" ? "Tie (no payout)" : "Better luck next round",
        emoji: "🃏",
        colors: "from-rose-900/95 to-gray-900/95",
        border: "border-rose-400/60",
        titleColor: "text-rose-300",
      };

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          className="absolute inset-0 z-50 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          {/* Backdrop */}
          <motion.div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          />

          {/* Card */}
          <motion.div
            className={`relative z-10 flex flex-col items-center gap-4 p-8 rounded-2xl border-2 bg-gradient-to-br ${config.colors} ${config.border} shadow-2xl max-w-xs w-full mx-4`}
            initial={{ scale: 0.5, rotate: -5, opacity: 0 }}
            animate={{ scale: 1, rotate: 0, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
          >
            {/* Winner particles */}
            {playerWon && !isGameOver && (
              <Particles />
            )}

            <motion.span
              className="text-5xl"
              animate={{ rotate: [0, -10, 10, -5, 5, 0] }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              {config.emoji}
            </motion.span>

            <motion.h2
              className={`text-3xl font-black tracking-wider ${config.titleColor}`}
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.15 }}
            >
              {config.title}
            </motion.h2>

            <motion.p
              className="text-white/80 text-lg font-semibold"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.25 }}
            >
              {config.subtitle}
            </motion.p>

            <motion.p
              className="text-white/50 text-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.35 }}
            >
              Balance: ${balance.toLocaleString()}
            </motion.p>

            {/* Buttons */}
            <motion.div
              className="flex gap-3 mt-2 w-full"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              {isGameOver ? (
                <button
                  onClick={onReset}
                  className="flex-1 py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-bold transition-colors"
                >
                  New Game
                </button>
              ) : (
                <>
                  <button
                    onClick={onContinue}
                    className="flex-1 py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-bold transition-colors"
                  >
                    Continue
                  </button>
                  <button
                    onClick={onReset}
                    className="px-4 py-3 rounded-xl bg-white/10 hover:bg-white/20 text-white/70 font-medium transition-colors text-sm"
                  >
                    Reset
                  </button>
                </>
              )}
            </motion.div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/** Confetti-like particles for win state */
function Particles() {
  const particles = Array.from({ length: 12 }, (_, i) => i);

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-2xl">
      {particles.map((i) => (
        <motion.div
          key={i}
          className="absolute w-2 h-2 rounded-full"
          style={{
            left: `${10 + (i * 7) % 80}%`,
            top: "100%",
            backgroundColor: ["#a78bfa", "#f472b6", "#60a5fa", "#34d399", "#fbbf24"][i % 5],
          }}
          animate={{
            y: [0, -(150 + Math.random() * 100)],
            x: [0, (Math.random() - 0.5) * 60],
            opacity: [1, 0],
            scale: [1, 0.5],
          }}
          transition={{
            duration: 1.5 + Math.random() * 0.5,
            delay: Math.random() * 0.3,
            repeat: Infinity,
            repeatDelay: 0.5,
          }}
        />
      ))}
    </div>
  );
}
