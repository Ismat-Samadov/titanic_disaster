/**
 * PlayingCard component
 * Renders a single playing card with flip animation
 */
"use client";

import { motion } from "framer-motion";
import { Card } from "@/app/lib/baccarat";

interface PlayingCardProps {
  card: Card;
  delay?: number;
  faceDown?: boolean;
  size?: "sm" | "md" | "lg";
}

const SIZE_CLASSES = {
  sm: "w-12 h-16 text-xs",
  md: "w-16 h-24 text-sm",
  lg: "w-20 h-28 text-base",
};

const RED_SUITS = ["♥", "♦"];

export function PlayingCard({
  card,
  delay = 0,
  faceDown = false,
  size = "md",
}: PlayingCardProps) {
  const isRed = RED_SUITS.includes(card.suit);
  const sizeClass = SIZE_CLASSES[size];

  return (
    <motion.div
      className={`${sizeClass} relative rounded-lg cursor-default select-none`}
      initial={{ scale: 0, rotate: -10, opacity: 0, y: -30 }}
      animate={{ scale: 1, rotate: 0, opacity: 1, y: 0 }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 20,
        delay,
      }}
      style={{ perspective: 600 }}
    >
      {faceDown ? (
        /* Card back */
        <div
          className={`${sizeClass} rounded-lg border-2 border-violet-400/60 bg-gradient-to-br from-violet-900 to-indigo-900 flex items-center justify-center`}
        >
          <div className="w-full h-full rounded-md border border-violet-500/30 m-1 bg-[repeating-linear-gradient(45deg,rgba(139,92,246,0.1)_0px,rgba(139,92,246,0.1)_2px,transparent_2px,transparent_8px)]" />
        </div>
      ) : (
        /* Card face */
        <div
          className={`${sizeClass} rounded-lg border border-white/20 bg-gradient-to-br from-white/95 to-white/85 shadow-[0_0_15px_rgba(139,92,246,0.3)] flex flex-col justify-between p-1`}
        >
          {/* Top-left rank + suit */}
          <div className={`flex flex-col leading-none ${isRed ? "text-red-500" : "text-gray-900"}`}>
            <span className="font-bold">{card.rank}</span>
            <span className="text-[0.6em]">{card.suit}</span>
          </div>

          {/* Center suit */}
          <div
            className={`text-center font-bold ${
              size === "sm" ? "text-lg" : size === "md" ? "text-2xl" : "text-3xl"
            } ${isRed ? "text-red-500" : "text-gray-900"}`}
          >
            {card.suit}
          </div>

          {/* Bottom-right rank + suit (rotated) */}
          <div
            className={`flex flex-col leading-none rotate-180 self-end ${isRed ? "text-red-500" : "text-gray-900"}`}
          >
            <span className="font-bold">{card.rank}</span>
            <span className="text-[0.6em]">{card.suit}</span>
          </div>
        </div>
      )}
    </motion.div>
  );
}
