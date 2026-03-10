/**
 * Collapsible rules/info panel
 */
"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

export function RulesPanel() {
  const [open, setOpen] = useState(false);

  return (
    <div className="w-full">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full text-center text-xs text-white/30 hover:text-white/60 transition-colors py-2 flex items-center justify-center gap-2"
      >
        <span>{open ? "▲" : "▼"}</span>
        How to Play
        <span>{open ? "▲" : "▼"}</span>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-xs text-white/60 space-y-3">
              <Section title="Objective">
                Bet on whether the Player or Banker hand will score closest to 9,
                or if they will Tie.
              </Section>

              <Section title="Card Values">
                Aces = 1 · Cards 2–9 = face value · 10, J, Q, K = 0
                <br />
                Only the last digit counts (e.g. 15 = 5)
              </Section>

              <Section title="Drawing Rules">
                <strong className="text-white/70">Player:</strong> Draws on 0–5, stands on 6–7
                <br />
                <strong className="text-white/70">Banker:</strong> Complex rules based on both hands
              </Section>

              <Section title="Payouts">
                Player wins: 1:1 · Banker wins: 1:1 (minus commission) · Tie: 8:1
              </Section>

              <Section title="Controls">
                Select a chip amount → click Player, Banker, or Tie to bet
                <br />
                Keyboard: <kbd className="bg-white/10 px-1 rounded">P</kbd> = Player ·{" "}
                <kbd className="bg-white/10 px-1 rounded">B</kbd> = Banker ·{" "}
                <kbd className="bg-white/10 px-1 rounded">T</kbd> = Tie ·{" "}
                <kbd className="bg-white/10 px-1 rounded">N</kbd> = New Round
              </Section>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-white/50 font-bold uppercase text-[10px] tracking-wider mb-1">
        {title}
      </div>
      <div className="leading-relaxed">{children}</div>
    </div>
  );
}
