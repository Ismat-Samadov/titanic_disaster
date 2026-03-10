/**
 * Top stats bar: balance, high score, round count, win/loss record
 */
"use client";

import { motion } from "framer-motion";

interface StatsBarProps {
  balance: number;
  highScore: number;
  roundCount: number;
  wins: number;
  losses: number;
  ties: number;
}

export function StatsBar({
  balance,
  highScore,
  roundCount,
  wins,
  losses,
  ties,
}: StatsBarProps) {
  const winRate =
    roundCount > 0 ? Math.round((wins / roundCount) * 100) : 0;

  return (
    <div className="w-full grid grid-cols-2 sm:grid-cols-4 gap-2">
      {/* Balance */}
      <StatCard
        label="Balance"
        value={`$${balance.toLocaleString()}`}
        highlight={balance > 5000}
        danger={balance < 500}
      />

      {/* High Score */}
      <StatCard
        label="Best"
        value={`$${highScore.toLocaleString()}`}
        icon="🏆"
      />

      {/* Round */}
      <StatCard label="Round" value={`#${roundCount}`} />

      {/* Win Rate */}
      <StatCard
        label="Record"
        value={`${wins}W / ${losses}L${ties > 0 ? ` / ${ties}T` : ""}`}
        sub={roundCount > 0 ? `${winRate}% win rate` : ""}
      />
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string;
  sub?: string;
  icon?: string;
  highlight?: boolean;
  danger?: boolean;
}

function StatCard({ label, value, sub, icon, highlight, danger }: StatCardProps) {
  return (
    <motion.div
      className={`rounded-xl border px-3 py-2 flex flex-col gap-0.5 transition-all duration-300 ${
        highlight
          ? "border-emerald-400/40 bg-emerald-500/10"
          : danger
          ? "border-red-400/40 bg-red-500/10"
          : "border-white/10 bg-white/5"
      }`}
      layout
    >
      <span className="text-[10px] sm:text-xs text-white/40 uppercase tracking-widest">
        {label}
      </span>
      <motion.span
        key={value}
        initial={{ y: -5, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className={`text-sm sm:text-base font-bold ${
          highlight
            ? "text-emerald-400"
            : danger
            ? "text-red-400"
            : "text-white"
        }`}
      >
        {icon && <span className="mr-1">{icon}</span>}
        {value}
      </motion.span>
      {sub && (
        <span className="text-[10px] text-white/30">{sub}</span>
      )}
    </motion.div>
  );
}
