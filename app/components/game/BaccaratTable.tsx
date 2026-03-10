/**
 * Main Baccarat game table — orchestrates all components
 */
"use client";

import { useEffect, useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useBaccarat } from "@/app/hooks/useBaccarat";
import { useSound } from "@/app/hooks/useSound";
import { BetType } from "@/app/lib/baccarat";
import { HandArea } from "./HandArea";
import { StatsBar } from "./StatsBar";
import { ControlBar } from "./ControlBar";
import { DifficultySelector } from "./DifficultySelector";
import { RulesPanel } from "./RulesPanel";
import { BetButton } from "@/app/components/ui/BetButton";
import { ChipSelector } from "@/app/components/ui/ChipSelector";
import { ResultOverlay } from "@/app/components/ui/ResultOverlay";

export function BaccaratTable() {
  const { state, placeBet, setBetAmount, newRound, setDifficulty, resetGame } =
    useBaccarat();
  const { soundEnabled, musicEnabled, toggleSound, toggleMusic } = useSound();
  const [isPaused, setIsPaused] = useState(false);
  const [showResult, setShowResult] = useState(false);

  const isBetting = state.phase === "betting";
  const isResult = state.phase === "result";
  const isGameOver = state.balance < 25 && isResult;

  // Show result overlay after round ends
  useEffect(() => {
    if (isResult) {
      const t = setTimeout(() => setShowResult(true), 600);
      return () => clearTimeout(t);
    } else {
      setShowResult(false);
    }
  }, [isResult]);

  // Keyboard controls
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (isPaused) {
        if (e.key === "Escape" || e.key === " ") setIsPaused(false);
        return;
      }

      switch (e.key.toLowerCase()) {
        case "p":
          if (isBetting) placeBet("player");
          break;
        case "b":
          if (isBetting) placeBet("banker");
          break;
        case "t":
          if (isBetting) placeBet("tie");
          break;
        case "n":
          if (isResult) {
            setShowResult(false);
            setTimeout(newRound, 100);
          }
          break;
        case "escape":
          setIsPaused((v) => !v);
          break;
        case "1":
          setBetAmount(25);
          break;
        case "2":
          setBetAmount(50);
          break;
        case "3":
          setBetAmount(100);
          break;
        case "4":
          setBetAmount(250);
          break;
        case "5":
          setBetAmount(500);
          break;
        case "6":
          setBetAmount(1000);
          break;
      }
    },
    [isBetting, isResult, isPaused, placeBet, newRound, setBetAmount]
  );

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const handleContinue = () => {
    setShowResult(false);
    setTimeout(newRound, 150);
  };

  const handleReset = () => {
    setShowResult(false);
    setTimeout(resetGame, 150);
  };

  const handleBet = (type: BetType) => {
    if (!isBetting || isPaused) return;
    placeBet(type);
  };

  const playerWon =
    state.result === "player" ||
    (state.result === "tie" && state.betType === "tie");
  const bankerWon =
    state.result === "banker" ||
    (state.result === "tie" && state.betType === "tie");
  const showScores = state.phase === "result" || state.phase === "bankerDraw";

  return (
    <div className="relative w-full max-w-2xl mx-auto flex flex-col gap-4 min-h-screen sm:min-h-0 px-4 py-4 sm:py-6">
      {/* Header */}
      <ControlBar
        soundEnabled={soundEnabled}
        musicEnabled={musicEnabled}
        onToggleSound={toggleSound}
        onToggleMusic={toggleMusic}
        isPaused={isPaused}
        onPause={() => setIsPaused((v) => !v)}
      />

      {/* Stats */}
      <StatsBar
        balance={state.balance}
        highScore={state.highScore}
        roundCount={state.roundCount}
        wins={state.wins}
        losses={state.losses}
        ties={state.ties}
      />

      {/* Main Table Area */}
      <motion.div
        className="relative rounded-2xl border border-white/10 bg-gradient-to-b from-green-950/40 to-emerald-950/30 overflow-hidden"
        style={{
          backgroundImage:
            "radial-gradient(ellipse at 50% 50%, rgba(16,185,129,0.08) 0%, transparent 70%)",
        }}
      >
        {/* Table felt texture overlay */}
        <div className="absolute inset-0 opacity-5 pointer-events-none bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAiIGhlaWdodD0iMjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTAgMEwyMCAyME0yMCAwTDAgMjAiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMC41Ii8+PC9zdmc+')]" />

        <div className="relative z-10 p-4 sm:p-6">
          {/* VS divider */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-20">
            <motion.div
              className="w-10 h-10 rounded-full bg-black/60 border border-white/20 flex items-center justify-center text-xs font-black text-white/50"
              animate={
                state.phase === "dealing"
                  ? { rotate: 360 }
                  : {}
              }
              transition={{ duration: 1, ease: "linear", repeat: Infinity }}
            >
              VS
            </motion.div>
          </div>

          {/* Hands */}
          <div className="flex gap-4 sm:gap-8">
            <HandArea
              label="Player"
              cards={state.playerHand}
              score={state.playerScore}
              isWinner={state.result === "player"}
              showScore={showScores}
              side="left"
            />
            <HandArea
              label="Banker"
              cards={state.bankerHand}
              score={state.bankerScore}
              isWinner={state.result === "banker"}
              showScore={showScores}
              side="right"
            />
          </div>

          {/* Game message */}
          <AnimatePresence mode="wait">
            <motion.div
              key={state.message}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="text-center mt-4 text-sm text-white/70 min-h-[20px]"
            >
              {state.message}
            </motion.div>
          </AnimatePresence>

          {/* Natural badge */}
          <AnimatePresence>
            {state.isNatural && isResult && (
              <motion.div
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0, opacity: 0 }}
                className="text-center"
              >
                <span className="inline-block px-3 py-1 rounded-full bg-yellow-500/20 border border-yellow-400/40 text-yellow-300 text-xs font-bold uppercase tracking-wider">
                  ✨ Natural
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Result overlay */}
        <ResultOverlay
          result={state.result}
          betType={state.betType}
          winAmount={state.lastWinAmount}
          balance={state.balance}
          show={showResult}
          onContinue={handleContinue}
          isGameOver={isGameOver}
          onReset={handleReset}
        />

        {/* Pause overlay */}
        <AnimatePresence>
          {isPaused && (
            <motion.div
              className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm rounded-2xl"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="text-center">
                <div className="text-4xl mb-3">⏸</div>
                <div className="text-white font-bold text-xl mb-2">Paused</div>
                <div className="text-white/50 text-sm">
                  Press <kbd className="bg-white/10 px-1.5 py-0.5 rounded text-xs">Esc</kbd> or click to resume
                </div>
                <button
                  onClick={() => setIsPaused(false)}
                  className="mt-4 px-6 py-2 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-bold transition-colors"
                >
                  Resume
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {/* Betting Controls */}
      <motion.div
        className="flex flex-col gap-4"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        {/* Chip selector */}
        <ChipSelector
          currentBet={state.currentBet}
          balance={state.balance}
          onSelect={setBetAmount}
          disabled={!isBetting || isPaused}
        />

        {/* Bet buttons */}
        <div className="grid grid-cols-3 gap-3">
          <BetButton
            type="player"
            selected={state.betType === "player"}
            disabled={!isBetting || isPaused}
            onClick={() => handleBet("player")}
            payout="Pays 1:1"
          />
          <BetButton
            type="tie"
            selected={state.betType === "tie"}
            disabled={!isBetting || isPaused}
            onClick={() => handleBet("tie")}
            payout="Pays 8:1"
          />
          <BetButton
            type="banker"
            selected={state.betType === "banker"}
            disabled={!isBetting || isPaused}
            onClick={() => handleBet("banker")}
            payout="Pays 1:1*"
          />
        </div>

        {/* Difficulty + new round (post-result) */}
        <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
          <DifficultySelector
            current={state.difficulty}
            onChange={setDifficulty}
            disabled={!isBetting}
          />

          {isResult && !showResult && (
            <motion.button
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleContinue}
              className="px-8 py-3 rounded-xl bg-gradient-to-r from-violet-600 to-pink-600 hover:from-violet-500 hover:to-pink-500 text-white font-bold transition-all shadow-lg shadow-violet-500/25"
            >
              Next Round →
            </motion.button>
          )}
        </div>
      </motion.div>

      {/* Rules */}
      <RulesPanel />

      {/* Keyboard hint */}
      <div className="text-center text-[10px] text-white/20 pb-2">
        Keyboard: P=Player · B=Banker · T=Tie · N=Next · 1–6=Chip · Esc=Pause
      </div>
    </div>
  );
}
