/**
 * Main game state hook for Baccarat
 * Manages full game lifecycle, animations timing, and localStorage persistence
 */
"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";
import {
  GameState,
  BetType,
  Difficulty,
  createInitialState,
  createShoe,
  dealRound,
  handScore,
  isNatural,
  playerDraws,
  bankerDraws,
  determineResult,
  calculatePayout,
} from "@/app/lib/baccarat";
import {
  playDeal,
  playWin,
  playLose,
  playTie,
  playNatural,
  playChip,
  playClick,
  playFlip,
  resumeAudio,
} from "@/app/lib/sound";

const HIGH_SCORE_KEY = "baccarat_high_score";
const MIN_DECK_SIZE = 104; // Reshuffle when fewer than ~2 decks remain

type Action =
  | { type: "SET_BET_TYPE"; betType: BetType }
  | { type: "SET_BET_AMOUNT"; amount: number }
  | { type: "DEAL_START" }
  | { type: "DEAL_CARDS"; state: Partial<GameState> }
  | { type: "PLAYER_DRAW"; state: Partial<GameState> }
  | { type: "BANKER_DRAW"; state: Partial<GameState> }
  | { type: "SET_RESULT"; state: Partial<GameState> }
  | { type: "NEW_ROUND" }
  | { type: "SET_DIFFICULTY"; difficulty: Difficulty }
  | { type: "RESET" };

function reducer(state: GameState, action: Action): GameState {
  switch (action.type) {
    case "SET_BET_TYPE":
      return { ...state, betType: action.betType };
    case "SET_BET_AMOUNT":
      return { ...state, currentBet: action.amount };
    case "DEAL_START":
      return { ...state, phase: "dealing", message: "Dealing cards…" };
    case "DEAL_CARDS":
    case "PLAYER_DRAW":
    case "BANKER_DRAW":
    case "SET_RESULT":
      return { ...state, ...action.state };
    case "NEW_ROUND":
      return {
        ...state,
        playerHand: [],
        bankerHand: [],
        playerScore: 0,
        bankerScore: 0,
        betType: null,
        result: null,
        phase: "betting",
        message: "Place your bet to begin",
        isNatural: false,
        lastWinAmount: 0,
        // Reshuffle if deck running low
        deck: state.deck.length < MIN_DECK_SIZE ? createShoe() : state.deck,
      };
    case "SET_DIFFICULTY":
      return { ...state, difficulty: action.difficulty };
    case "RESET": {
      const saved = parseInt(
        localStorage.getItem(HIGH_SCORE_KEY) ?? "5000",
        10
      );
      return createInitialState(saved);
    }
    default:
      return state;
  }
}

export function useBaccarat() {
  const [state, dispatch] = useReducer(reducer, undefined, () => {
    // Read high score on init (safe for SSR — will be undefined on server)
    if (typeof window !== "undefined") {
      const saved = parseInt(
        localStorage.getItem(HIGH_SCORE_KEY) ?? "5000",
        10
      );
      return createInitialState(saved);
    }
    return createInitialState();
  });

  const isAnimating = useRef(false);

  // Persist high score to localStorage whenever it changes
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(HIGH_SCORE_KEY, state.highScore.toString());
    }
  }, [state.highScore]);

  /** Place a bet and start the round */
  const placeBet = useCallback(
    async (betType: BetType) => {
      if (isAnimating.current) return;
      if (state.phase !== "betting") return;
      if (state.currentBet > state.balance) return;

      resumeAudio();
      isAnimating.current = true;

      dispatch({ type: "SET_BET_TYPE", betType });
      dispatch({ type: "DEAL_START" });

      // Small pause before dealing
      await delay(300);

      // Deal initial 4 cards
      const { playerHand, bankerHand, remainingDeck } = dealRound(state.deck);
      playDeal();

      const pScore = handScore(playerHand);
      const bScore = handScore(bankerHand);
      const natural = isNatural(pScore) || isNatural(bScore);

      dispatch({
        type: "DEAL_CARDS",
        state: {
          playerHand,
          bankerHand,
          playerScore: pScore,
          bankerScore: bScore,
          deck: remainingDeck,
          phase: natural ? "result" : "playerDraw",
          isNatural: natural,
          message: natural
            ? `Natural ${pScore > bScore ? pScore : bScore}!`
            : "Evaluating hands…",
        },
      });

      if (natural) {
        playNatural();
        await delay(800);
        await resolveRound(pScore, bScore, playerHand, bankerHand, betType, remainingDeck, false, undefined, natural);
        return;
      }

      await delay(600);

      // Player third card
      let finalPlayerHand = [...playerHand];
      let finalDeck = [...remainingDeck];
      let playerDrewThird = false;
      let playerThirdValue: number | undefined;

      if (playerDraws(pScore)) {
        playFlip();
        const thirdCard = finalDeck.shift()!;
        finalPlayerHand = [...finalPlayerHand, thirdCard];
        playerDrewThird = true;
        playerThirdValue = thirdCard.value;

        const newPScore = handScore(finalPlayerHand);
        dispatch({
          type: "PLAYER_DRAW",
          state: {
            playerHand: finalPlayerHand,
            playerScore: newPScore,
            deck: finalDeck,
            phase: "bankerDraw",
            message: `Player draws ${thirdCard.rank}${thirdCard.suit}`,
          },
        });

        await delay(600);
      }

      // Banker third card
      let finalBankerHand = [...bankerHand];
      const currentBScore = handScore(bankerHand);

      if (bankerDraws(currentBScore, playerDrewThird, playerThirdValue)) {
        playFlip();
        const thirdCard = finalDeck.shift()!;
        finalBankerHand = [...finalBankerHand, thirdCard];

        const newBScore = handScore(finalBankerHand);
        dispatch({
          type: "BANKER_DRAW",
          state: {
            bankerHand: finalBankerHand,
            bankerScore: newBScore,
            deck: finalDeck,
            phase: "result",
            message: `Banker draws ${thirdCard.rank}${thirdCard.suit}`,
          },
        });

        await delay(600);
      }

      const finalPScore = handScore(finalPlayerHand);
      const finalBScore = handScore(finalBankerHand);

      await resolveRound(
        finalPScore,
        finalBScore,
        finalPlayerHand,
        finalBankerHand,
        betType,
        finalDeck,
        playerDrewThird,
        playerThirdValue,
        false
      );
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [state]
  );

  /** Resolve round results and update balance/stats */
  const resolveRound = async (
    pScore: number,
    bScore: number,
    playerHand: GameState["playerHand"],
    bankerHand: GameState["bankerHand"],
    betType: BetType,
    deck: GameState["deck"],
    _playerDrewThird: boolean,
    _playerThirdValue: number | undefined,
    natural: boolean
  ) => {
    const result = determineResult(pScore, bScore);
    const payoutMult = calculatePayout(result, betType, state.difficulty);
    const bet = state.currentBet;

    // payoutMult: positive = win, -1 = lose
    const winAmount =
      payoutMult > 0 ? Math.floor(bet * payoutMult) : 0;
    const netChange = payoutMult > 0 ? winAmount : -bet;
    const newBalance = state.balance + netChange;

    let message = "";
    if (result === "tie" && betType === "tie") {
      message = `Tie! You win $${winAmount}!`;
      playTie();
    } else if (
      (result === "player" && betType === "player") ||
      (result === "banker" && betType === "banker")
    ) {
      message = natural
        ? `Natural! You win $${winAmount}!`
        : `You win $${winAmount}!`;
      playWin();
    } else {
      message =
        result === "tie"
          ? "Tie — push (no payout on this bet)"
          : `${result === "player" ? "Player" : "Banker"} wins. You lose $${bet}.`;
      if (result !== "tie") playLose();
      else playTie();
    }

    const won =
      (result === "player" && betType === "player") ||
      (result === "banker" && betType === "banker") ||
      (result === "tie" && betType === "tie");

    const tied = result === "tie" && betType !== "tie";

    dispatch({
      type: "SET_RESULT",
      state: {
        result,
        phase: "result",
        playerHand,
        bankerHand,
        playerScore: pScore,
        bankerScore: bScore,
        deck,
        balance: newBalance,
        highScore: Math.max(newBalance, state.highScore),
        message,
        roundCount: state.roundCount + 1,
        wins: state.wins + (won ? 1 : 0),
        losses: state.losses + (!won && !tied ? 1 : 0),
        ties: state.ties + (tied ? 1 : 0),
        lastWinAmount: won ? winAmount : 0,
      },
    });

    isAnimating.current = false;
  };

  /** Set bet amount */
  const setBetAmount = useCallback((amount: number) => {
    resumeAudio();
    playChip();
    dispatch({ type: "SET_BET_AMOUNT", amount });
  }, []);

  /** Start a new round */
  const newRound = useCallback(() => {
    if (isAnimating.current) return;
    resumeAudio();
    playClick();
    dispatch({ type: "NEW_ROUND" });
  }, []);

  /** Set difficulty level */
  const setDifficulty = useCallback((difficulty: Difficulty) => {
    playClick();
    dispatch({ type: "SET_DIFFICULTY", difficulty });
  }, []);

  /** Full game reset */
  const resetGame = useCallback(() => {
    isAnimating.current = false;
    dispatch({ type: "RESET" });
  }, []);

  return {
    state,
    placeBet,
    setBetAmount,
    newRound,
    setDifficulty,
    resetGame,
  };
}

/** Simple delay helper */
function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
