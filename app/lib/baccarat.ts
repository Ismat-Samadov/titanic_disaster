/**
 * Core Baccarat game logic
 * Implements standard casino baccarat rules
 */

export type Suit = "♠" | "♥" | "♦" | "♣";
export type Rank =
  | "A"
  | "2"
  | "3"
  | "4"
  | "5"
  | "6"
  | "7"
  | "8"
  | "9"
  | "10"
  | "J"
  | "Q"
  | "K";

export interface Card {
  suit: Suit;
  rank: Rank;
  value: number; // Baccarat value (0-9)
  faceDown?: boolean;
}

export type BetType = "player" | "banker" | "tie";
export type GameResult = "player" | "banker" | "tie" | null;
export type GamePhase =
  | "betting"
  | "dealing"
  | "playerDraw"
  | "bankerDraw"
  | "result"
  | "idle";
export type Difficulty = "easy" | "medium" | "hard";

export interface GameState {
  deck: Card[];
  playerHand: Card[];
  bankerHand: Card[];
  playerScore: number;
  bankerScore: number;
  currentBet: number;
  betType: BetType | null;
  balance: number;
  result: GameResult;
  phase: GamePhase;
  message: string;
  roundCount: number;
  wins: number;
  losses: number;
  ties: number;
  highScore: number;
  difficulty: Difficulty;
  isNatural: boolean;
  lastWinAmount: number;
}

const SUITS: Suit[] = ["♠", "♥", "♦", "♣"];
const RANKS: Rank[] = [
  "A","2","3","4","5","6","7","8","9","10","J","Q","K",
];

/** Returns baccarat point value for a card rank */
export function getBaccaratValue(rank: Rank): number {
  if (rank === "A") return 1;
  if (["10", "J", "Q", "K"].includes(rank)) return 0;
  return parseInt(rank, 10);
}

/** Creates a fresh shuffled 8-deck shoe */
export function createShoe(): Card[] {
  const shoe: Card[] = [];
  for (let d = 0; d < 8; d++) {
    for (const suit of SUITS) {
      for (const rank of RANKS) {
        shoe.push({ suit, rank, value: getBaccaratValue(rank) });
      }
    }
  }
  return shuffle(shoe);
}

/** Fisher-Yates shuffle */
export function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/** Calculates hand score (mod 10) */
export function handScore(hand: Card[]): number {
  return hand.reduce((sum, c) => sum + c.value, 0) % 10;
}

/** Determines if a hand is a natural (8 or 9) */
export function isNatural(score: number): boolean {
  return score === 8 || score === 9;
}

/**
 * Determines if player draws a third card.
 * Player draws on 0-5; stands on 6 or 7.
 */
export function playerDraws(playerScore: number): boolean {
  return playerScore <= 5;
}

/**
 * Determines if banker draws a third card.
 * If player did NOT draw, banker draws on 0-5.
 * If player DID draw, complex rules based on player's third card.
 */
export function bankerDraws(
  bankerScore: number,
  playerDrewThird: boolean,
  playerThirdCardValue?: number
): boolean {
  if (!playerDrewThird) {
    // Banker draws on 0-5 when player stands
    return bankerScore <= 5;
  }

  const p = playerThirdCardValue ?? 0;

  if (bankerScore <= 2) return true;
  if (bankerScore === 3) return p !== 8;
  if (bankerScore === 4) return p >= 2 && p <= 7;
  if (bankerScore === 5) return p >= 4 && p <= 7;
  if (bankerScore === 6) return p === 6 || p === 7;
  return false; // bankerScore 7 stands
}

/** Calculates payout multiplier based on result and bet type */
export function calculatePayout(
  result: GameResult,
  betType: BetType,
  difficulty: Difficulty
): number {
  // Commission rate varies by difficulty (higher difficulty = standard commission)
  const bankerCommission = difficulty === "easy" ? 0.02 : difficulty === "medium" ? 0.04 : 0.05;

  if (result === "tie" && betType === "tie") return 8; // 8:1
  if (result === "player" && betType === "player") return 1; // 1:1
  if (result === "banker" && betType === "banker") return 1 - bankerCommission; // 1:1 minus commission
  return -1; // Lose bet
}

/** Full game simulation — returns updated hands and deck */
export function dealRound(deck: Card[]): {
  playerHand: Card[];
  bankerHand: Card[];
  remainingDeck: Card[];
} {
  const remaining = [...deck];
  const deal = () => remaining.shift()!;

  // Standard deal order: P, B, P, B
  const playerHand: Card[] = [deal(), deal()];
  const bankerHand: Card[] = [deal(), deal()];

  return { playerHand, bankerHand, remainingDeck: remaining };
}

/** Initial game state factory */
export function createInitialState(savedHighScore?: number): GameState {
  return {
    deck: createShoe(),
    playerHand: [],
    bankerHand: [],
    playerScore: 0,
    bankerScore: 0,
    currentBet: 100,
    betType: null,
    balance: 5000,
    result: null,
    phase: "betting",
    message: "Place your bet to begin",
    roundCount: 0,
    wins: 0,
    losses: 0,
    ties: 0,
    highScore: savedHighScore ?? 5000,
    difficulty: "medium",
    isNatural: false,
    lastWinAmount: 0,
  };
}

/** Determine game result from scores */
export function determineResult(
  playerScore: number,
  bankerScore: number
): GameResult {
  if (playerScore > bankerScore) return "player";
  if (bankerScore > playerScore) return "banker";
  return "tie";
}

/** AI betting suggestion based on difficulty and history */
export function getAISuggestion(
  difficulty: Difficulty,
  wins: number,
  losses: number,
  ties: number
): { betType: BetType; confidence: string } {
  const total = wins + losses + ties;

  if (difficulty === "easy") {
    // Easy AI: purely random
    const types: BetType[] = ["player", "banker", "tie"];
    return {
      betType: types[Math.floor(Math.random() * 3)],
      confidence: "random",
    };
  }

  if (difficulty === "medium") {
    // Medium AI: slightly favors banker (mathematically correct)
    const rand = Math.random();
    return {
      betType: rand < 0.46 ? "banker" : rand < 0.86 ? "player" : "tie",
      confidence: "statistical",
    };
  }

  // Hard AI: optimal strategy — always bet banker (lowest house edge)
  // With small variance based on recent trends
  if (total > 0) {
    const bankerRate = wins / (total || 1);
    if (bankerRate > 0.55) {
      return { betType: "banker", confidence: "trend-following" };
    }
  }
  return { betType: "banker", confidence: "optimal" };
}
