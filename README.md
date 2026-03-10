# Baccarat Royal

A full-featured browser-based Baccarat casino game built with **Next.js**, **TypeScript**, and **Tailwind CSS**.

---

## Features

- **Complete Baccarat rules** — standard casino rules including all drawing conditions, natural wins (8/9), and tie pushes
- **Three difficulty levels** — Easy (2% commission), Medium (4%), Hard (5% standard commission)
- **Procedural sound effects** — Web Audio API-generated sounds; no external files required
- **Ambient background music** — toggle on/off with synthesized casino music
- **Animated cards** — spring physics card dealing animations with Framer Motion
- **Win/lose result overlay** — animated results screen with particle effects on big wins
- **Pause / resume** — full pause support with keyboard shortcut
- **High score persistence** — localStorage saves your best balance across sessions
- **Fully responsive** — works on mobile, tablet, and desktop without horizontal scroll
- **Touch + keyboard controls** — on-screen buttons for mobile, keyboard shortcuts for desktop
- **8-deck shoe** — realistic 416-card shoe that reshuffles when running low
- **Neon glassmorphism theme** — cohesive dark theme with violet/pink neon accents

---

## Gameplay

### Rules

| Concept | Rule |
|---------|------|
| Card values | A=1, 2-9=face value, 10/J/Q/K=0 |
| Score | Sum of cards mod 10 (only last digit counts) |
| Natural | Either hand scores 8 or 9 on first two cards -> instant result |
| Player draws | Player draws a 3rd card if score <= 5 |
| Banker draws | Complex rules based on banker score and player's 3rd card |

### Payouts

| Bet | Payout |
|-----|--------|
| Player wins | 1:1 |
| Banker wins | 1:1 (minus commission) |
| Tie | 8:1 |

Banker commission: Easy=2%, Medium=4%, Hard=5%

---

## Controls

### Keyboard

| Key | Action |
|-----|--------|
| `P` | Bet on Player |
| `B` | Bet on Banker |
| `T` | Bet on Tie |
| `N` | Next Round |
| `Esc` | Pause / Resume |
| `1` | Chip $25 |
| `2` | Chip $50 |
| `3` | Chip $100 |
| `4` | Chip $250 |
| `5` | Chip $500 |
| `6` | Chip $1,000 |

### Mobile / Touch

Tap chip amounts to select bet size, then tap **PLAYER**, **TIE**, or **BANKER** to place your bet. After each round, tap **Continue** or **Next Round**.

---

## Tech Stack

- **Next.js 15** — App Router, server components, Vercel-optimized
- **TypeScript** — strict mode throughout
- **Tailwind CSS 4** — utility-first styling, no inline styles
- **Framer Motion** — spring-physics animations for cards and overlays
- **Web Audio API** — procedural sound effects and music (no external assets)
- **localStorage** — high score persistence

---

## Project Structure

```
app/
  lib/
    baccarat.ts        # Core game logic, rules, payout calculations
    sound.ts           # Web Audio API sound engine
  hooks/
    useBaccarat.ts     # Main game state reducer + async deal sequence
    useSound.ts        # Sound/music toggle with localStorage persistence
  components/
    game/
      BaccaratTable.tsx    # Main game orchestrator
      HandArea.tsx         # Card hand display
      StatsBar.tsx         # Balance, high score, round stats
      ControlBar.tsx       # Sound, music, pause controls
      DifficultySelector.tsx
      RulesPanel.tsx       # Collapsible rules reference
    ui/
      PlayingCard.tsx      # Animated card component
      ScoreDisplay.tsx     # Score badge
      BetButton.tsx        # Player/Banker/Tie bet buttons
      ChipSelector.tsx     # Casino chip bet selector
      ResultOverlay.tsx    # Win/lose animated overlay
  page.tsx             # Root page with ambient background
  layout.tsx           # App metadata, fonts, SVG favicon
  globals.css          # Tailwind base + custom scrollbar
```

---

## Running Locally

### Prerequisites

- Node.js 18+
- npm, yarn, pnpm, or bun

### Install & Run

```bash
# Clone the repo
git clone <your-repo-url>
cd baccarat

# Install dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:3000 in your browser.

### Build for Production

```bash
npm run build
npm start
```

---

## Deploy to Vercel

The easiest way to deploy:

1. Push your code to a GitHub repository
2. Go to vercel.com/new
3. Import your repository
4. Click **Deploy** — no configuration needed

Or use the Vercel CLI:

```bash
npm install -g vercel
vercel
```

---

## License

MIT
