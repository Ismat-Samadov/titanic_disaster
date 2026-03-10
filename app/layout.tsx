import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Baccarat Royal — Casino Card Game",
  description:
    "Play classic casino baccarat in your browser. Bet on Player, Banker, or Tie. Multiple difficulty levels, procedural sound effects, and full keyboard controls.",
  keywords: ["baccarat", "casino", "card game", "browser game"],
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  themeColor: "#0d0d1a",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* SVG favicon — purple B on dark background */}
        <link
          rel="icon"
          href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='8' fill='%230d0d1a'/><rect width='32' height='32' rx='8' fill='url(%23g)' opacity='0.6'/><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%237c3aed'/><stop offset='1' stop-color='%23db2777'/></linearGradient></defs><text x='16' y='23' text-anchor='middle' font-family='Arial Black,sans-serif' font-weight='900' font-size='20' fill='white'>B</text></svg>"
        />
      </head>
      <body
        className="antialiased min-h-dvh"
        style={{
          fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          background: "linear-gradient(135deg, #0d0d1a 0%, #0f0f2e 50%, #0d1a0d 100%)",
        }}
      >
        {children}
      </body>
    </html>
  );
}
