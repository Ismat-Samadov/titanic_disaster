/**
 * Root page — renders the Baccarat game table
 */
import { BaccaratTable } from "@/app/components/game/BaccaratTable";

export default function HomePage() {
  return (
    <main className="min-h-dvh flex flex-col items-center justify-start sm:justify-center py-4">
      {/* Ambient background glow effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-emerald-600/8 rounded-full blur-3xl" />
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] rounded-full blur-3xl"
          style={{ background: "rgba(79, 70, 229, 0.06)" }}
        />
      </div>

      {/* Game */}
      <div className="relative z-10 w-full">
        <BaccaratTable />
      </div>
    </main>
  );
}
