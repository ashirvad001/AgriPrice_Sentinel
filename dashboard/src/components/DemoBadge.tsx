"use client";
/**
 * components/DemoBadge.tsx
 * ────────────────────────
 * Floating badge shown when demo mode is active.
 * Includes a small "Exit" button to disable demo mode.
 */

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { X, FlaskConical } from "lucide-react";
import { isDemoMode, disableDemoMode } from "@/lib/demo-mode";

export default function DemoBadge() {
  const router = useRouter();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    setVisible(isDemoMode());
  }, []);

  if (!visible) return null;

  const handleExit = () => {
    disableDemoMode();
    setVisible(false);
    router.push("/");
  };

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 z-[100] animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center gap-2 pl-3 pr-2 py-2 rounded-full bg-amber-500 text-white shadow-lg shadow-amber-500/30 text-xs font-bold">
        <FlaskConical className="w-3.5 h-3.5" />
        <span>Demo Mode Active</span>
        <button
          onClick={handleExit}
          className="ml-1 p-1 rounded-full hover:bg-amber-600 transition-colors"
          title="Exit demo mode"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}
