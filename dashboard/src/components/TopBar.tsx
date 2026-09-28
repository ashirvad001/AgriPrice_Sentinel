// src/components/TopBar.tsx — Dashboard header with crop/mandi selector, freshness, and recommendation
"use client";

import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { CropInfo } from "@/lib/crops";
import { usePriceStream } from "@/lib/useWebSocket";
import { ArrowUpIcon, ArrowDownIcon, Clock } from "lucide-react";
import CropMandiSelector from "@/components/CropMandiSelector";

interface TopBarProps {
  crop: CropInfo;
  mandi: string;
  currentPrice: number | null;
  recommendation: string | null;
  generatedAt?: string | null;
}

/* ── Freshness helpers ─────────────────────────────────────────────────────── */
function getTimeAgo(timestamp: string): string {
  const diff = Date.now() - new Date(timestamp).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function getFreshnessColor(timestamp: string) {
  const hoursDiff = (Date.now() - new Date(timestamp).getTime()) / 3600000;
  if (hoursDiff < 1) return { dot: "bg-emerald-500", text: "text-emerald-400" };
  if (hoursDiff < 6) return { dot: "bg-yellow-400", text: "text-yellow-400" };
  return { dot: "bg-red-500", text: "text-red-400" };
}

/* ── Component ─────────────────────────────────────────────────────────────── */
export default function TopBar({ crop, mandi, currentPrice: initialPrice, recommendation: initialRec, generatedAt }: TopBarProps) {
  const { latestPrice, changePct, recommendation: wsRec, isConnected } = usePriceStream(crop.name, mandi);
  
  const displayPrice = latestPrice ?? initialPrice;
  const displayRec = (wsRec && wsRec !== "HOLD") ? wsRec : (initialRec ?? "HOLD");
  const isSell = displayRec === "SELL";
  const recColor = isSell
    ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
    : "bg-amber-500/20 text-amber-400 border-amber-500/30";

  const freshness = useMemo(() => {
    if (!generatedAt) return null;
    return { timeAgo: getTimeAgo(generatedAt), ...getFreshnessColor(generatedAt) };
  }, [generatedAt]);

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between p-4 sm:p-6 bg-slate-900 border-b border-slate-800 sticky top-0 z-40">
      <div className="flex items-start sm:items-center gap-3 mb-4 sm:mb-0">
        <div className="text-4xl bg-slate-800 p-2 rounded-xl border border-slate-700 relative shrink-0">
          {crop.emoji}
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            {isConnected && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />}
            <span className={`relative inline-flex rounded-full h-3 w-3 ${isConnected ? 'bg-emerald-500' : 'bg-slate-500'}`} />
          </span>
        </div>
        <div className="space-y-1.5">
          {/* Crop + Mandi Selector (TASK 9) */}
          <CropMandiSelector currentCrop={crop.name} currentMandi={mandi} />

          <div className="flex flex-wrap items-center gap-2">
            {/* Live Price with freshness dot */}
            <Badge variant="outline" className={`bg-slate-800 border-slate-700 font-medium px-2.5 py-1 ${changePct > 0 ? 'text-emerald-400 border-emerald-500/30' : changePct < 0 ? 'text-red-400 border-red-500/30' : 'text-slate-300'}`}>
              {freshness && <span className={`w-2 h-2 rounded-full ${freshness.dot} mr-1.5 shrink-0 inline-block`} />}
              Live Price:
              <span className="text-white ml-1 font-semibold inline-flex items-center">
                {displayPrice ? `₹${displayPrice.toFixed(0)}` : "N/A"}
                {changePct > 0 && <ArrowUpIcon className="w-3 h-3 ml-1 text-emerald-400" />}
                {changePct < 0 && <ArrowDownIcon className="w-3 h-3 ml-1 text-red-400" />}
                {changePct !== 0 && <span className="ml-1 text-xs opacity-80">{Math.abs(changePct).toFixed(1)}%</span>}
              </span>
            </Badge>

            {freshness && (
              <span className={`flex items-center gap-1 text-[11px] font-medium ${freshness.text}`}>
                <Clock className="w-3 h-3" /> {freshness.timeAgo}
              </span>
            )}

            <Badge variant="outline" className="bg-slate-800 text-amber-400 border-amber-500/30 font-medium px-2.5 py-1">
              MSP: ₹{crop.msp}
            </Badge>
          </div>
        </div>
      </div>
      
      {displayRec && (
        <div className="flex items-center gap-2">
          <span className="text-sm text-slate-400 font-medium hidden sm:inline-block">AI Recommendation:</span>
          <Badge variant="outline" className={`px-4 py-2 text-sm sm:text-base font-bold uppercase tracking-wider ${recColor}`}>
            {displayRec}
          </Badge>
        </div>
      )}
    </div>
  );
}
