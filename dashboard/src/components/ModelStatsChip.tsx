// src/components/ModelStatsChip.tsx — Model performance badge
"use client";

import { Brain } from "lucide-react";

interface ModelStatsChipProps {
  rmse?: number;
  mae?: number;
  mape?: number;
  accuracy?: number;
}

export default function ModelStatsChip({
  rmse = 87,
  mae,
  mape,
  accuracy = 73,
}: ModelStatsChipProps) {
  return (
    <div className="inline-flex flex-wrap items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/50 text-[11px] font-medium text-slate-400">
      <span className="flex items-center gap-1 text-slate-300">
        <Brain className="w-3.5 h-3.5 text-emerald-500" />
        BiLSTM+Attention
      </span>
      <span className="w-px h-3 bg-slate-700" />
      <span>RMSE <span className="text-white font-semibold">₹{rmse}</span></span>
      {mae != null && (
        <>
          <span className="w-px h-3 bg-slate-700" />
          <span>MAE <span className="text-white font-semibold">₹{mae}</span></span>
        </>
      )}
      {mape != null && (
        <>
          <span className="w-px h-3 bg-slate-700" />
          <span>MAPE <span className="text-white font-semibold">{mape.toFixed(1)}%</span></span>
        </>
      )}
      <span className="w-px h-3 bg-slate-700" />
      <span className="text-emerald-400">{accuracy}% directional accuracy</span>
    </div>
  );
}
