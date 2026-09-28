// src/components/EmptyState.tsx — Reusable empty/error state component
"use client";

import { SearchX, WifiOff, RefreshCw } from "lucide-react";

interface EmptyStateProps {
  type?: "empty" | "error";
  title?: string;
  message?: string;
  suggestion?: string;
  onRetry?: () => void;
}

export default function EmptyState({
  type = "empty",
  title,
  message,
  suggestion,
  onRetry,
}: EmptyStateProps) {
  const isError = type === "error";
  const Icon = isError ? WifiOff : SearchX;

  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 ${isError ? "bg-red-500/10" : "bg-slate-800"}`}>
        <Icon className={`w-8 h-8 ${isError ? "text-red-400" : "text-slate-500"}`} />
      </div>
      <h3 className="text-lg font-semibold text-white mb-1">
        {title ?? (isError ? "Failed to Load Data" : "No Data Found")}
      </h3>
      <p className="text-sm text-slate-400 max-w-xs mb-1">
        {message ?? (isError
          ? "Something went wrong while fetching data. Please try again."
          : "No price data found for this crop and mandi combination."
        )}
      </p>
      {suggestion && (
        <p className="text-xs text-slate-500 mb-4">💡 {suggestion}</p>
      )}
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 rounded-lg transition-all mt-2"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Try again
        </button>
      )}
    </div>
  );
}
