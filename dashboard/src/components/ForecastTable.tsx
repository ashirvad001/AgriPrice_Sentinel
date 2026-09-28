// src/components/ForecastTable.tsx — Daily forecast data table
"use client";

import type { ForecastResponse } from "@/lib/api";

interface ForecastTableProps {
  data: ForecastResponse | undefined;
  isLoading: boolean;
}

export default function ForecastTable({ data, isLoading }: ForecastTableProps) {
  if (isLoading) {
    return (
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-2xl p-6 animate-pulse">
        <div className="h-5 w-40 bg-slate-700 rounded mb-4" />
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="h-8 bg-slate-700/30 rounded mb-2" />
        ))}
      </div>
    );
  }

  if (!data) return null;

  const msp = data.msp ?? 0;

  return (
    <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4 sm:p-6">
      <h2 className="text-lg font-semibold text-white mb-4">📋 Daily Forecast Values</h2>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700/50">
              <th className="text-left py-3 px-3 text-[10px] uppercase tracking-wider text-slate-400 font-medium">Date</th>
              <th className="text-right py-3 px-3 text-[10px] uppercase tracking-wider text-slate-400 font-medium">Predicted (₹)</th>
              <th className="text-right py-3 px-3 text-[10px] uppercase tracking-wider text-slate-400 font-medium hidden sm:table-cell">Lower (₹)</th>
              <th className="text-right py-3 px-3 text-[10px] uppercase tracking-wider text-slate-400 font-medium hidden sm:table-cell">Upper (₹)</th>
              <th className="text-center py-3 px-3 text-[10px] uppercase tracking-wider text-slate-400 font-medium">vs MSP</th>
            </tr>
          </thead>
          <tbody>
            {data.forecast.map((row, i) => {
              const aboveMsp = row.predicted_price > msp;
              const pct = msp > 0 ? (((row.predicted_price - msp) / msp) * 100).toFixed(1) : "N/A";

              return (
                <tr
                  key={row.date}
                  className={`border-b border-slate-700/20 transition-colors hover:bg-slate-700/20 ${
                    i % 2 === 0 ? "bg-slate-800/20" : ""
                  }`}
                >
                  <td className="py-2.5 px-3 text-slate-300 font-mono text-xs">{row.date}</td>
                  <td className="py-2.5 px-3 text-right text-white font-semibold">
                    ₹{row.predicted_price.toLocaleString("en-IN")}
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-400 hidden sm:table-cell">
                    ₹{row.lower_bound.toLocaleString("en-IN")}
                  </td>
                  <td className="py-2.5 px-3 text-right text-slate-400 hidden sm:table-cell">
                    ₹{row.upper_bound.toLocaleString("en-IN")}
                  </td>
                  <td className="py-2.5 px-3 text-center">
                    <span
                      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold ${
                        aboveMsp
                          ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/20"
                          : "bg-red-500/15 text-red-400 border border-red-500/20"
                      }`}
                    >
                      {aboveMsp ? "▲" : "▼"} {pct}%
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[10px] text-slate-500 mt-3 text-right">
        Showing {data.forecast.length} days • MSP ₹{msp.toLocaleString("en-IN")}
      </p>
    </div>
  );
}
