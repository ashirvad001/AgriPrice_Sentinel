// src/components/ShapChart.tsx — SHAP feature importance with toggle + scroll + hover tooltips
"use client";

import { useState, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import type { ShapFeature } from "@/lib/api";
import EmptyState from "@/components/EmptyState";

interface ShapChartProps {
  features: ShapFeature[] | undefined;
  isLoading: boolean;
}

type FeatureCount = 5 | 10 | 15;

/* ── Custom Tooltip ────────────────────────────────────────────────────────── */
function ShapTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const data = payload[0].payload as ShapFeature;
  const isPositive = data.shap_value >= 0;

  return (
    <div className="bg-slate-800/95 backdrop-blur border border-slate-700 rounded-xl px-4 py-3 shadow-2xl text-xs space-y-1.5 min-w-[220px] pointer-events-none z-50">
      <p className="font-semibold text-white text-[13px] leading-tight">{data.farmer_label}</p>
      <p className="text-[11px] text-slate-500 font-mono">{data.feature_name}</p>
      <div className="h-px bg-slate-700 my-1" />
      <div className="flex items-center justify-between gap-4">
        <span className="text-slate-400">SHAP Impact</span>
        <span className={`font-bold ${isPositive ? "text-emerald-400" : "text-red-400"}`}>
          {isPositive ? "+" : ""}{data.shap_value.toFixed(6)}
        </span>
      </div>
      <div className="flex items-center justify-between gap-4">
        <span className="text-slate-400">Direction</span>
        <span className={`font-semibold flex items-center gap-1 ${isPositive ? "text-emerald-400" : "text-red-400"}`}>
          {isPositive ? "↑ Pushes price up" : "↓ Pushes price down"}
        </span>
      </div>
      <div className="flex items-center justify-between gap-4">
        <span className="text-slate-400">Importance Rank</span>
        <span className="text-slate-300 font-medium">#{data.rank}</span>
      </div>
    </div>
  );
}

/* ── Main Component ────────────────────────────────────────────────────────── */
export default function ShapChart({ features, isLoading }: ShapChartProps) {
  const [count, setCount] = useState<FeatureCount>(10);

  const sorted = useMemo(() => {
    if (!features) return [];
    return [...features]
      .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
      .slice(0, count)
      .reverse();
  }, [features, count]);

  const absMax = useMemo(
    () => Math.max(...sorted.map(f => Math.abs(f.shap_value)), 0.0001),
    [sorted]
  );

  if (isLoading) {
    return (
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-2xl p-6 h-[420px] animate-pulse flex flex-col items-center justify-center gap-3">
        <div className="w-10 h-10 rounded-full border-4 border-slate-700 border-t-emerald-500 animate-spin" />
        <div className="text-slate-500 text-sm">Loading SHAP…</div>
      </div>
    );
  }

  if (!features || features.length === 0) {
    return (
      <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4 sm:p-6">
        <EmptyState
          type="empty"
          title="No SHAP Data"
          message="Feature importance data is not available for this crop."
          suggestion="SHAP values are generated after model training."
        />
      </div>
    );
  }

  // Dynamic height based on count
  const barHeight = 26;
  const chartHeight = Math.max(sorted.length * barHeight + 40, 180);

  return (
    <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4 sm:p-6 flex flex-col h-full">
      {/* Header */}
      <div className="mb-4 shrink-0">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold text-white">🧠 SHAP Feature Importance</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Top {count} factors driving the price prediction
            </p>
          </div>

          {/* Toggle: Top 5 / 10 / 15 */}
          <div className="flex items-center gap-0.5 bg-slate-900/80 p-0.5 rounded-lg border border-slate-700/50 shrink-0">
            {([5, 10, 15] as FeatureCount[]).map((n) => {
              const isActive = count === n;
              return (
                <button
                  key={n}
                  onClick={() => setCount(n)}
                  className={`
                    px-2.5 py-1 rounded-md text-[11px] font-semibold transition-all duration-200
                    ${isActive
                      ? "bg-emerald-600 text-white shadow-sm shadow-emerald-500/25"
                      : "text-slate-500 hover:text-slate-300 hover:bg-slate-800"
                    }
                  `}
                >
                  Top {n}
                </button>
              );
            })}
          </div>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 mt-2.5 text-[11px] text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2.5 rounded-sm bg-emerald-500" /> Pushes price up
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-2.5 rounded-sm bg-red-500" /> Pushes price down
          </span>
        </div>
      </div>

      {/* Scrollable chart container */}
      <div
        className="flex-1 overflow-y-auto overflow-x-hidden min-h-0"
        style={{ maxHeight: "340px" }}
      >
        <div style={{ height: `${chartHeight}px`, minHeight: "100%" }}>
          <ResponsiveContainer width="100%" height="100%" minHeight={1}>
            <BarChart
              data={sorted}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#334155"
                strokeOpacity={0.5}
                horizontal={false}
              />

              <XAxis
                type="number"
                stroke="#64748b"
                fontSize={10}
                tick={{ fill: "#94a3b8" }}
                tickFormatter={(v: number) => v.toFixed(4)}
                domain={["auto", "auto"]}
              />

              <YAxis
                type="category"
                dataKey="farmer_label"
                width={160}
                stroke="#64748b"
                fontSize={11}
                tick={{ fill: "#cbd5e1" }}
                tickLine={false}
              />

              <Tooltip
                content={<ShapTooltip />}
                cursor={{ fill: "rgba(148,163,184,0.08)" }}
                wrapperStyle={{ zIndex: 50 }}
              />

              {/* Zero baseline reference line */}
              <ReferenceLine
                x={0}
                stroke="#64748b"
                strokeWidth={1}
                strokeDasharray="3 3"
              />

              <Bar
                dataKey="shap_value"
                radius={[0, 6, 6, 0]}
                maxBarSize={18}
                isAnimationActive={true}
                animationDuration={500}
                animationEasing="ease-out"
              >
                {sorted.map((entry, index) => {
                  const isPositive = entry.shap_value >= 0;
                  const intensity = 0.45 + 0.55 * (Math.abs(entry.shap_value) / absMax);

                  return (
                    <Cell
                      key={`cell-${index}`}
                      fill={
                        isPositive
                          ? `rgba(22, 163, 74, ${intensity})`
                          : `rgba(239, 68, 68, ${intensity})`
                      }
                      className="transition-all duration-300 hover:brightness-125 cursor-pointer"
                    />
                  );
                })}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Bottom info */}
      <div className="mt-3 pt-3 border-t border-slate-700/50 shrink-0">
        <p className="text-[10px] text-slate-500 text-center">
          Hover over bars for detailed impact info • Sorted by absolute importance
        </p>
      </div>
    </div>
  );
}
