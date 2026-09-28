// src/components/ForecastChart.tsx — Recharts ComposedChart with CI band + forecast line + MSP
"use client";
import { useMemo } from "react";

import {
  ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";
import type { ForecastResponse, PriceHistoryResponse } from "@/lib/api";
import type { CropInfo } from "@/lib/crops";
import EmptyState from "@/components/EmptyState";
import ModelStatsChip from "@/components/ModelStatsChip";

interface ForecastChartProps {
  forecast: ForecastResponse | undefined;
  history: PriceHistoryResponse | undefined;
  crop: CropInfo;
  isLoading: boolean;
}

interface ChartPoint {
  date: string;
  historical?: number;
  predicted?: number;
  lower?: number;
  upper?: number;
}

const MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function formatDate(dateStr: string): string {
  const d = new Date(dateStr);
  return `${d.getDate()} ${MONTH_NAMES[d.getMonth()]}`;
}

/* ── Custom Tooltip ────────────────────────────────────────────────────────── */
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;

  const dateLabel = formatDate(String(label));
  const historical = payload.find((p: any) => p.dataKey === "historical");
  const predicted = payload.find((p: any) => p.dataKey === "predicted");
  const upper = payload.find((p: any) => p.dataKey === "upper");
  const lower = payload.find((p: any) => p.dataKey === "lower");

  const fmt = (v: number) => `₹${v.toLocaleString("en-IN")}`;

  return (
    <div className="bg-slate-800/95 backdrop-blur border border-slate-700 rounded-xl px-4 py-3 shadow-xl text-xs space-y-1.5 min-w-[180px]">
      <p className="text-slate-400 font-medium text-[11px]">{dateLabel}</p>
      {historical?.value != null && (
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-blue-400" />
            <span className="text-slate-300">Historical</span>
          </span>
          <span className="font-bold text-white">{fmt(historical.value)}</span>
        </div>
      )}
      {predicted?.value != null && (
        <div className="flex items-center justify-between gap-4">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
            <span className="text-slate-300">Forecast</span>
          </span>
          <span className="font-bold text-emerald-400">{fmt(predicted.value)}</span>
        </div>
      )}
      {upper?.value != null && lower?.value != null && (
        <div className="flex items-center justify-between gap-4 pt-1 border-t border-slate-700">
          <span className="flex items-center gap-1.5">
            <span className="w-2.5 h-1 rounded bg-emerald-500/30" />
            <span className="text-slate-400">95% CI</span>
          </span>
          <span className="font-medium text-slate-300">
            {fmt(lower.value)} – {fmt(upper.value)}
          </span>
        </div>
      )}
    </div>
  );
}

/* ── Main Component ────────────────────────────────────────────────────────── */
export default function ForecastChart({ forecast, history, crop, isLoading }: ForecastChartProps) {
  if (isLoading) {
    return (
      <div className="bg-slate-800/60 border border-slate-700/50 rounded-2xl p-6 h-[420px] animate-pulse flex flex-col items-center justify-center gap-3">
        <div className="w-10 h-10 rounded-full border-4 border-slate-700 border-t-emerald-500 animate-spin" />
        <div className="text-slate-500 text-sm">Loading forecast chart…</div>
      </div>
    );
  }

  const data: ChartPoint[] = useMemo(() => {
    const dataMap = new Map<string, ChartPoint>();

    // Historical prices
    if (history?.prices) {
      const recent = history.prices.slice(-90);
      for (const p of recent) {
        if (p.modal_price != null) {
          dataMap.set(p.date, { date: p.date, historical: p.modal_price });
        }
      }
    }

    // Forecast prices
    if (forecast?.forecast && forecast.forecast.length > 0) {
      const histEntries = Array.from(dataMap.values());
      const lastHist = histEntries.length > 0 ? histEntries[histEntries.length - 1] : undefined;

      if (lastHist?.historical !== undefined) {
        const bridgeDate = lastHist.date;
        const existing = dataMap.get(bridgeDate);
        dataMap.set(bridgeDate, {
          ...existing,
          date: bridgeDate,
          historical: lastHist.historical,
          predicted: lastHist.historical,
          upper: lastHist.historical,
          lower: lastHist.historical,
        });
      }

      for (const f of forecast.forecast) {
        const existing = dataMap.get(f.date);
        dataMap.set(f.date, {
          ...existing,
          date: f.date,
          predicted: f.predicted_price,
          lower: f.lower_bound,
          upper: f.upper_bound,
        });
      }
    }

    return Array.from(dataMap.values()).sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );
  }, [history, forecast]);

  // Empty state: no data at all
  if (data.length === 0) {
    return (
      <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4 sm:p-6">
        <EmptyState
          type="empty"
          title={`No ${crop.name} data available`}
          message="No price data found for this crop and mandi combination."
          suggestion="Try selecting a different mandi or check back later."
        />
      </div>
    );
  }

  const mspValue = forecast?.msp ?? crop.msp;

  // Seasonal annotations
  const seasonLabel = crop.season === "rabi" ? "Rabi" : crop.season === "kharif" ? "Kharif" : "Year-round";
  const harvestMonth = MONTH_NAMES[crop.harvestMonth];

  return (
    <div className="bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4 sm:p-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 gap-2">
        <div>
          <h2 className="text-lg font-semibold text-white">
            {crop.emoji} {crop.name} Price Forecast
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Historical (90d) + {forecast?.horizon_days ?? 30}-day forecast • {seasonLabel} crop • Harvest: {harvestMonth}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-blue-400 rounded" /> Historical
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-emerald-400 rounded" /> Forecast
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-5 h-3 rounded bg-emerald-500/20 border border-emerald-500/30" /> 95% CI
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 bg-amber-400 rounded border-dashed" /> MSP
          </span>
        </div>
      </div>

      {/* Model Performance Chip (TASK 13) */}
      <div className="mb-3 overflow-x-auto">
        <ModelStatsChip />
      </div>

      <ResponsiveContainer width="100%" height={350} minHeight={1}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="gradHistorical" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="gradCI" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#10b981" stopOpacity={0.25} />
              <stop offset="100%" stopColor="#10b981" stopOpacity={0.08} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#334155" strokeOpacity={0.5} />

          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            stroke="#64748b"
            fontSize={11}
            interval="preserveStartEnd"
            tick={{ fill: "#94a3b8" }}
          />
          <YAxis
            stroke="#64748b"
            fontSize={11}
            tick={{ fill: "#94a3b8" }}
            tickFormatter={(v: number) => `₹${v.toLocaleString("en-IN")}`}
            domain={["auto", "auto"]}
          />

          <Tooltip content={<ChartTooltip />} />

          {/* MSP reference line */}
          <ReferenceLine
            y={mspValue}
            stroke="#f59e0b"
            strokeDasharray="6 3"
            strokeWidth={1.5}
            label={{
              value: `MSP ₹${mspValue.toLocaleString("en-IN")}`,
              fill: "#f59e0b",
              fontSize: 11,
              position: "insideTopRight",
            }}
          />

          {/* ═══ Confidence Band ═══
               Render upper_bound as a filled area, then lower_bound as a filled
               area with the same background color — this "punches out" the space
               below the lower bound, leaving only the band visible. */}
          <Area
            type="monotone"
            dataKey="upper"
            stroke="none"
            fill="url(#gradCI)"
            fillOpacity={1}
            connectNulls={false}
            isAnimationActive={true}
            animationDuration={800}
            name="upper"
          />
          <Area
            type="monotone"
            dataKey="lower"
            stroke="none"
            fill="#0f172a"
            fillOpacity={0.85}
            connectNulls={false}
            isAnimationActive={true}
            animationDuration={800}
            name="lower"
          />

          {/* Historical area + line */}
          <Area
            type="monotone"
            dataKey="historical"
            stroke="none"
            fill="url(#gradHistorical)"
            connectNulls
            isAnimationActive={true}
            animationDuration={600}
          />
          <Line
            type="monotone"
            dataKey="historical"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={false}
            connectNulls
            isAnimationActive={true}
            animationDuration={600}
          />

          {/* Forecast line — rendered LAST so it's on top */}
          <Line
            type="monotone"
            dataKey="predicted"
            stroke="#10b981"
            strokeWidth={2.5}
            dot={false}
            connectNulls
            isAnimationActive={true}
            animationDuration={800}
            activeDot={{
              r: 5,
              fill: "#10b981",
              stroke: "#fff",
              strokeWidth: 2,
            }}
          />

          <Legend content={() => null} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
