// src/components/SummaryCards.tsx — Four summary metric cards
"use client";

import type { ForecastResponse } from "@/lib/api";

interface SummaryCardsProps {
  data: ForecastResponse | undefined;
  isLoading: boolean;
}

function Card({
  label, value, sub, icon, color,
}: {
  label: string; value: string; sub?: React.ReactNode; icon: string; color: string;
}) {
  return (
    <div className="relative overflow-hidden bg-slate-800/60 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-5 hover:border-slate-600/60 transition-all group">
      <div className={`absolute top-0 right-0 w-24 h-24 rounded-full blur-3xl opacity-10 ${color} -translate-y-6 translate-x-6 group-hover:opacity-20 transition-opacity`} />
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] uppercase tracking-wider text-slate-400 font-medium mb-1">{label}</p>
          <p className="text-2xl sm:text-3xl font-bold text-white tracking-tight">{value}</p>
          {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
        </div>
        <span className="text-2xl opacity-60">{icon}</span>
      </div>
    </div>
  );
}

function Skeleton() {
  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-2xl p-5 animate-pulse">
      <div className="h-3 w-20 bg-slate-700 rounded mb-3" />
      <div className="h-8 w-28 bg-slate-700 rounded mb-2" />
      <div className="h-3 w-32 bg-slate-700/50 rounded" />
    </div>
  );
}

export default function SummaryCards({ data, isLoading }: SummaryCardsProps) {
  if (isLoading || !data) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => <Skeleton key={i} />)}
      </div>
    );
  }

  const ciWidth = data.forecast.length > 0
    ? Math.round(
        data.forecast.reduce((s, f) => s + (f.upper_bound - f.lower_bound), 0) /
          data.forecast.length
      )
    : 0;

  const recColor = data.recommendation === "SELL" ? "text-emerald-400" : "text-amber-400";
  const recIcon = data.recommendation === "SELL" ? "📈" : "⏸️";

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <Card
        label="Current Price"
        value={`₹${data.current_price?.toLocaleString("en-IN") ?? "—"}`}
        sub="Today's mandi price"
        icon="🏪"
        color="bg-blue-500"
      />
      <Card
        label="MSP"
        value={`₹${data.msp?.toLocaleString("en-IN") ?? "N/A"}`}
        sub="Min Support Price"
        icon="🏛️"
        color="bg-amber-500"
      />
      <Card
        label={`Forecast (${data.horizon_days}d avg)`}
        value={`₹${data.avg_predicted_price.toLocaleString("en-IN")}`}
        sub={
          <span className={recColor}>
            {recIcon} {data.recommendation} — {data.recommendation_reason.slice(0, 50)}
          </span>
        }
        icon="🔮"
        color="bg-emerald-500"
      />
      <Card
        label="CI Width (avg)"
        value={`±₹${ciWidth.toLocaleString("en-IN")}`}
        sub="95% confidence interval"
        icon="📊"
        color="bg-purple-500"
      />
    </div>
  );
}
