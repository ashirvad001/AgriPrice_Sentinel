"use client";

import { useState, useEffect, useCallback } from "react";
import { useParams } from "next/navigation";
import TopBar from "@/components/TopBar";
import dynamic from "next/dynamic";
const ForecastChart = dynamic(() => import("@/components/ForecastChart"), { ssr: false, loading: () => <div className="bg-slate-800/60 border border-slate-700/50 rounded-2xl p-6 h-[420px] animate-pulse" /> });
const ShapChart = dynamic(() => import("@/components/ShapChart"), { ssr: false, loading: () => <div className="bg-slate-800/60 border border-slate-700/50 rounded-2xl p-6 h-[420px] animate-pulse" /> });
import AlertForm from "@/components/AlertForm";
import { useForecast, usePriceHistory, useShapFeatures } from "@/lib/api";
import { CROPS } from "@/lib/crops";
import { useAuth } from "@/lib/auth-context";

export default function DashboardPage() {
  const params = useParams();
  const rawCrop = params.crop as string;
  const rawMandi = params.mandi as string;
  const { user, isLoggedIn } = useAuth();
  
  // Format casing nicely
  const cropStr = decodeURIComponent(rawCrop).charAt(0).toUpperCase() + decodeURIComponent(rawCrop).slice(1);
  const mandiStr = decodeURIComponent(rawMandi);
  
  const [horizon, setHorizon] = useState(30);
  const [switching, setSwitching] = useState(false);

  const { data: forecast, isLoading: isForecastLoading, isFetching: isForecastFetching } = useForecast(cropStr, mandiStr, horizon);
  const { data: history, isLoading: isHistoryLoading } = usePriceHistory(cropStr, mandiStr, 365);
  const { data: shap, isLoading: isShapLoading } = useShapFeatures(cropStr);

  const cropInfo = CROPS.find((c) => c.name.toLowerCase() === cropStr.toLowerCase()) || {
    name: cropStr,
    emoji: "🌾",
    msp: 2000,
    season: "both" as const,
    sowingMonth: 0,
    harvestMonth: 0,
  };

  // Track horizon switching animation
  useEffect(() => {
    if (!isForecastFetching) {
      const t = setTimeout(() => setSwitching(false), 300);
      return () => clearTimeout(t);
    }
  }, [isForecastFetching]);

  const handleHorizonChange = useCallback((newHorizon: number) => {
    if (newHorizon === horizon) return;
    setSwitching(true);
    setHorizon(newHorizon);
  }, [horizon]);

  const chartLoading = isForecastLoading || isHistoryLoading || switching;

  // Stats derived from forecast
  const avgPrice = forecast?.avg_predicted_price;
  const recommendation = forecast?.recommendation ?? null;
  const recommendationReason = forecast?.recommendation_reason ?? null;

  // Dynamic greeting (TASK 14)
  const greeting = user?.full_name
    ? `Hi, ${user.full_name.split(" ")[0]} 👋`
    : isLoggedIn
      ? "Hi, Farmer 👋"
      : "Welcome 👋";

  return (
    <div className="dark min-h-screen bg-slate-950 text-slate-100 pb-20">
      <TopBar 
        crop={cropInfo}
        mandi={mandiStr}
        currentPrice={forecast?.current_price ?? (history?.prices?.[history.prices.length - 1]?.modal_price || null)}
        recommendation={recommendation}
        generatedAt={forecast?.generated_at ?? null}
      />
      
      <div className="max-w-[1400px] mx-auto px-3 sm:px-6 mt-4 sm:mt-6 space-y-4 sm:space-y-6">
        {/* Greeting + Horizon Toggle + Stats Row */}
        <div className="flex flex-col gap-4">
          {/* Top row: greeting + horizon */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <p className="text-sm text-slate-400">{greeting} — viewing <span className="text-white font-medium">{cropStr}</span> forecast</p>

            {/* Horizon tabs */}
            <div className="flex items-center gap-1 bg-slate-800/60 p-1 rounded-xl border border-slate-700/50">
              {[30, 60, 90].map((d) => {
                const isActive = horizon === d;
                return (
                  <button
                    key={d}
                    onClick={() => handleHorizonChange(d)}
                    disabled={switching}
                    className={`
                      relative px-4 sm:px-5 py-2 rounded-lg text-sm font-semibold transition-all duration-200
                      ${isActive
                        ? "bg-emerald-600 text-white shadow-lg shadow-emerald-500/25"
                        : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                      }
                      ${switching ? "opacity-60 cursor-wait" : "cursor-pointer"}
                    `}
                  >
                    {d}d
                    {isActive && (
                      <span className="absolute -bottom-0.5 left-1/2 -translate-x-1/2 w-4 h-0.5 bg-emerald-400 rounded-full" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Stats chips row */}
          <div className="flex flex-wrap items-center gap-2 sm:gap-3">
            {/* Avg predicted price */}
            <div className={`px-3 sm:px-4 py-2 rounded-xl border transition-all duration-300 ${switching ? "animate-pulse bg-slate-800/60 border-slate-700" : "bg-slate-800/60 border-slate-700/50"}`}>
              <span className="text-[10px] uppercase tracking-wider text-slate-500 block">Avg Predicted</span>
              <span className="text-base sm:text-lg font-bold text-white">
                {switching ? (
                  <span className="inline-block w-20 h-5 bg-slate-700 rounded animate-pulse" />
                ) : avgPrice ? (
                  `₹${avgPrice.toLocaleString("en-IN")}`
                ) : (
                  "—"
                )}
              </span>
            </div>

            {/* Recommendation chip */}
            <div className={`px-3 sm:px-4 py-2 rounded-xl border transition-all duration-300 ${switching ? "animate-pulse bg-slate-800/60 border-slate-700" : recommendation === "SELL" ? "bg-emerald-500/10 border-emerald-500/30" : "bg-amber-500/10 border-amber-500/30"}`}>
              <span className="text-[10px] uppercase tracking-wider text-slate-500 block">AI Says</span>
              {switching ? (
                <span className="inline-block w-14 h-5 bg-slate-700 rounded animate-pulse" />
              ) : (
                <span className={`text-base sm:text-lg font-bold uppercase tracking-wide ${recommendation === "SELL" ? "text-emerald-400" : "text-amber-400"}`}>
                  {recommendation ?? "—"}
                </span>
              )}
            </div>

            {/* Reason */}
            {!switching && recommendationReason && (
              <p className="text-xs text-slate-500 max-w-sm hidden md:block">
                {recommendationReason}
              </p>
            )}
          </div>
        </div>

        {/* Charts Row — stacks on mobile (TASK 15) */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4 sm:gap-6">
          <div className="xl:col-span-2">
            <ForecastChart 
               forecast={forecast}
               history={history}
               crop={cropInfo}
               isLoading={chartLoading}
            />
          </div>
          <div className="xl:col-span-1">
            <ShapChart features={shap} isLoading={isShapLoading} />
          </div>
        </div>
        
        {/* Alerts Row */}
        <div className="pt-6 sm:pt-10">
          <AlertForm />
        </div>
      </div>
    </div>
  );
}
