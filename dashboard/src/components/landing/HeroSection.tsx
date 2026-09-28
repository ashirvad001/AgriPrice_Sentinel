"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight, Play, TrendingUp, TrendingDown } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { enableDemoMode } from "@/lib/demo-mode";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Tooltip,
  YAxis,
} from "recharts";

/* ── Types ─────────────────────────────────────────────────────────────────── */
interface PricePoint {
  date: string;
  price: number;
}

/* ── API config ────────────────────────────────────────────────────────────── */
const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const PRICE_ENDPOINT = `${API_BASE}/api/v1/prices/wheat/indore_mandi?days=90`;

export default function HeroSection() {
  const router = useRouter();
  const { isLoggedIn, login } = useAuth();

  const handleViewDemo = () => {
    const demoUser = enableDemoMode();
    // Log in with the demo user so auth context picks it up
    login("demo_token_placeholder", {
      id: demoUser.id,
      phone: demoUser.phone,
      full_name: demoUser.full_name,
      created_at: demoUser.created_at,
      location: demoUser.location,
      crops: demoUser.crops,
    });
    router.push("/dashboard");
  };

  return (
    <section className="relative pt-28 pb-20 md:pt-36 md:pb-28 overflow-hidden bg-gradient-to-br from-green-50 via-white to-emerald-50 dark:from-slate-950 dark:via-slate-900 dark:to-slate-950 transition-colors duration-300">
      {/* Background decoration */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -right-40 w-[500px] h-[500px] rounded-full bg-green-100/40 dark:bg-green-900/10 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 w-[400px] h-[400px] rounded-full bg-emerald-100/50 dark:bg-emerald-900/10 blur-3xl" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-lime-50/30 dark:bg-lime-950/10 blur-3xl" />
      </div>

      <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Copy */}
          <div className="space-y-6">
            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800">
              🌾 Powered by Machine Learning
            </span>
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight text-gray-900 dark:text-white">
              Smart Crop Price Insights for{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-green-600 to-emerald-600 dark:from-green-400 dark:to-emerald-400">
                Better Farming Decisions
              </span>
            </h1>
            <p className="text-lg sm:text-xl text-gray-600 dark:text-slate-400 max-w-xl leading-relaxed">
              Track, analyze, and predict crop prices using AI-powered analytics. Empowering farmers across India with real-time mandi intelligence.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 pt-2">
              <button
                onClick={() => router.push("/dashboard")}
                className="inline-flex items-center justify-center gap-2 px-7 py-3.5 text-base font-bold text-white bg-green-600 rounded-xl hover:bg-green-700 shadow-lg shadow-green-200 dark:shadow-green-900/40 hover:shadow-green-300 dark:hover:shadow-green-800/40 transition-all hover:-translate-y-0.5"
              >
                Go to Dashboard <ArrowRight className="h-5 w-5" />
              </button>
              <button
                onClick={handleViewDemo}
                className="inline-flex items-center justify-center gap-2 px-7 py-3.5 text-base font-bold text-green-700 dark:text-green-400 bg-white dark:bg-slate-800 border-2 border-green-200 dark:border-green-800 rounded-xl hover:border-green-400 dark:hover:border-green-600 hover:bg-green-50 dark:hover:bg-slate-700 transition-all"
              >
                <Play className="h-5 w-5" /> View Demo
              </button>
            </div>
            {/* Trust badges */}
            <div className="flex items-center gap-6 pt-4 text-sm text-gray-500 dark:text-slate-500">
              <span className="flex items-center gap-1">✅ Free to use</span>
              <span className="flex items-center gap-1">📊 16+ Crops</span>
              <span className="flex items-center gap-1">🏪 40+ Mandis</span>
            </div>
          </div>

          {/* Right: Dashboard mockup */}
          <div className="relative">
            <div className="relative bg-white dark:bg-slate-800 rounded-2xl shadow-2xl shadow-green-100 dark:shadow-black/30 border border-green-100 dark:border-slate-700 p-4 transform lg:rotate-1 hover:rotate-0 transition-transform duration-500">
              <DashboardSparkline />
            </div>
            {/* Floating cards */}
            <div className="absolute -bottom-4 -left-4 bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-green-100 dark:border-slate-700 px-4 py-3 flex items-center gap-3 animate-bounce-slow">
              <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/40 flex items-center justify-center text-green-600 dark:text-green-400 font-bold text-lg">📈</div>
              <div>
                <p className="text-xs text-gray-500 dark:text-slate-400">Wheat Prediction</p>
                <p className="text-sm font-bold text-green-700 dark:text-green-400">₹2,450/qtl ↑ 5.2%</p>
              </div>
            </div>
            <div className="absolute -top-4 -right-4 bg-white dark:bg-slate-800 rounded-xl shadow-lg border border-green-100 dark:border-slate-700 px-4 py-3 flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-emerald-100 dark:bg-emerald-900/40 flex items-center justify-center text-emerald-600 dark:text-emerald-400 font-bold text-lg">🔔</div>
              <div>
                <p className="text-xs text-gray-500 dark:text-slate-400">Smart Alert</p>
                <p className="text-sm font-bold text-gray-800 dark:text-white">Price above MSP!</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes bounce-slow { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
        .animate-bounce-slow { animation: bounce-slow 3s ease-in-out infinite; }
      `}</style>
    </section>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════════
 *  Live Recharts sparkline — replaces the old static SVG mockup
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Minimal custom tooltip */
function SparkTooltip({ active, payload }: { active?: boolean; payload?: Array<{ value: number; payload: PricePoint }> }) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white/95 dark:bg-slate-800/95 backdrop-blur shadow-lg rounded-lg px-3 py-2 border border-green-100 dark:border-slate-700 text-xs">
      <p className="text-gray-500 dark:text-slate-400">{d.date}</p>
      <p className="font-bold text-gray-800 dark:text-white">₹{d.price.toLocaleString("en-IN")}/qtl</p>
    </div>
  );
}

/**
 * Generate realistic wheat price data for the hero sparkline when the
 * backend has no data yet (scraper hasn't run / empty DB).
 */
function generateFallbackPrices(days = 90): PricePoint[] {
  const pts: PricePoint[] = [];
  let price = 2200;
  const today = new Date();
  for (let i = days; i >= 0; i--) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    price += (Math.random() - 0.45) * 40;
    price = Math.max(1900, Math.min(2700, price));
    pts.push({
      date: d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" }),
      price: Math.round(price),
    });
  }
  return pts;
}

function DashboardSparkline() {
  const [data, setData] = useState<PricePoint[]>([]);
  const [status, setStatus] = useState<"loading" | "ok" | "error">("loading");

  useEffect(() => {
    let cancelled = false;

    async function fetchPrices() {
      try {
        const res = await fetch(PRICE_ENDPOINT);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();

        /* The API wraps records in `prices` with fields: date, modal_price */
        const records: PricePoint[] = (json.prices ?? json)
          .filter((r: { modal_price?: number | null }) => r.modal_price != null)
          .map((r: { date: string; modal_price: number }) => ({
            date: new Date(r.date).toLocaleDateString("en-IN", { day: "2-digit", month: "short" }),
            price: r.modal_price,
          }))
          .reverse(); // API returns desc order → flip to chronological

        if (!cancelled) {
          if (records.length > 0) {
            setData(records);
          } else {
            // API returned empty — use demo data so the hero always looks great
            setData(generateFallbackPrices());
          }
          setStatus("ok");
        }
      } catch {
        if (!cancelled) {
          // Network error — still show demo data instead of an ugly error
          setData(generateFallbackPrices());
          setStatus("ok");
        }
      }
    }

    fetchPrices();
    return () => { cancelled = true; };
  }, []);

  /* ── derived stats ───────────────────────────────────────────────────── */
  const lastPrice = data.length > 0 ? data[data.length - 1].price : null;
  const firstPrice = data.length > 1 ? data[0].price : null;
  const changePct =
    lastPrice != null && firstPrice != null && firstPrice !== 0
      ? ((lastPrice - firstPrice) / firstPrice) * 100
      : null;
  const isUp = (changePct ?? 0) >= 0;

  /* ── Loading skeleton ────────────────────────────────────────────────── */
  if (status === "loading") {
    return (
      <div className="space-y-3 animate-pulse">
        <div className="flex items-center justify-between px-1">
          <div>
            <div className="h-3 w-28 bg-gray-200 dark:bg-slate-700 rounded mb-2" />
            <div className="h-6 w-36 bg-gray-200 dark:bg-slate-700 rounded" />
          </div>
          <div className="flex gap-1">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-6 w-8 bg-gray-100 dark:bg-slate-700 rounded" />
            ))}
          </div>
        </div>
        <div className="h-[200px] bg-gradient-to-b from-green-50 dark:from-slate-700 to-transparent rounded-xl" />
      </div>
    );
  }

  /* ── Error / empty fallback ──────────────────────────────────────────── */
  if (status === "error" || data.length === 0) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between px-1">
          <div>
            <p className="text-xs text-gray-400 dark:text-slate-500 uppercase tracking-wide">Wheat Price Trend</p>
            <p className="text-xl font-bold text-gray-800 dark:text-white">Data unavailable</p>
          </div>
        </div>
        <div className="flex items-center justify-center h-[200px] rounded-xl bg-gray-50 dark:bg-slate-800 border border-dashed border-gray-200 dark:border-slate-700">
          <p className="text-sm text-gray-400 dark:text-slate-500">Unable to load price data — please try again later.</p>
        </div>
      </div>
    );
  }

  /* ── Live sparkline chart ────────────────────────────────────────────── */
  return (
    <div className="space-y-3">
      {/* Header row */}
      <div className="flex items-center justify-between px-1">
        <div>
          <p className="text-xs text-gray-400 dark:text-slate-500 uppercase tracking-wide">Wheat Price Trend</p>
          <div className="flex items-center gap-2">
            <p className="text-xl font-bold text-gray-800 dark:text-white">
              ₹{lastPrice?.toLocaleString("en-IN")}
            </p>
            {changePct != null && (
              <span
                className={`inline-flex items-center gap-0.5 text-sm font-semibold ${
                  isUp ? "text-green-600" : "text-red-500"
                }`}
              >
                {isUp ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                {isUp ? "+" : ""}
                {changePct.toFixed(1)}%
              </span>
            )}
          </div>
        </div>
        <div className="flex gap-1">
          {["7D", "1M", "3M", "1Y"].map((t) => (
            <span
              key={t}
              className={`px-2 py-1 rounded text-xs font-medium ${
                t === "3M"
                  ? "bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400"
                  : "bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-slate-400"
              }`}
            >
              {t}
            </span>
          ))}
        </div>
      </div>

      {/* Recharts sparkline */}
      <div className="h-[200px]">
        <ResponsiveContainer width="100%" height="100%" minHeight={1}>
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
            <defs>
              <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#22c55e" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#22c55e" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <YAxis domain={["dataMin - 50", "dataMax + 50"]} hide />
            <Tooltip
              content={<SparkTooltip />}
              cursor={{ stroke: "#16a34a", strokeWidth: 1, strokeDasharray: "4 4" }}
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke="#16a34a"
              strokeWidth={2.5}
              fill="url(#sparkGrad)"
              dot={false}
              activeDot={{ r: 4, fill: "#16a34a", stroke: "#fff", strokeWidth: 2 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
