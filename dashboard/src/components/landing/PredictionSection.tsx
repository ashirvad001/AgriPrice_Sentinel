"use client";
import { useState } from "react";

const crops = ["Wheat", "Rice", "Maize", "Gram", "Cotton"];
const locations = ["Azadpur Mandi", "Indore Mandi", "Ludhiana Mandi", "Lasalgaon Mandi"];

// Simulated prediction data
const predData: Record<string, { actual: number[]; predicted: number[]; months: string[] }> = {
  Wheat: { actual: [2100,2180,2250,2200,2320,2280], predicted: [2280,2350,2420,2390,2460,2510], months: ["Apr","May","Jun","Jul","Aug","Sep"] },
  Rice: { actual: [2400,2350,2420,2500,2480,2550], predicted: [2550,2600,2580,2650,2700,2720], months: ["Apr","May","Jun","Jul","Aug","Sep"] },
  Maize: { actual: [1800,1850,1900,1880,1950,1920], predicted: [1920,1980,2050,2020,2080,2100], months: ["Apr","May","Jun","Jul","Aug","Sep"] },
  Gram: { actual: [5200,5300,5250,5400,5350,5450], predicted: [5450,5520,5600,5580,5650,5700], months: ["Apr","May","Jun","Jul","Aug","Sep"] },
  Cotton: { actual: [6800,6900,6850,7000,6950,7050], predicted: [7050,7120,7200,7180,7250,7300], months: ["Apr","May","Jun","Jul","Aug","Sep"] },
};

export default function PredictionSection() {
  const [crop, setCrop] = useState("Wheat");
  const [location, setLocation] = useState(locations[0]);
  const data = predData[crop] || predData.Wheat;

  return (
    <section id="predictions" className="py-20 md:py-28 bg-white dark:bg-slate-950 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-2xl mx-auto mb-14">
          <span className="inline-block px-3 py-1 text-xs font-semibold bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-400 rounded-full mb-4">🧠 ML Predictions</span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-4">AI-Powered Price Forecasting</h2>
          <p className="text-gray-600 dark:text-slate-400 text-lg">See how our BiLSTM + Attention model predicts crop prices up to 90 days ahead with 95% confidence intervals.</p>
        </div>

        <div className="grid lg:grid-cols-5 gap-8">
          {/* Input panel */}
          <div className="lg:col-span-2 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-slate-800 dark:to-slate-900 rounded-2xl border border-green-100 dark:border-slate-700 p-6 space-y-5">
            <h3 className="font-bold text-gray-900 dark:text-white text-lg">Try a Prediction</h3>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1.5">Crop Type</label>
              <select value={crop} onChange={e => setCrop(e.target.value)} className="w-full px-3 py-2.5 rounded-xl border border-green-200 dark:border-slate-600 bg-white dark:bg-slate-800 text-gray-800 dark:text-white text-sm focus:ring-2 focus:ring-green-500 focus:border-green-500 outline-none">
                {crops.map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1.5">Location (Mandi)</label>
              <select value={location} onChange={e => setLocation(e.target.value)} className="w-full px-3 py-2.5 rounded-xl border border-green-200 dark:border-slate-600 bg-white dark:bg-slate-800 text-gray-800 dark:text-white text-sm focus:ring-2 focus:ring-green-500 focus:border-green-500 outline-none">
                {locations.map(l => <option key={l}>{l}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1.5">Forecast Horizon</label>
              <div className="flex gap-2">
                {["30 Days","60 Days","90 Days"].map(h => (
                  <button key={h} className={`flex-1 py-2 rounded-lg text-xs font-semibold transition-all ${h === "90 Days" ? "bg-green-600 text-white shadow-md" : "bg-white dark:bg-slate-700 border border-green-200 dark:border-slate-600 text-gray-600 dark:text-slate-300 hover:bg-green-50 dark:hover:bg-slate-600"}`}>{h}</button>
                ))}
              </div>
            </div>

            {/* Prediction result card */}
            <div className="bg-white dark:bg-slate-800 rounded-xl border border-green-200 dark:border-slate-600 p-4 space-y-2 shadow-sm">
              <p className="text-xs text-gray-500 dark:text-slate-400 uppercase tracking-wide">Predicted Average Price (90 days)</p>
              <p className="text-3xl font-extrabold text-green-700 dark:text-green-400">₹{data.predicted[data.predicted.length - 1].toLocaleString()}<span className="text-sm font-medium text-gray-500 dark:text-slate-400">/qtl</span></p>
              <div className="flex items-center gap-2 text-xs">
                <span className="px-2 py-0.5 rounded-full bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 font-semibold">
                  ↑ {(((data.predicted[data.predicted.length-1] - data.actual[0]) / data.actual[0]) * 100).toFixed(1)}%
                </span>
                <span className="text-gray-400 dark:text-slate-500">vs current price</span>
              </div>
            </div>
            <p className="text-[10px] text-gray-400 dark:text-slate-500 text-center">Powered by Machine Learning • BiLSTM + Bahdanau Attention</p>
          </div>

          {/* Chart panel */}
          <div className="lg:col-span-3 bg-white dark:bg-slate-900 rounded-2xl border border-gray-100 dark:border-slate-800 p-6 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-gray-900 dark:text-white">Predicted vs Actual Prices</h3>
              <div className="flex items-center gap-4 text-xs text-gray-600 dark:text-slate-400">
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-500 rounded" /> Actual</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-green-500 rounded" /> Predicted</span>
                <span className="flex items-center gap-1"><span className="w-6 h-3 bg-green-100 dark:bg-green-900/30 rounded" /> 95% CI</span>
              </div>
            </div>
            <PredictionChart data={data} />
          </div>
        </div>
      </div>
    </section>
  );
}

function PredictionChart({ data }: { data: { actual: number[]; predicted: number[]; months: string[] } }) {
  const all = [...data.actual, ...data.predicted];
  const max = Math.max(...all) + 100;
  const min = Math.min(...all) - 100;
  const w = 500, h = 250, px = 40, py = 20;
  const totalPts = data.months.length;

  const toXY = (val: number, idx: number) => ({
    x: px + (idx / (totalPts - 1)) * (w - 2 * px),
    y: py + ((max - val) / (max - min)) * (h - 2 * py),
  });

  const actualPts = data.actual.map((v, i) => toXY(v, i));
  const predPts = data.predicted.map((v, i) => toXY(v, i));

  const ciUpper = data.predicted.map((v, i) => toXY(v + 80, i));
  const ciLower = data.predicted.map((v, i) => toXY(v - 80, i));
  const ciPath = `M${ciUpper.map(p => `${p.x},${p.y}`).join(" L")} L${ciLower.reverse().map(p => `${p.x},${p.y}`).join(" L")} Z`;

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-auto">
      {/* Grid */}
      {[0,1,2,3,4].map(i => {
        const y = py + (i / 4) * (h - 2 * py);
        const val = Math.round(max - (i / 4) * (max - min));
        return (
          <g key={i}>
            <line x1={px} y1={y} x2={w-px} y2={y} stroke="currentColor" strokeWidth="1" className="text-gray-100 dark:text-slate-800" />
            <text x={px - 5} y={y + 3} textAnchor="end" className="text-[8px] fill-gray-400 dark:fill-slate-500">₹{val}</text>
          </g>
        );
      })}
      {/* CI band */}
      <path d={ciPath} fill="#22c55e" fillOpacity="0.12" />
      {/* Actual line */}
      <polyline points={actualPts.map(p => `${p.x},${p.y}`).join(" ")} fill="none" stroke="#3b82f6" strokeWidth="2.5" strokeLinecap="round" />
      {/* Predicted line - dashed */}
      <polyline points={predPts.map(p => `${p.x},${p.y}`).join(" ")} fill="none" stroke="#16a34a" strokeWidth="2.5" strokeDasharray="6,3" strokeLinecap="round" />
      {/* Dots */}
      {actualPts.map((p, i) => <circle key={`a${i}`} cx={p.x} cy={p.y} r="3.5" fill="#3b82f6" />)}
      {predPts.map((p, i) => <circle key={`p${i}`} cx={p.x} cy={p.y} r="3.5" fill="#16a34a" stroke="white" strokeWidth="1.5" />)}
      {/* X labels */}
      {data.months.map((m, i) => {
        const x = px + (i / (totalPts - 1)) * (w - 2 * px);
        return <text key={m} x={x} y={h - 2} textAnchor="middle" className="text-[9px] fill-gray-400 dark:fill-slate-500">{m}</text>;
      })}
    </svg>
  );
}
