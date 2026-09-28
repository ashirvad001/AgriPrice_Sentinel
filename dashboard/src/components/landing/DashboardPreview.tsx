"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { CROPS, STATES, getDistricts, getMandis } from "@/lib/crops";

export default function DashboardPreview() {
  const router = useRouter();
  const [crop, setCrop] = useState("");
  const [state, setState] = useState("");
  const [district, setDistrict] = useState("");
  const [mandi, setMandi] = useState("");
  const districts = state ? getDistricts(state) : [];
  const mandis = district ? getMandis(state, district) : [];

  // Regional comparison data
  const regions = [
    { name: "Delhi", price: 2480, change: 5.2 },
    { name: "Punjab", price: 2350, change: 3.1 },
    { name: "UP", price: 2200, change: -1.4 },
    { name: "MP", price: 2150, change: 2.8 },
    { name: "Rajasthan", price: 2100, change: -0.6 },
    { name: "Haryana", price: 2300, change: 4.0 },
  ];
  const maxPrice = Math.max(...regions.map(r => r.price));

  return (
    <section id="dashboard-preview" className="py-20 md:py-28 bg-gradient-to-b from-white to-green-50 dark:from-slate-950 dark:to-slate-900 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-2xl mx-auto mb-14">
          <span className="inline-block px-3 py-1 text-xs font-semibold bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 rounded-full mb-4">Dashboard</span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-4">Explore the Dashboard</h2>
          <p className="text-gray-600 dark:text-slate-400 text-lg">Select a crop and mandi to jump straight into AI-powered forecasts, historical trends, and SHAP explainability.</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Crop selector card */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-green-100 dark:border-slate-800 shadow-lg p-6 space-y-5">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white">🌾 Select Crop & Market</h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Crop</label>
                <select value={crop} onChange={e => setCrop(e.target.value)} className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-slate-700 text-sm text-gray-800 dark:text-white bg-gray-50 dark:bg-slate-800 focus:ring-2 focus:ring-green-500 outline-none">
                  <option value="">Select Crop</option>
                  {CROPS.map(c => <option key={c.name} value={c.name}>{c.emoji} {c.name}</option>)}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">State</label>
                  <select value={state} onChange={e => { setState(e.target.value); setDistrict(""); setMandi(""); }} className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-slate-700 text-sm text-gray-800 dark:text-white bg-gray-50 dark:bg-slate-800 focus:ring-2 focus:ring-green-500 outline-none">
                    <option value="">Select State</option>
                    {STATES.map(s => <option key={s}>{s}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">District</label>
                  <select value={district} onChange={e => { setDistrict(e.target.value); setMandi(""); }} disabled={!state} className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-slate-700 text-sm text-gray-800 dark:text-white bg-gray-50 dark:bg-slate-800 focus:ring-2 focus:ring-green-500 outline-none disabled:opacity-50">
                    <option value="">Select District</option>
                    {districts.map(d => <option key={d}>{d}</option>)}
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Mandi</label>
                <select value={mandi} onChange={e => setMandi(e.target.value)} disabled={!district} className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-slate-700 text-sm text-gray-800 dark:text-white bg-gray-50 dark:bg-slate-800 focus:ring-2 focus:ring-green-500 outline-none disabled:opacity-50">
                  <option value="">Select Mandi</option>
                  {mandis.map(m => <option key={m}>{m}</option>)}
                </select>
              </div>
              <button
                disabled={!crop || !mandi}
                onClick={() => router.push(`/dashboard/${encodeURIComponent(crop.toLowerCase())}/${encodeURIComponent(mandi)}`)}
                className="w-full py-3 mt-2 text-sm font-bold text-white bg-green-600 rounded-xl hover:bg-green-700 shadow-lg shadow-green-200 dark:shadow-green-900/30 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              >
                🚀 Open Dashboard
              </button>
            </div>
          </div>

          {/* Regional comparison chart */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-green-100 dark:border-slate-800 shadow-lg p-6">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-1">📊 Regional Price Comparison</h3>
            <p className="text-xs text-gray-500 dark:text-slate-500 mb-5">Wheat prices across major states (₹/qtl)</p>
            <div className="space-y-3">
              {regions.map((r, i) => (
                <div key={i} className="flex items-center gap-3">
                  <span className="w-20 text-sm font-medium text-gray-700 dark:text-slate-300 shrink-0">{r.name}</span>
                  <div className="flex-1 h-8 bg-gray-100 dark:bg-slate-800 rounded-lg overflow-hidden relative">
                    <div
                      className="h-full bg-gradient-to-r from-green-400 to-emerald-500 rounded-lg transition-all duration-700"
                      style={{ width: `${(r.price / maxPrice) * 100}%` }}
                    />
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-semibold text-gray-700 dark:text-slate-300">₹{r.price.toLocaleString()}</span>
                  </div>
                  <span className={`text-xs font-semibold w-12 text-right ${r.change >= 0 ? "text-green-600" : "text-red-500"}`}>
                    {r.change >= 0 ? "↑" : "↓"}{Math.abs(r.change)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
