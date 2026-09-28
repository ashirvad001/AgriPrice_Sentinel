"use client";
import { BarChart3, Brain, Bell, MapPin, TrendingUp, LayoutDashboard } from "lucide-react";

const features = [
  { icon: BarChart3, title: "Real-Time Mandi Prices", desc: "Live price tracking from 40+ mandis across India, updated daily via data.gov.in APIs.", color: "bg-green-100 text-green-600 dark:bg-green-900/40 dark:text-green-400" },
  { icon: Brain, title: "AI Price Prediction", desc: "BiLSTM + Attention neural network with Monte Carlo Dropout for 95% confidence intervals.", color: "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/40 dark:text-emerald-400" },
  { icon: TrendingUp, title: "Historical Trends", desc: "Analyze multi-year price patterns with seasonal annotations & harvest cycle overlays.", color: "bg-lime-100 text-lime-700 dark:bg-lime-900/40 dark:text-lime-400" },
  { icon: Bell, title: "Smart Alerts", desc: "WhatsApp notifications when prices cross your threshold—never miss the right selling time.", color: "bg-amber-100 text-amber-600 dark:bg-amber-900/40 dark:text-amber-400" },
  { icon: MapPin, title: "Region-wise Analytics", desc: "Compare prices across states, districts, and mandis with interactive heatmaps.", color: "bg-blue-100 text-blue-600 dark:bg-blue-900/40 dark:text-blue-400" },
  { icon: LayoutDashboard, title: "Easy Dashboard", desc: "Clean, farmer-friendly interface with SHAP explainability—AI insights you can understand.", color: "bg-purple-100 text-purple-600 dark:bg-purple-900/40 dark:text-purple-400" },
];

export default function FeaturesSection() {
  return (
    <section id="features" className="py-20 md:py-28 bg-white dark:bg-slate-950 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-2xl mx-auto mb-14">
          <span className="inline-block px-3 py-1 text-xs font-semibold bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 rounded-full mb-4">Features</span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-4">Everything You Need for Smarter Farming</h2>
          <p className="text-gray-600 dark:text-slate-400 text-lg">From real-time price tracking to AI-powered predictions, AgriPrice Sentinel gives farmers and traders every tool they need.</p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f, i) => (
            <div
              key={i}
              className="group relative bg-white dark:bg-slate-900 rounded-2xl border border-gray-100 dark:border-slate-800 p-6 hover:shadow-xl hover:shadow-green-50 dark:hover:shadow-green-900/10 hover:-translate-y-1 transition-all duration-300"
            >
              <div className={`w-12 h-12 rounded-xl ${f.color} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                <f.icon className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mb-2">{f.title}</h3>
              <p className="text-sm text-gray-500 dark:text-slate-400 leading-relaxed">{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
