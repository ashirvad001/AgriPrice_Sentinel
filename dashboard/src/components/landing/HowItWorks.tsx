"use client";
import { Database, Brain, LineChart } from "lucide-react";

const steps = [
  { icon: Database, num: "01", title: "Collect Data", desc: "We pull daily price records from 40+ mandis and weather stations across India.", color: "from-green-500 to-emerald-500" },
  { icon: Brain, num: "02", title: "Analyze with AI", desc: "Our BiLSTM + Attention models analyze 48 engineered features to find hidden patterns.", color: "from-emerald-500 to-teal-500" },
  { icon: LineChart, num: "03", title: "Deliver Insights", desc: "Get forecasts up to 90 days ahead with confidence intervals, SHAP explanations, and alerts.", color: "from-teal-500 to-green-600" },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-20 md:py-28 bg-gradient-to-b from-green-50 to-white dark:from-slate-900 dark:to-slate-950 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block px-3 py-1 text-xs font-semibold bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 rounded-full mb-4">How It Works</span>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-4">From Raw Data to Smart Decisions</h2>
          <p className="text-gray-600 dark:text-slate-400 text-lg">Three simple steps power an advanced AI pipeline behind the scenes.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 relative">
          {/* Connector line (desktop) */}
          <div className="hidden md:block absolute top-24 left-[16.7%] right-[16.7%] h-0.5 bg-gradient-to-r from-green-300 via-emerald-300 to-teal-300 dark:from-green-700 dark:via-emerald-700 dark:to-teal-700" />

          {steps.map((s, i) => (
            <div key={i} className="relative flex flex-col items-center text-center group">
              {/* Step circle */}
              <div className={`relative w-20 h-20 rounded-full bg-gradient-to-br ${s.color} flex items-center justify-center mb-6 shadow-lg group-hover:scale-110 transition-transform`}>
                <s.icon className="h-8 w-8 text-white" />
                <span className="absolute -top-2 -right-2 w-7 h-7 rounded-full bg-white dark:bg-slate-800 shadow-md text-xs font-bold text-green-700 dark:text-green-400 flex items-center justify-center border border-green-200 dark:border-green-800">{s.num}</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">{s.title}</h3>
              <p className="text-sm text-gray-500 dark:text-slate-400 max-w-xs">{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
