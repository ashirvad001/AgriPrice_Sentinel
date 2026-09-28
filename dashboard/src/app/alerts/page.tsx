// src/app/alerts/page.tsx — Alerts configuration page
import AlertForm from "@/components/AlertForm";
import Link from "next/link";

export const metadata = {
  title: "Price Alerts — AgriPrice Sentinel",
  description: "Configure WhatsApp price alerts for your crops and mandis.",
};

export default function AlertsPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <header className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-700/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center text-xl shadow-lg shadow-emerald-500/20">
              🔔
            </div>
            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">Price Alerts</h1>
              <p className="text-xs text-slate-400">Configure WhatsApp notifications</p>
            </div>
          </div>
          <Link
            href="/"
            className="text-sm text-slate-400 hover:text-emerald-400 transition-colors flex items-center gap-1"
          >
            ← Back to Dashboard
          </Link>
        </div>
      </header>

      {/* Form */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
        <AlertForm />
      </main>
    </div>
  );
}
