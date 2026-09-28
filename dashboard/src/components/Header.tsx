// src/components/Header.tsx — Crop/Mandi/Horizon selectors
"use client";

import { CROPS, STATES, getDistricts, getMandis, type CropInfo } from "@/lib/crops";
import Link from "next/link";

interface HeaderProps {
  selectedCrop: CropInfo;
  selectedState: string;
  selectedDistrict: string;
  selectedMandi: string;
  horizon: number;
  onCropChange: (crop: CropInfo) => void;
  onStateChange: (state: string) => void;
  onDistrictChange: (district: string) => void;
  onMandiChange: (mandi: string) => void;
  onHorizonChange: (h: number) => void;
}

export default function Header({
  selectedCrop, selectedState, selectedDistrict, selectedMandi,
  horizon, onCropChange, onStateChange, onDistrictChange, onMandiChange, onHorizonChange,
}: HeaderProps) {
  const districts = selectedState ? getDistricts(selectedState) : [];
  const mandis = selectedState && selectedDistrict ? getMandis(selectedState, selectedDistrict) : [];

  return (
    <header className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b border-slate-700/50 sticky top-0 z-50 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-4">
        {/* Title row */}
        <div className="flex items-center justify-between gap-4 mb-4">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="Logo" className="w-14 h-14 object-contain" />
            <div>
              <h1 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
                AgriPrice Sentinel
              </h1>
              <p className="text-xs text-slate-400">Crop Price Intelligence Dashboard</p>
            </div>
          </div>
          <Link href="/" className="flex items-center gap-1.5 text-xs font-semibold text-emerald-400 hover:text-emerald-300 bg-emerald-500/10 hover:bg-emerald-500/20 px-3.5 py-2 rounded-full border border-emerald-500/20 transition-all shadow-sm">
            <span className="text-sm">←</span> Back to Home
          </Link>
        </div>

        {/* Selectors */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {/* Crop */}
          <div className="col-span-2 sm:col-span-1">
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-medium">Crop</label>
            <select
              value={selectedCrop.name}
              onChange={(e) => {
                const crop = CROPS.find((c) => c.name === e.target.value)!;
                onCropChange(crop);
              }}
              className="w-full bg-slate-800/80 border border-slate-600/50 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-emerald-500/40 focus:border-emerald-500/60 transition-all appearance-none cursor-pointer"
            >
              {CROPS.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.emoji} {c.name}
                </option>
              ))}
            </select>
          </div>

          {/* State */}
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-medium">State</label>
            <select
              value={selectedState}
              onChange={(e) => onStateChange(e.target.value)}
              className="w-full bg-slate-800/80 border border-slate-600/50 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-emerald-500/40 transition-all appearance-none cursor-pointer"
            >
              {STATES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>

          {/* District */}
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-medium">District</label>
            <select
              value={selectedDistrict}
              onChange={(e) => onDistrictChange(e.target.value)}
              className="w-full bg-slate-800/80 border border-slate-600/50 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-emerald-500/40 transition-all appearance-none cursor-pointer"
            >
              {districts.map((d) => (
                <option key={d} value={d}>{d}</option>
              ))}
            </select>
          </div>

          {/* Mandi */}
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-medium">Mandi</label>
            <select
              value={selectedMandi}
              onChange={(e) => onMandiChange(e.target.value)}
              className="w-full bg-slate-800/80 border border-slate-600/50 rounded-lg px-3 py-2 text-sm text-white focus:ring-2 focus:ring-emerald-500/40 transition-all appearance-none cursor-pointer"
            >
              {mandis.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* Horizon */}
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-medium">Horizon</label>
            <div className="flex gap-1">
              {[30, 60, 90].map((h) => (
                <button
                  key={h}
                  onClick={() => onHorizonChange(h)}
                  className={`flex-1 py-2 rounded-lg text-sm font-medium transition-all ${
                    horizon === h
                      ? "bg-emerald-500 text-white shadow-lg shadow-emerald-500/30"
                      : "bg-slate-800/80 text-slate-400 hover:bg-slate-700/80 hover:text-white border border-slate-600/50"
                  }`}
                >
                  {h}d
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
