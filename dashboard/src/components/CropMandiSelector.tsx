// src/components/CropMandiSelector.tsx — Inline crop + mandi selector for dashboard header
"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { ChevronDown } from "lucide-react";
import { CROPS, STATES, getDistricts, getMandis, type CropInfo } from "@/lib/crops";

interface CropMandiSelectorProps {
  currentCrop: string;
  currentMandi: string;
}

export default function CropMandiSelector({ currentCrop, currentMandi }: CropMandiSelectorProps) {
  const router = useRouter();
  const [cropOpen, setCropOpen] = useState(false);
  const [mandiOpen, setMandiOpen] = useState(false);

  // Build a flat list of all mandis for the dropdown
  const allMandis: string[] = [];
  for (const state of STATES) {
    for (const district of getDistricts(state)) {
      for (const mandi of getMandis(state, district)) {
        if (!allMandis.includes(mandi)) allMandis.push(mandi);
      }
    }
  }

  // Persist selection
  useEffect(() => {
    localStorage.setItem("selectedCrop", currentCrop);
    localStorage.setItem("selectedMandi", currentMandi);
  }, [currentCrop, currentMandi]);

  const handleCropChange = useCallback((cropName: string) => {
    setCropOpen(false);
    router.push(`/dashboard/${encodeURIComponent(cropName.toLowerCase())}/${encodeURIComponent(currentMandi)}`);
  }, [currentMandi, router]);

  const handleMandiChange = useCallback((mandi: string) => {
    setMandiOpen(false);
    router.push(`/dashboard/${encodeURIComponent(currentCrop.toLowerCase())}/${encodeURIComponent(mandi)}`);
  }, [currentCrop, router]);

  // Close dropdowns on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-selector]")) {
        setCropOpen(false);
        setMandiOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const currentCropInfo = CROPS.find(c => c.name.toLowerCase() === currentCrop.toLowerCase());

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {/* Crop Selector */}
      <div className="relative" data-selector>
        <button
          onClick={() => { setCropOpen(!cropOpen); setMandiOpen(false); }}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white font-semibold text-base transition-all"
        >
          <span>{currentCropInfo?.emoji ?? "🌾"}</span>
          <span>{currentCrop}</span>
          <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${cropOpen ? "rotate-180" : ""}`} />
        </button>

        {cropOpen && (
          <div className="absolute top-full left-0 mt-1.5 w-56 max-h-72 overflow-y-auto bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-50 py-1 animate-in fade-in slide-in-from-top-2 duration-200">
            {CROPS.map((crop) => {
              const isActive = crop.name.toLowerCase() === currentCrop.toLowerCase();
              return (
                <button
                  key={crop.name}
                  onClick={() => handleCropChange(crop.name)}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 text-sm text-left transition-colors ${
                    isActive
                      ? "bg-emerald-500/15 text-emerald-400 font-semibold"
                      : "text-slate-300 hover:bg-slate-700 hover:text-white"
                  }`}
                >
                  <span className="text-base">{crop.emoji}</span>
                  <span>{crop.name}</span>
                  {isActive && <span className="ml-auto text-emerald-400 text-xs">✓</span>}
                </button>
              );
            })}
          </div>
        )}
      </div>

      <span className="text-slate-500 font-normal text-lg">at</span>

      {/* Mandi Selector */}
      <div className="relative" data-selector>
        <button
          onClick={() => { setMandiOpen(!mandiOpen); setCropOpen(false); }}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-white font-semibold text-base transition-all"
        >
          <span>🏪</span>
          <span>{currentMandi}</span>
          <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${mandiOpen ? "rotate-180" : ""}`} />
        </button>

        {mandiOpen && (
          <div className="absolute top-full left-0 mt-1.5 w-64 max-h-72 overflow-y-auto bg-slate-800 border border-slate-700 rounded-xl shadow-2xl z-50 py-1 animate-in fade-in slide-in-from-top-2 duration-200">
            {STATES.map((state) => (
              <div key={state}>
                <p className="px-3 pt-2.5 pb-1 text-[10px] uppercase tracking-wider text-slate-500 font-semibold">{state}</p>
                {getDistricts(state).map((district) =>
                  getMandis(state, district).map((mandi) => {
                    const isActive = mandi === currentMandi;
                    return (
                      <button
                        key={mandi}
                        onClick={() => handleMandiChange(mandi)}
                        className={`w-full flex items-center gap-2 px-3 py-1.5 text-sm text-left transition-colors ${
                          isActive
                            ? "bg-emerald-500/15 text-emerald-400 font-semibold"
                            : "text-slate-300 hover:bg-slate-700 hover:text-white"
                        }`}
                      >
                        <span>{mandi}</span>
                        {isActive && <span className="ml-auto text-emerald-400 text-xs">✓</span>}
                      </button>
                    );
                  })
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
