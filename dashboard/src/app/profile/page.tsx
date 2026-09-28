"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";
import { isDemoMode } from "@/lib/demo-mode";
import { 
  User, MapPin, Phone, Leaf, Bell, TrendingUp, Settings, Edit3, 
  Search, ShieldAlert, CheckCircle2, ChevronRight, LogOut, FlaskConical 
} from "lucide-react";
import { CROPS, STATES, getDistricts, getMandis } from "@/lib/crops";

export default function FarmerProfileDashboard() {
  const router = useRouter();
  const { user, isLoggedIn, isLoading, isDemo, updateProfile, logout } = useAuth();
  
  const [activeTab, setActiveTab] = useState<"dashboard" | "settings">("dashboard");
  const [editMode, setEditMode] = useState(false);
  const [editedUser, setEditedUser] = useState<Partial<any>>({});

  useEffect(() => {
    if (!isLoading && !isLoggedIn) {
      router.push("/login");
    }
    if (user) {
      setEditedUser(user);
    }
  }, [isLoading, isLoggedIn, router, user]);

  if (isLoading || !isLoggedIn) {
    return <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-slate-950"><div className="animate-spin w-8 h-8 rounded-full border-4 border-green-200 dark:border-green-900 border-t-green-600 dark:border-t-green-400"></div></div>;
  }

  // Mock data for Farmer Dashboard
  const myCrops = user?.crops?.length ? user.crops : ["Wheat", "Mustard"];
  const location = user?.location || "Karnal, Haryana";

  const handleSaveProfile = () => {
    updateProfile(editedUser);
    setEditMode(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-slate-950 pt-20 pb-24 transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        
        {/* Header & greeting */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
          <div>
            <div className="mb-3 flex items-center gap-2">
              <button onClick={() => router.push("/")} className="text-sm font-bold text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-900/30 hover:bg-green-100 dark:hover:bg-green-900/50 flex items-center gap-2 transition-colors px-3.5 py-1.5 rounded-full border border-green-200 dark:border-green-800 w-fit">
                <span className="text-sm leading-none">←</span> Back to Home
              </button>
              {isDemo && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-400 text-xs font-bold border border-amber-200 dark:border-amber-800">
                  <FlaskConical className="w-3.5 h-3.5" /> Demo Mode
                </span>
              )}
            </div>
            <h1 className="text-3xl font-extrabold text-gray-900 dark:text-white tracking-tight">Farmer Dashboard</h1>
            <p className="text-gray-500 dark:text-slate-400 mt-1 flex items-center gap-2">
              Welcome back, <span className="font-semibold text-green-700 dark:text-green-400">{user?.full_name?.split(" ")[0] || "Farmer"}</span> {isDemo ? "🧪" : "👋"}
            </p>
          </div>
          <div className="flex bg-white dark:bg-slate-900 rounded-lg shadow-sm border border-gray-200 dark:border-slate-800 p-1">
            <button 
              onClick={() => setActiveTab("dashboard")} 
              className={`px-6 py-2 rounded-md font-medium text-sm transition-all ${activeTab === "dashboard" ? "bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-400 shadow-sm" : "text-gray-600 dark:text-slate-400 hover:bg-gray-50 dark:hover:bg-slate-800"}`}
            >
              Overview
            </button>
            <button 
              onClick={() => setActiveTab("settings")} 
              className={`px-6 py-2 rounded-md font-medium text-sm transition-all flex items-center gap-2 ${activeTab === "settings" ? "bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-400 shadow-sm" : "text-gray-600 dark:text-slate-400 hover:bg-gray-50 dark:hover:bg-slate-800"}`}
            >
              Settings
            </button>
          </div>
        </div>

        {activeTab === "dashboard" ? (
          <div className="grid lg:grid-cols-3 gap-6">
            
            {/* LEFT COLUMN: Profile & Alerts */}
            <div className="space-y-6 lg:col-span-1">
              
              {/* Profile Card */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 shadow-sm overflow-hidden">
                <div className="h-24 bg-gradient-to-r from-green-500 to-emerald-600 w-full relative">
                   <button onClick={() => setActiveTab("settings")} className="absolute top-4 right-4 w-8 h-8 bg-white/20 hover:bg-white/30 backdrop-blur rounded-full flex items-center justify-center text-white transition-colors">
                      <Edit3 className="w-4 h-4" />
                   </button>
                </div>
                <div className="px-6 pb-6 relative">
                  <div className="w-20 h-20 rounded-full border-4 border-white dark:border-slate-900 bg-green-100 dark:bg-green-900/50 flex items-center justify-center text-green-700 dark:text-green-400 text-2xl font-bold shadow-md absolute -top-10 left-6">
                    {user?.avatar ? <img src={user.avatar} className="w-full h-full rounded-full object-cover" alt="avatar" /> : user?.full_name?.charAt(0) || "F"}
                  </div>
                  <div className="pt-12">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">{user?.full_name || "Guest Farmer"}</h2>
                    <p className="text-sm text-gray-500 dark:text-slate-400 font-medium mb-4 flex items-center gap-1 mt-1">
                       <MapPin className="w-4 h-4" /> {location}
                    </p>
                    <div className="space-y-3">
                      <div className="flex items-center gap-3 text-sm">
                        <div className="w-8 h-8 rounded-full bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 flex items-center justify-center"><Phone className="w-4 h-4" /></div>
                        <span className="text-gray-700 dark:text-slate-300 font-medium">{user?.phone}</span>
                      </div>
                      <div className="flex gap-3 text-sm items-start">
                        <div className="w-8 h-8 rounded-full bg-emerald-50 dark:bg-emerald-900/30 text-emerald-600 dark:text-emerald-400 flex items-center justify-center shrink-0 mt-0.5"><Leaf className="w-4 h-4" /></div>
                        <div className="flex flex-wrap gap-1.5 pt-1">
                          {myCrops.map(c => <span key={c} className="px-2.5 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-900/40 text-emerald-800 dark:text-emerald-400 text-xs font-semibold">{c}</span>)}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Alerts & Notifications */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 shadow-sm p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-bold text-gray-900 dark:text-white flex items-center gap-2"><Bell className="w-5 h-5 text-amber-500" /> Smart Alerts</h3>
                  <span className="bg-red-100 dark:bg-red-900/40 text-red-600 dark:text-red-400 text-xs font-bold px-2 py-0.5 rounded-full">2 New</span>
                </div>
                <div className="space-y-4">
                  <div className="flex gap-3 p-3 rounded-xl bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800 items-start">
                    <CheckCircle2 className="w-5 h-5 text-green-600 dark:text-green-400 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-bold text-gray-900 dark:text-white">Sell Now: Wheat</p>
                      <p className="text-xs text-gray-600 dark:text-slate-400 mt-0.5">Price in Karnal Mandi is ₹2,480, which is 9% above MSP. Peak expected this week.</p>
                    </div>
                  </div>
                  <div className="flex gap-3 p-3 rounded-xl bg-amber-50 dark:bg-amber-900/20 border border-amber-100 dark:border-amber-800 items-start">
                    <ShieldAlert className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-bold text-gray-900 dark:text-white">Hold: Mustard</p>
                      <p className="text-xs text-gray-600 dark:text-slate-400 mt-0.5">Current price ₹5,200. Expected to rise by ~4% in the next 15 days.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* RIGHT COLUMN: Insights & ML Panel */}
            <div className="space-y-6 lg:col-span-2">
              
              {/* My Crops Insight Cards */}
              <h3 className="font-bold text-gray-900 dark:text-white text-lg mb-4 flex items-center gap-2"><TrendingUp className="w-5 h-5 text-green-600 dark:text-green-400" /> Market Snapshot (Your Crops)</h3>
              <div className="grid sm:grid-cols-2 gap-4">
                {myCrops.slice(0, 2).map((cropStr, i) => {
                  const isUp = i === 0;
                  return (
                    <div key={cropStr} className="bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 shadow-sm p-5 relative overflow-hidden group hover:border-green-300 dark:hover:border-green-700 transition-colors cursor-pointer" onClick={() => router.push(`/dashboard/${cropStr.toLowerCase()}/${encodeURIComponent(location.split(",")[0].trim() + " Mandi")}`)}>
                      <div className="absolute right-0 top-0 w-24 h-24 bg-gradient-to-br from-green-50 dark:from-green-900/20 to-transparent rounded-bl-full opacity-50" />
                      <div className="flex justify-between items-start mb-4 relative z-10">
                        <div>
                           <p className="text-xs font-bold text-gray-500 dark:text-slate-500 uppercase tracking-wider">{location.split(",")[0]}</p>
                           <h4 className="text-xl font-extrabold text-gray-900 dark:text-white mt-1">{cropStr}</h4>
                        </div>
                        <button className="w-8 h-8 rounded-full bg-gray-50 dark:bg-slate-800 flex items-center justify-center text-gray-400 dark:text-slate-500 group-hover:bg-green-100 dark:group-hover:bg-green-900/40 group-hover:text-green-700 dark:group-hover:text-green-400 transition-colors">
                          <ChevronRight className="w-5 h-5" />
                        </button>
                      </div>
                      <div className="flex items-end gap-3 mb-4">
                        <span className="text-3xl font-black text-gray-900 dark:text-white">₹{isUp ? "2,480" : "5,200"}</span>
                        <span className={`text-sm font-bold pb-1 flex items-center ${isUp ? 'text-green-600' : 'text-red-500'}`}>
                          {isUp ? "↑ 5.2%" : "↓ 1.4%"}
                        </span>
                      </div>
                      
                      <div className="bg-gray-50 dark:bg-slate-800 rounded-lg p-3 text-sm flex justify-between items-center border border-gray-100 dark:border-slate-700">
                         <span className="text-gray-500 dark:text-slate-400 font-medium flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> ML Predicts
                         </span>
                         <span className="font-bold text-gray-900 dark:text-white">₹{isUp ? "2,550" : "5,350"} (30d)</span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* ML Prediction Panel (Demo UI embedded) */}
              <div className="bg-gradient-to-br from-gray-900 to-slate-900 rounded-2xl shadow-xl overflow-hidden text-white relative">
                 {/* Decorative background grid */}
                 <div className="absolute inset-0 bg-[url('https://transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
                 
                 <div className="p-6 relative z-10">
                    <div className="flex items-center justify-between mb-6">
                      <div>
                        <h3 className="font-bold text-xl flex items-center gap-2"><span className="text-xl">🤖</span> Custom Price Forecast</h3>
                        <p className="text-slate-400 text-sm mt-1">Run AI predictions for any crop and mandi.</p>
                      </div>
                      <span className="bg-green-500/20 text-green-400 text-[10px] font-bold uppercase tracking-widest px-3 py-1 rounded-full border border-green-500/30 backdrop-blur-sm shadow-[0_0_15px_rgba(34,197,94,0.2)]">
                        BiLSTM Engine Live
                      </span>
                    </div>

                    <div className="grid sm:grid-cols-3 gap-4 mb-6">
                      <div className="space-y-1.5">
                        <label className="text-xs font-semibold text-slate-400">SELECT CROP</label>
                        <select className="w-full bg-slate-800/80 border border-slate-700 text-white text-sm rounded-xl py-2.5 px-3 focus:ring-2 focus:ring-green-500 outline-none">
                          {CROPS.map(c => <option key={c.name}>{c.name}</option>)}
                        </select>
                      </div>
                      <div className="space-y-1.5">
                        <label className="text-xs font-semibold text-slate-400">SELECT MARKET</label>
                        <select className="w-full bg-slate-800/80 border border-slate-700 text-white text-sm rounded-xl py-2.5 px-3 focus:ring-2 focus:ring-green-500 outline-none">
                          <option>Indore Mandi</option>
                          <option>Azadpur Mandi</option>
                          <option>Karnal Mandi</option>
                        </select>
                      </div>
                      <div className="flex items-end">
                        <button className="w-full bg-green-500 hover:bg-green-400 text-slate-900 font-bold text-sm rounded-xl py-2.5 transition-all shadow-[0_0_20px_rgba(34,197,94,0.3)] hover:shadow-[0_0_25px_rgba(34,197,94,0.5)]">
                          Run Prediction
                        </button>
                      </div>
                    </div>

                    {/* Chart Mockup inside dark container */}
                    <div className="bg-slate-800/50 rounded-xl border border-slate-700 p-4 h-48 flex flex-col justify-end relative overflow-hidden backdrop-blur-sm">
                       {/* SVG Line mockup */}
                       <svg className="w-full h-full absolute inset-0 pt-4 px-2" preserveAspectRatio="none" viewBox="0 0 100 100">
                          <path d="M0,80 Q25,70 50,40 T100,20 L100,100 L0,100 Z" fill="rgba(34,197,94,0.1)"/>
                          <polyline points="0,80 25,70 50,40 75,45 100,20" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round"/>
                          <circle cx="100" cy="20" r="3" fill="#22c55e" stroke="#0f172a" strokeWidth="1"/>
                       </svg>
                       <div className="relative z-10 text-right">
                         <p className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">90-Day Forecast</p>
                         <p className="text-2xl font-black text-green-400 drop-shadow-md">₹2,840</p>
                         <p className="text-xs text-green-300 font-medium">↑ 14% high confidence</p>
                       </div>
                    </div>
                 </div>
              </div>

            </div>
          </div>
        ) : (
          /* SETTINGS TAB */
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-gray-200 dark:border-slate-800 shadow-sm overflow-hidden max-w-3xl mx-auto">
            <div className="border-b border-gray-100 dark:border-slate-800 p-6 flex justify-between items-center">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white flex items-center gap-2"><Settings className="w-5 h-5 text-gray-500 dark:text-slate-400" /> Account Settings</h2>
            </div>
            
            <div className="p-6 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Full Name</label>
                  <input 
                    type="text" 
                    disabled={!editMode}
                    value={editedUser.full_name || ""}
                    onChange={e => setEditedUser({...editedUser, full_name: e.target.value})}
                    className="w-full px-4 py-2 bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg text-gray-900 dark:text-white disabled:opacity-70 outline-none focus:ring-2 focus:ring-green-500 focus:bg-white dark:focus:bg-slate-700 transition-all"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Phone Number</label>
                  <input 
                    type="text" 
                    disabled 
                    value={user?.phone || ""}
                    className="w-full px-4 py-2 bg-gray-100 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg text-gray-500 dark:text-slate-400 opacity-70 cursor-not-allowed"
                  />
                  <p className="text-xs text-gray-400 dark:text-slate-500 mt-1">Phone number cannot be changed.</p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Primary Location / Village</label>
                  <input 
                    type="text" 
                    disabled={!editMode}
                    value={editedUser.location || location}
                    onChange={e => setEditedUser({...editedUser, location: e.target.value})}
                    className="w-full px-4 py-2 bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg text-gray-900 dark:text-white disabled:opacity-70 outline-none focus:ring-2 focus:ring-green-500 focus:bg-white dark:focus:bg-slate-700 transition-all"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">Crops I Grow (comma separated)</label>
                  <input 
                    type="text" 
                    disabled={!editMode}
                    value={editedUser.crops?.join(", ") || myCrops.join(", ")}
                    onChange={e => setEditedUser({...editedUser, crops: e.target.value.split(",").map(s => s.trim())})}
                    className="w-full px-4 py-2 bg-gray-50 dark:bg-slate-800 border border-gray-200 dark:border-slate-700 rounded-lg text-gray-900 dark:text-white disabled:opacity-70 outline-none focus:ring-2 focus:ring-green-500 focus:bg-white dark:focus:bg-slate-700 transition-all"
                  />
                </div>
              </div>

              <div className="pt-4 border-t border-gray-100 dark:border-slate-800 flex justify-between items-center">
                {editMode ? (
                   <div className="flex gap-2 w-full sm:w-auto">
                     <button onClick={() => setEditMode(false)} className="px-5 py-2 text-sm font-medium text-gray-600 dark:text-slate-400 bg-gray-100 dark:bg-slate-800 hover:bg-gray-200 dark:hover:bg-slate-700 rounded-lg transition-colors">Cancel</button>
                     <button onClick={handleSaveProfile} className="px-5 py-2 text-sm font-bold text-white bg-green-600 hover:bg-green-700 rounded-lg shadow-sm transition-all sm:w-auto w-full">Save Changes</button>
                   </div>
                ) : (
                   <button onClick={() => setEditMode(true)} className="px-5 py-2 text-sm font-medium text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-900/30 hover:bg-green-100 dark:hover:bg-green-900/50 border border-green-200 dark:border-green-800 rounded-lg transition-all w-full sm:w-auto">
                     Edit Profile
                   </button>
                )}
              </div>
            </div>

            {/* Logout Area */}
            <div className="bg-red-50 dark:bg-red-900/20 p-6 border-t border-red-100 dark:border-red-900/30 mt-4">
               <h3 className="text-red-800 dark:text-red-400 font-bold mb-1">{isDemo ? "Exit Demo Mode" : "Log Out of Sentinel"}</h3>
               <p className="text-sm text-red-600/80 dark:text-red-400/60 mb-4">{isDemo ? "This will clear all demo data and return to the homepage." : "You will stop receiving browser notifications."}</p>
               <button onClick={() => { logout(); router.push("/"); }} className="px-5 py-2 text-sm font-bold text-red-600 dark:text-red-400 bg-white dark:bg-slate-800 border border-red-200 dark:border-red-800 hover:bg-red-50 dark:hover:bg-red-900/30 hover:text-red-700 rounded-lg shadow-sm transition-all flex items-center gap-2">
                 <LogOut className="w-4 h-4" /> {isDemo ? "Exit Demo" : "Sign Out"}
               </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
