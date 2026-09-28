"use client";
import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Menu, X, User, LogOut, LayoutDashboard, Settings, Sun, Moon } from "lucide-react";
import { useAuth } from "@/lib/auth-context";
import { useTheme } from "@/lib/theme-context";


export default function Navbar() {
  const router = useRouter();
  const { user, isLoggedIn, isDemo, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef<HTMLDivElement>(null);

  const links = [
    { label: "Features", href: "/#features" },
    { label: "How It Works", href: "/#how-it-works" },
    { label: "Predictions", href: "/#predictions" },
    { label: "Dashboard", href: "/#dashboard-preview" },
    { label: "Testimonials", href: "/#testimonials" },
  ];

  // Close profile dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) {
        setProfileOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogout = () => {
    logout();
    router.push("/");
    setProfileOpen(false);
    setOpen(false);
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg border-b border-green-100 dark:border-slate-800 shadow-sm transition-colors duration-300">
      <div className="max-w-7xl mx-auto flex items-center justify-between px-4 sm:px-6 h-20">
        {/* Logo */}
        <button onClick={() => router.push("/")} className="flex items-center gap-2 text-green-700 dark:text-green-400 font-bold text-xl hover:opacity-80 transition-opacity">
          <img src="/logo.png" alt="AgriPrice Sentinel" className="h-16 w-auto" />
          <span className="hidden sm:inline">AgriPrice Sentinel</span>
        </button>



        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-6">
          {links.map((l) => (
            <a key={l.href} href={l.href} className="text-sm font-medium text-gray-600 dark:text-slate-400 hover:text-green-700 dark:hover:text-green-400 transition-colors">
              {l.label}
            </a>
          ))}
        </div>

        {/* Auth area + Theme Toggle (Desktop) */}
        <div className="hidden md:flex items-center gap-3">
          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-600 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-700 transition-all shadow-sm"
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          {isLoggedIn ? (
            <div className="relative" ref={profileRef}>
              <button 
                onClick={() => setProfileOpen(!profileOpen)}
                className="flex items-center gap-2 pl-3 pr-2 py-1.5 rounded-full border border-green-200 dark:border-green-800 hover:border-green-300 dark:hover:border-green-700 hover:bg-green-50 dark:hover:bg-green-900/30 transition-all bg-white dark:bg-slate-800 shadow-sm"
              >
                <span className="text-sm font-semibold text-gray-700 dark:text-slate-200 max-w-[120px] truncate">
                  Hi, {user?.full_name?.split(" ")[0] || "Farmer"} {isDemo ? "🧪" : "👋"}
                </span>
                <div className="w-8 h-8 rounded-full bg-green-100 dark:bg-green-900/50 flex items-center justify-center text-green-700 dark:text-green-400">
                  {user?.avatar ? (
                    <img src={user.avatar} alt="Profile" className="w-full h-full rounded-full object-cover" />
                  ) : (
                    <User className="w-4 h-4" />
                  )}
                </div>
              </button>

              {/* Profile Dropdown */}
              {profileOpen && (
                <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-slate-800 rounded-xl shadow-xl border border-gray-100 dark:border-slate-700 overflow-hidden py-1 animate-in fade-in slide-in-from-top-2">
                  <div className="px-4 py-3 border-b border-gray-50 dark:border-slate-700 flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/50 flex items-center justify-center text-green-700 dark:text-green-400 shrink-0">
                      {user?.avatar ? <img src={user.avatar} alt="Profile" className="w-full h-full rounded-full object-cover" /> : <User className="w-5 h-5" />}
                    </div>
                    <div className="overflow-hidden">
                      <p className="text-sm font-bold text-gray-900 dark:text-white truncate">{user?.full_name || "Guest User"}</p>
                      <p className="text-xs text-gray-500 dark:text-slate-400 truncate">{user?.phone}</p>
                    </div>
                  </div>
                  <div className="p-1">
                    <button onClick={() => { router.push("/profile"); setProfileOpen(false); }} className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-slate-300 hover:bg-green-50 dark:hover:bg-green-900/30 hover:text-green-700 dark:hover:text-green-400 rounded-lg transition-colors">
                      <LayoutDashboard className="w-4 h-4" /> My Dashboard
                    </button>
                    <button onClick={() => { router.push("/profile"); setProfileOpen(false); }} className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-700 dark:text-slate-300 hover:bg-green-50 dark:hover:bg-green-900/30 hover:text-green-700 dark:hover:text-green-400 rounded-lg transition-colors">
                      <Settings className="w-4 h-4" /> Settings
                    </button>
                    <div className="h-px bg-gray-100 dark:bg-slate-700 my-1 mx-2" />
                    <button onClick={handleLogout} className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors font-medium">
                      <LogOut className="w-4 h-4" /> {isDemo ? "Exit Demo" : "Log out"}
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <>
              <button onClick={() => router.push("/login")} className="px-4 py-2 text-sm font-semibold text-green-700 dark:text-green-400 border border-green-300 dark:border-green-800 rounded-lg hover:bg-green-50 dark:hover:bg-green-900/30 transition-all">
                Log In
              </button>
              <button onClick={() => router.push("/login")} className="px-4 py-2 text-sm font-semibold text-white bg-green-600 rounded-lg hover:bg-green-700 shadow-md shadow-green-200 dark:shadow-green-900/50 transition-all">
                Sign Up
              </button>
            </>
          )}
        </div>

        {/* Mobile: theme toggle + hamburger */}
        <div className="flex md:hidden items-center gap-2">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg border border-gray-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-gray-600 dark:text-slate-300 transition-all"
          >
            {theme === "dark" ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>
          <button className="text-gray-700 dark:text-slate-300" onClick={() => setOpen(!open)}>
            {open ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {open && (
        <div className="md:hidden bg-white dark:bg-slate-900 border-t border-green-100 dark:border-slate-800 px-4 pb-4 pt-2 space-y-1 shadow-lg transition-colors">
          {isLoggedIn && user ? (
            <div className="px-2 py-3 border-b border-green-50 dark:border-slate-800 mb-2 flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/50 flex items-center justify-center text-green-700 dark:text-green-400">
                {user.avatar ? <img src={user.avatar} alt="Profile" className="w-full h-full rounded-full object-cover" /> : <User className="w-5 h-5" />}
              </div>
              <div>
                <p className="text-sm font-bold text-gray-900 dark:text-white">{user.full_name || "Guest User"}</p>
                <p className="text-xs text-gray-500 dark:text-slate-400">{user.phone}</p>
              </div>
            </div>
          ) : null}

          {links.map((l) => (
            <a key={l.href} href={l.href} onClick={() => setOpen(false)} className="block px-2 py-2.5 text-gray-600 dark:text-slate-400 hover:text-green-700 dark:hover:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-lg font-medium transition-colors">
              {l.label}
            </a>
          ))}

          {isLoggedIn ? (
            <div className="pt-2 mt-2 border-t border-gray-100 dark:border-slate-800">
              <button onClick={() => { router.push("/profile"); setOpen(false); }} className="w-full flex items-center gap-3 px-2 py-2.5 text-gray-700 dark:text-slate-300 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-lg font-medium">
                <LayoutDashboard className="w-5 h-5 text-gray-500 dark:text-slate-400" /> My Dashboard
              </button>
              <button onClick={handleLogout} className="w-full flex items-center gap-3 px-2 py-2.5 text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg font-medium mt-1">
                <LogOut className="w-5 h-5 text-red-500" /> {isDemo ? "Exit Demo" : "Log out"}
              </button>
            </div>
          ) : (
            <div className="flex gap-2 pt-3 mt-2 border-t border-gray-100 dark:border-slate-800">
              <button onClick={() => { router.push("/login"); setOpen(false); }} className="flex-1 py-2.5 text-sm font-semibold text-green-700 dark:text-green-400 border border-green-300 dark:border-green-800 rounded-lg">Log In</button>
              <button onClick={() => { router.push("/login"); setOpen(false); }} className="flex-1 py-2.5 text-sm font-semibold text-white bg-green-600 rounded-lg shadow-sm">Sign Up</button>
            </div>
          )}
        </div>
      )}
    </nav>
  );
}
