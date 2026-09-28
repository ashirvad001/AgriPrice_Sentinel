"use client";
import { Tractor, TrendingUp, BookOpen } from "lucide-react";

const users = [
  { icon: Tractor, title: "Farmers", desc: "Make better selling decisions by knowing when prices will rise above MSP.", color: "bg-green-100 text-green-600 dark:bg-green-900/40 dark:text-green-400", border: "border-green-200 dark:border-green-800" },
  { icon: TrendingUp, title: "Traders", desc: "Get real-time market insights and cross-mandi comparisons to maximize margins.", color: "bg-blue-100 text-blue-600 dark:bg-blue-900/40 dark:text-blue-400", border: "border-blue-200 dark:border-blue-800" },
  { icon: BookOpen, title: "Analysts", desc: "Access multi-year data with statistical analysis for data-driven agricultural research.", color: "bg-purple-100 text-purple-600 dark:bg-purple-900/40 dark:text-purple-400", border: "border-purple-200 dark:border-purple-800" },
];

const testimonials = [
  { name: "Ramesh Patel", role: "Wheat Farmer, Punjab", avatar: "👨‍🌾", text: "AgriPrice Sentinel helped me sell my wheat at the right time. The predictions were very close to actual prices. I earned ₹15,000 more this season!" },
  { name: "Sunita Devi", role: "Rice Farmer, UP", avatar: "👩‍🌾", text: "The WhatsApp alerts are so convenient. I don't need to check prices daily—the app tells me when it's the right time to sell." },
  { name: "Anil Kumar", role: "Mandi Trader, Delhi", avatar: "🧑‍💼", text: "The regional comparison feature helps me find price differences across mandis. It's become essential for my daily trading decisions." },
];

export default function UsersAndTestimonials() {
  return (
    <>
      {/* Benefits */}
      <section className="py-20 md:py-24 bg-white dark:bg-slate-950 transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center max-w-2xl mx-auto mb-14">
            <span className="inline-block px-3 py-1 text-xs font-semibold bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 rounded-full mb-4">Who Benefits</span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-4">Built for Everyone in Agriculture</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {users.map((u, i) => (
              <div key={i} className={`rounded-2xl border ${u.border} p-8 text-center hover:shadow-lg hover:-translate-y-1 transition-all duration-300 bg-white dark:bg-slate-900`}>
                <div className={`w-16 h-16 rounded-2xl ${u.color} flex items-center justify-center mx-auto mb-5`}>
                  <u.icon className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-2">{u.title}</h3>
                <p className="text-sm text-gray-500 dark:text-slate-400">{u.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section id="testimonials" className="py-20 md:py-24 bg-gradient-to-b from-green-50 to-white dark:from-slate-900 dark:to-slate-950 transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center max-w-2xl mx-auto mb-14">
            <span className="inline-block px-3 py-1 text-xs font-semibold bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-400 rounded-full mb-4">Testimonials</span>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-gray-900 dark:text-white mb-4">Trusted by Farmers Across India</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-6">
            {testimonials.map((t, i) => (
              <div key={i} className="bg-white dark:bg-slate-900 rounded-2xl border border-green-100 dark:border-slate-800 p-6 shadow-sm hover:shadow-lg transition-shadow">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-12 h-12 rounded-full bg-green-100 dark:bg-green-900/40 flex items-center justify-center text-2xl">{t.avatar}</div>
                  <div>
                    <p className="font-bold text-gray-900 dark:text-white text-sm">{t.name}</p>
                    <p className="text-xs text-gray-500 dark:text-slate-400">{t.role}</p>
                  </div>
                </div>
                <p className="text-sm text-gray-600 dark:text-slate-400 leading-relaxed italic">&quot;{t.text}&quot;</p>
                <div className="flex gap-0.5 mt-3">
                  {[1,2,3,4,5].map(s => <span key={s} className="text-amber-400 text-sm">★</span>)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
