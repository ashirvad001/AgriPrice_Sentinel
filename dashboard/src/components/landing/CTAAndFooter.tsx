"use client";
import { useRouter } from "next/navigation";
import { ArrowRight } from "lucide-react";

export default function CTAAndFooter() {
  const router = useRouter();
  return (
    <>
      {/* Final CTA */}
      <section className="py-20 md:py-28 bg-gradient-to-br from-green-600 via-emerald-600 to-teal-600 relative overflow-hidden">
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute -top-20 -right-20 w-80 h-80 rounded-full bg-white/5 blur-2xl" />
          <div className="absolute -bottom-20 -left-20 w-60 h-60 rounded-full bg-white/5 blur-2xl" />
        </div>
        <div className="relative max-w-3xl mx-auto px-4 sm:px-6 text-center">
          <h2 className="text-3xl sm:text-4xl lg:text-5xl font-extrabold text-white mb-4">
            Start Using AI for Smarter Farming Today
          </h2>
          <p className="text-green-100 text-lg mb-8 max-w-xl mx-auto">
            Join thousands of farmers and traders who are already making better decisions with AgriPrice Sentinel.
          </p>
          <button
            onClick={() => router.push("/login")}
            className="inline-flex items-center gap-2 px-8 py-4 text-lg font-bold text-green-700 bg-white rounded-xl hover:bg-green-50 shadow-xl shadow-green-900/20 transition-all hover:-translate-y-0.5"
          >
            Sign Up Now <ArrowRight className="h-5 w-5" />
          </button>
          <p className="text-green-200 text-sm mt-4">Free to use • No credit card required</p>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 dark:bg-slate-950 text-gray-400 py-12 transition-colors duration-300">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-8 mb-10">
            <div>
              <a href="#" className="flex items-center gap-2 text-white font-bold text-lg mb-3">
                <img src="/logo.png" alt="Logo" className="h-10 w-auto object-contain" /> AgriPrice Sentinel
              </a>
              <p className="text-sm leading-relaxed">AI-powered crop price forecasting platform for Indian mandi markets. Empowering farmers with data-driven decisions.</p>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-3 text-sm">Platform</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="#features" className="hover:text-green-400 transition-colors">Features</a></li>
                <li><a href="#predictions" className="hover:text-green-400 transition-colors">ML Predictions</a></li>
                <li><a href="#dashboard-preview" className="hover:text-green-400 transition-colors">Dashboard</a></li>
                <li><a href="#how-it-works" className="hover:text-green-400 transition-colors">How It Works</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-3 text-sm">Resources</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="#" className="hover:text-green-400 transition-colors">API Documentation</a></li>
                <li><a href="#" className="hover:text-green-400 transition-colors">Research Paper</a></li>
                <li><a href="#" className="hover:text-green-400 transition-colors">Data Sources</a></li>
                <li><a href="#" className="hover:text-green-400 transition-colors">Model Architecture</a></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-semibold mb-3 text-sm">Connect</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="#" className="hover:text-green-400 transition-colors">GitHub</a></li>
                <li><a href="#" className="hover:text-green-400 transition-colors">Contact Us</a></li>
                <li><a href="#" className="hover:text-green-400 transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-green-400 transition-colors">Terms of Service</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-800 dark:border-slate-800 pt-6 flex flex-col sm:flex-row items-center justify-between gap-3">
            <p className="text-xs">&copy; {new Date().getFullYear()} AgriPrice Sentinel. All rights reserved.</p>
            <p className="text-xs">Built with ❤️ for Indian Farmers</p>
          </div>
        </div>
      </footer>
    </>
  );
}
