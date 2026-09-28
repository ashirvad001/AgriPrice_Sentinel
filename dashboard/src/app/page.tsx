"use client";

import Navbar from "@/components/landing/Navbar";
import HeroSection from "@/components/landing/HeroSection";
import FeaturesSection from "@/components/landing/FeaturesSection";
import HowItWorks from "@/components/landing/HowItWorks";
import PredictionSection from "@/components/landing/PredictionSection";
import DashboardPreview from "@/components/landing/DashboardPreview";
import UsersAndTestimonials from "@/components/landing/UsersAndTestimonials";
import CTAAndFooter from "@/components/landing/CTAAndFooter";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white dark:bg-slate-950 text-gray-900 dark:text-white transition-colors duration-300">
      <Navbar />
      <HeroSection />
      <FeaturesSection />
      <HowItWorks />
      <PredictionSection />
      <DashboardPreview />
      <UsersAndTestimonials />
      <CTAAndFooter />
    </div>
  );
}
