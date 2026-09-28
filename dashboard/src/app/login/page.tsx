"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useLogin, useRegister } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Leaf, User, Lock, Phone } from "lucide-react";

export default function AuthPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");
  const router = useRouter();
  const { login } = useAuth();

  const loginMutation = useLogin();
  const registerMutation = useRegister();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg("");
    setSuccessMsg("");

    if (isLogin) {
      loginMutation.mutate(
        { phone, password },
        {
          onSuccess: (data) => {
            login(data.access_token, data.user || { id: 1, phone, full_name: "Farmer", created_at: "" });
            router.push("/profile");
          },
          onError: (err: any) => {
            setErrorMsg(err?.response?.data?.detail || "Login failed. Please check your credentials.");
          },
        }
      );
    } else {
      registerMutation.mutate(
        { phone, password, full_name: fullName },
        {
          onSuccess: () => {
            setSuccessMsg("Account created successfully! Please log in.");
            setIsLogin(true);
            setFullName("");
            setPassword("");
          },
          onError: (err: any) => {
            setErrorMsg(err?.response?.data?.detail || "Registration failed. Please try again.");
          },
        }
      );
    }
  };

  return (
    <div className="dark flex min-h-screen w-full items-center justify-center p-4 bg-slate-950 text-slate-100 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(16,185,129,0.15),rgba(255,255,255,0))]">
      <div className="w-full max-w-md">
        <div className="mb-8 flex flex-col items-center justify-center text-center">
          <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-500 shadow-lg shadow-emerald-500/10">
            <Leaf size={32} />
          </div>
          <h1 className="text-3xl font-bold tracking-tight text-white mb-2">AgriPrice Sentinel</h1>
          <p className="text-slate-400">Your AI-powered agricultural market intelligence.</p>
        </div>

        <Card className="w-full bg-slate-900/80 backdrop-blur-xl border-slate-800 shadow-2xl overflow-hidden transition-all duration-300">
          <div className="flex w-full border-b border-slate-800">
            <button
              type="button"
              onClick={() => { setIsLogin(true); setErrorMsg(""); setSuccessMsg(""); }}
              className={`flex-1 py-4 text-sm font-medium transition-colors ${
                isLogin
                  ? "text-emerald-500 border-b-2 border-emerald-500 bg-emerald-500/5"
                  : "text-slate-400 hover:text-slate-300 hover:bg-slate-800/50"
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => { setIsLogin(false); setErrorMsg(""); setSuccessMsg(""); }}
              className={`flex-1 py-4 text-sm font-medium transition-colors ${
                !isLogin
                  ? "text-emerald-500 border-b-2 border-emerald-500 bg-emerald-500/5"
                  : "text-slate-400 hover:text-slate-300 hover:bg-slate-800/50"
              }`}
            >
              Sign Up
            </button>
          </div>

          <CardHeader className="pb-4">
            <CardTitle className="text-2xl text-slate-100">
              {isLogin ? "Welcome back" : "Create an account"}
            </CardTitle>
            <CardDescription className="text-slate-400">
              {isLogin
                ? "Enter your credentials to access your dashboard."
                : "Join Sentinel to get AI price forecasts and alerts."}
            </CardDescription>
          </CardHeader>

          <form onSubmit={handleSubmit}>
            <CardContent className="space-y-4">
              {errorMsg && (
                <div className="rounded-md bg-red-500/10 border border-red-500/20 p-3 text-red-400 text-sm font-medium text-center">
                  {errorMsg}
                </div>
              )}
              {successMsg && (
                <div className="rounded-md bg-emerald-500/10 border border-emerald-500/20 p-3 text-emerald-400 text-sm font-medium text-center">
                  {successMsg}
                </div>
              )}

              <div className="space-y-4 transition-all duration-500">
                {!isLogin && (
                  <div className="space-y-2 animate-in fade-in slide-in-from-top-2">
                    <Label htmlFor="fullName" className="text-slate-300">Full Name</Label>
                    <div className="relative">
                      <User className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                      <Input
                        id="fullName"
                        type="text"
                        placeholder="Ramesh Kumar"
                        value={fullName}
                        onChange={(e) => setFullName(e.target.value)}
                        required={!isLogin}
                        className="pl-9 bg-slate-950/50 border-slate-800 text-slate-100 focus:border-emerald-500/50 focus:ring-emerald-500/50"
                      />
                    </div>
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="phone" className="text-slate-300">Phone Number</Label>
                  <div className="relative">
                    <Phone className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                    <Input
                      id="phone"
                      type="text"
                      placeholder="9876543210"
                      value={phone}
                      onChange={(e) => setPhone(e.target.value)}
                      required
                      className="pl-9 bg-slate-950/50 border-slate-800 text-slate-100 focus:border-emerald-500/50 focus:ring-emerald-500/50"
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="password" className="text-slate-300">Password</Label>
                    {isLogin && (
                      <a href="#" className="text-xs text-emerald-500 hover:text-emerald-400 transition-colors">
                        Forgot password?
                      </a>
                    )}
                  </div>
                  <div className="relative">
                    <Lock className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" />
                    <Input
                      id="password"
                      type="password"
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      required
                      minLength={6}
                      className="pl-9 bg-slate-950/50 border-slate-800 text-slate-100 focus:border-emerald-500/50 focus:ring-emerald-500/50"
                    />
                  </div>
                </div>
              </div>
            </CardContent>
            
            <CardFooter className="pt-2 pb-6">
              <Button
                type="submit"
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-600/20 transition-all h-11"
                disabled={loginMutation.isPending || registerMutation.isPending}
              >
                {loginMutation.isPending || registerMutation.isPending 
                  ? <span className="flex items-center gap-2">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-white" />
                      Processing...
                    </span>
                  : isLogin ? "Sign In" : "Create Account"
                }
              </Button>
            </CardFooter>
          </form>
        </Card>
      </div>
    </div>
  );
}
