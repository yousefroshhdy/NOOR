import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { ThemeProvider } from "./contexts/ThemeContext";
import Auth from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import MapPage from "./pages/MapPage";
import VehicleDetailPage from "./pages/VehicleDetailPage";
import Alerts from "./pages/Alerts";
import Settings from "./pages/Settings";
import DriversMonitor from "./pages/DriversMonitor";
import NotFound from "./pages/NotFound";
import { isAuthenticated } from "./services/authService";
import { useMobileDetection } from "./hooks/useMobileDetection";
import { useEffect } from "react";
import "./App.css";

const queryClient = new QueryClient();

const App = () => {
  const { isNative } = useMobileDetection();

  useEffect(() => {
    // Mobile-specific optimizations
    if (isNative) {
      // Prevent zoom on double tap
      document.addEventListener('gesturestart', (e) => e.preventDefault());
      document.addEventListener('gesturechange', (e) => e.preventDefault());
      
      // Set mobile viewport meta tag
      const viewport = document.querySelector('meta[name=viewport]');
      if (viewport) {
        viewport.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no');
      }
    }
  }, [isNative]);

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <div className={`min-h-screen ${isNative ? 'pt-safe-top pb-safe-bottom' : ''}`}>
            <Toaster />
            <Sonner theme="system" />
            <BrowserRouter>
              <Routes>
                <Route path="/" element={
                  isAuthenticated() ? <Navigate to="/dashboard" /> : <Navigate to="/auth" />
                } />
                <Route path="/auth" element={<Auth />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/map" element={<MapPage />} />
                <Route path="/vehicle/:id" element={<VehicleDetailPage />} />
                <Route path="/alerts" element={<Alerts />} />
                <Route path="/monitor" element={<DriversMonitor />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </BrowserRouter>
          </div>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;
