
import { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'app.lovable.b3e6d745f15747f38ff98d3ef7138460',
  appName: 'drivepulse-tracker',
  webDir: 'dist',
  server: {
    url: 'https://b3e6d745-f157-47f3-8ff9-8d3ef7138460.lovableproject.com?forceHideBadge=true',
    cleartext: true
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: "#1a1a1a",
      showSpinner: false
    }
  }
};

export default config;
