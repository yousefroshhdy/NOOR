
import { useState, useEffect } from 'react';

// Extend the Window interface to include Capacitor
declare global {
  interface Window {
    Capacitor?: {
      isNativePlatform(): boolean;
      getPlatform(): string;
    };
  }
}

export const useMobileDetection = () => {
  const [isMobile, setIsMobile] = useState(false);
  const [isNative, setIsNative] = useState(false);

  useEffect(() => {
    // Check if running in Capacitor (native app)
    const checkNative = () => {
      if (window.Capacitor) {
        setIsNative(true);
        setIsMobile(true);
      } else {
        // Fallback to user agent detection
        const userAgent = navigator.userAgent.toLowerCase();
        const mobileKeywords = ['android', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone'];
        const isMobileDevice = mobileKeywords.some(keyword => userAgent.includes(keyword));
        setIsMobile(isMobileDevice);
      }
    };

    checkNative();
  }, []);

  return { isMobile, isNative };
};
