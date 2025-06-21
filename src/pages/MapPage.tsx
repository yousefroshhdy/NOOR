
import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { isAuthenticated } from '@/services/authService';
import AppLayout from '@/components/layout/AppLayout';
import MapView from '@/components/map/MapView';
import { useTheme } from '@/contexts/ThemeContext';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { AlertTriangle } from 'lucide-react';

const MapPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [title, setTitle] = useState('Live Map');
  const { theme } = useTheme();
  const [showAlert, setShowAlert] = useState(false);
  const [alertData, setAlertData] = useState({ vehicleId: '', behavior: '', details: '' });

  // Check if user is authenticated
  useEffect(() => {
    if (!isAuthenticated()) {
      navigate('/auth');
    }

    // Check if a specific vehicle is being tracked
    const vehicleId = searchParams.get('vehicle');
    if (vehicleId) {
      setTitle('Tracking Vehicle');
    }
  }, [navigate, searchParams]);

  // Handle behavior alerts
  const handleBehaviorAlert = (vehicleId: string, behavior: string, details: string) => {
    setAlertData({ vehicleId, behavior, details });
    setShowAlert(true);
    
    // Auto-hide alert after 5 seconds
    setTimeout(() => {
      setShowAlert(false);
    }, 5000);
  };

  return (
    <AppLayout title={title}>
      <div className={`max-w-6xl mx-auto ${theme === 'dark' ? 'text-white' : ''}`}>
        <h1 className="text-2xl font-bold mb-4">Live Map</h1>
        
        {showAlert && (
          <Alert 
            variant="destructive" 
            className="mb-4 border-red-400 dark:border-red-600 bg-red-50 dark:bg-red-900/20"
          >
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Driving Behavior Alert</AlertTitle>
            <AlertDescription>
              Vehicle {alertData.vehicleId} detected with {alertData.behavior} behavior. {alertData.details}
            </AlertDescription>
          </Alert>
        )}
        
        <MapView onBehaviorAlert={handleBehaviorAlert} />
      </div>
    </AppLayout>
  );
};

export default MapPage;
