
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { isAuthenticated } from '@/services/authService';
import AppLayout from '@/components/layout/AppLayout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { mockDrivingEvents } from '@/services/mockData';
import { formatDate, formatTime } from '@/utils/formatters';
import { Bell, AlertTriangle, Info, Car } from 'lucide-react';

const Alerts = () => {
  const navigate = useNavigate();

  // Check if user is authenticated
  useEffect(() => {
    if (!isAuthenticated()) {
      navigate('/auth');
    }
  }, [navigate]);

  // Get sorted events (newest first)
  const sortedEvents = [...mockDrivingEvents].sort((a, b) => b.timestamp - a.timestamp);

  // Get behavior icon
  const getBehaviorIcon = (behavior: string, severity: string) => {
    switch (behavior) {
      case 'aggressive':
      case 'speeding':
        return <AlertTriangle className={`h-5 w-5 ${severity === 'high' ? 'text-red-500' : severity === 'medium' ? 'text-orange-500' : 'text-yellow-500'}`} />;
      case 'distracted':
      case 'tired':
        return <Info className={`h-5 w-5 ${severity === 'high' ? 'text-red-500' : severity === 'medium' ? 'text-orange-500' : 'text-yellow-500'}`} />;
      default:
        return <Bell className="h-5 w-5 text-gray-500" />;
    }
  };

  // Get behavior badge
  const getBehaviorBadge = (behavior: string, severity: string) => {
    switch (behavior) {
      case 'aggressive':
        return <Badge className={severity === 'high' ? 'bg-red-500' : severity === 'medium' ? 'bg-orange-500' : 'bg-yellow-500'}>Aggressive Driving</Badge>;
      case 'distracted':
        return <Badge className={severity === 'high' ? 'bg-red-500' : severity === 'medium' ? 'bg-orange-500' : 'bg-yellow-500'}>Distracted</Badge>;
      case 'speeding':
        return <Badge className={severity === 'high' ? 'bg-red-500' : severity === 'medium' ? 'bg-orange-500' : 'bg-yellow-500'}>Speeding</Badge>;
      case 'tired':
        return <Badge className={severity === 'high' ? 'bg-red-500' : severity === 'medium' ? 'bg-orange-500' : 'bg-yellow-500'}>Driver Fatigue</Badge>;
      case 'normal':
        return <Badge className="bg-green-500">Normal Driving</Badge>;
      default:
        return <Badge className="bg-gray-500">{behavior}</Badge>;
    }
  };

  return (
    <AppLayout title="Alerts">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold mb-4">Alert Center</h1>
        
        <Card className="mb-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg">Recent Driving Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-gray-500 mb-4">
              Monitoring alerts from all vehicles in your fleet
            </p>
            
            <div className="space-y-4">
              {sortedEvents.length > 0 ? (
                sortedEvents.map((event) => (
                  <div 
                    key={event.id} 
                    className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                    onClick={() => navigate(`/vehicle/${event.vehicleId}`)}
                  >
                    <div className="mt-1">
                      {getBehaviorIcon(event.behavior, event.severity)}
                    </div>
                    
                    <div className="flex-1">
                      <div className="flex flex-wrap items-center gap-2 mb-1">
                        {getBehaviorBadge(event.behavior, event.severity)}
                        <span className="text-sm text-gray-500">
                          {formatDate(event.timestamp)} • {formatTime(event.timestamp)}
                        </span>
                      </div>
                      
                      <p className="text-sm">{event.details || 'No details available'}</p>
                    </div>
                    
                    <div className="flex items-center gap-1 text-xs text-gray-500">
                      <Car className="h-3 w-3" />
                      <span>
                        {
                          mockDrivingEvents.find(e => e.vehicleId === event.vehicleId) 
                            ? 'Work Sedan' 
                            : 'Unknown Vehicle'
                        }
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <Bell className="h-8 w-8 text-gray-300 mx-auto mb-2" />
                  <p className="text-gray-500">No alerts to display</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  );
};

export default Alerts;
