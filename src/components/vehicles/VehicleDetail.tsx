
import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Vehicle, DrivingSession, DrivingEvent } from '@/services/mockData';
import { getVehicleById, getVehicleSessions, getVehicleEvents, subscribeToVehicleUpdates } from '@/services/vehicleService';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { formatDate, formatTime, formatSpeed, formatBatteryLevel, formatDrivingScore } from '@/utils/formatters';
import { Calendar, Clock, Navigation, Battery, Gauge, AlertTriangle, Car } from 'lucide-react';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

const VehicleDetail = () => {
  const { id = '' } = useParams<{ id: string }>();
  const [vehicle, setVehicle] = useState<Vehicle | null>(null);
  const [sessions, setSessions] = useState<DrivingSession[]>([]);
  const [events, setEvents] = useState<DrivingEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchVehicleData = async () => {
      setIsLoading(true);
      try {
        // Fetch vehicle details
        const vehicleData = await getVehicleById(id);
        if (vehicleData) {
          setVehicle(vehicleData);
          
          // Fetch sessions and events
          const sessionsData = await getVehicleSessions(id);
          setSessions(sessionsData);
          
          const eventsData = await getVehicleEvents(id);
          setEvents(eventsData);
        }
      } catch (error) {
        console.error('Failed to fetch vehicle data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchVehicleData();

    // Subscribe to real-time updates
    const unsubscribe = subscribeToVehicleUpdates((updatedVehicle) => {
      if (updatedVehicle.id === id) {
        setVehicle(updatedVehicle);
      }
    });

    return () => {
      unsubscribe();
    };
  }, [id]);

  if (isLoading) {
    return (
      <div className="flex flex-col space-y-4 animate-pulse">
        <div className="h-40 bg-gray-200 rounded-lg" />
        <div className="h-20 bg-gray-200 rounded-lg" />
        <div className="h-60 bg-gray-200 rounded-lg" />
      </div>
    );
  }

  if (!vehicle) {
    return (
      <div className="flex flex-col items-center justify-center py-12 space-y-4">
        <div className="p-4 bg-red-100 rounded-full">
          <AlertTriangle className="h-8 w-8 text-red-500" />
        </div>
        <h2 className="text-xl font-semibold">Vehicle Not Found</h2>
        <p className="text-gray-500">The vehicle you're looking for doesn't exist or has been removed.</p>
        <Button asChild variant="outline">
          <Link to="/dashboard">Return to Dashboard</Link>
        </Button>
      </div>
    );
  }

  // Get the latest session (either ongoing or most recent)
  const latestSession = sessions.sort((a, b) => b.startTime - a.startTime)[0];
  
  // Sort events by timestamp (newest first)
  const sortedEvents = [...events].sort((a, b) => b.timestamp - a.timestamp);
  
  // Generate status badge
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'driving':
        return <Badge className="bg-green-500">Driving</Badge>;
      case 'parked':
        return <Badge className="bg-blue-500">Parked</Badge>;
      case 'offline':
        return <Badge className="bg-gray-500">Offline</Badge>;
      default:
        return <Badge className="bg-gray-500">Unknown</Badge>;
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

  // Format score
  const score = formatDrivingScore(vehicle.drivingScore);

  return (
    <div className="space-y-6 pb-20">
      {/* Vehicle Header */}
      <div className="flex flex-col sm:flex-row gap-4 items-start">
        <div className="h-24 w-24 rounded-lg overflow-hidden bg-gray-100 flex-shrink-0">
          {vehicle.image ? (
            <img 
              src={vehicle.image} 
              alt={`${vehicle.make} ${vehicle.model}`} 
              className="w-full h-full object-cover"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-200">
              <Car className="h-12 w-12 text-gray-400" />
            </div>
          )}
        </div>
        
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold">{vehicle.name}</h1>
            {getStatusBadge(vehicle.status)}
          </div>
          <p className="text-gray-500">
            {vehicle.make} {vehicle.model} • {vehicle.year}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            License: {vehicle.licensePlate}
          </p>
          
          <div className="flex gap-3 mt-3">
            <Button asChild size="sm" variant="outline">
              <Link to={`/map?vehicle=${vehicle.id}`}>
                <Navigation className="mr-1 h-4 w-4" />
                Track
              </Link>
            </Button>
            <Button size="sm" variant="outline">
              <AlertTriangle className="mr-1 h-4 w-4" />
              Alert History
            </Button>
          </div>
        </div>
      </div>
      
      <Separator />
      
      {/* Tab Navigation */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid grid-cols-3 w-full">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="trips">Trips</TabsTrigger>
          <TabsTrigger value="events">Events</TabsTrigger>
        </TabsList>
        
        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="pt-4 px-4 flex flex-col items-center">
                <Battery className="h-5 w-5 text-gray-500 mb-1" />
                <p className="text-xs text-gray-500">Fuel Level</p>
                <p className="text-lg font-medium">{formatBatteryLevel(vehicle.fuelLevel)}</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-4 px-4 flex flex-col items-center">
                <Gauge className="h-5 w-5 text-gray-500 mb-1" />
                <p className="text-xs text-gray-500">Driving Score</p>
                <p className={`text-lg font-medium ${score.color}`}>{score.text}</p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-4 px-4 flex flex-col items-center">
                <Navigation className="h-5 w-5 text-gray-500 mb-1" />
                <p className="text-xs text-gray-500">Speed</p>
                <p className="text-lg font-medium">
                  {vehicle.status === 'driving' && vehicle.lastPosition?.speed
                    ? formatSpeed(vehicle.lastPosition.speed)
                    : 'N/A'}
                </p>
              </CardContent>
            </Card>
            
            <Card>
              <CardContent className="pt-4 px-4 flex flex-col items-center">
                <Clock className="h-5 w-5 text-gray-500 mb-1" />
                <p className="text-xs text-gray-500">Last Updated</p>
                <p className="text-lg font-medium">
                  {vehicle.lastPosition
                    ? formatTime(vehicle.lastPosition.timestamp)
                    : 'N/A'}
                </p>
              </CardContent>
            </Card>
          </div>
          
          {/* Current/Latest Trip */}
          {latestSession && (
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">
                  {!latestSession.endTime ? 'Current Trip' : 'Latest Trip'}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-gray-500" />
                      <span className="text-sm text-gray-500">Date</span>
                    </div>
                    <span className="font-medium">
                      {formatDate(latestSession.startTime)}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-gray-500" />
                      <span className="text-sm text-gray-500">Time</span>
                    </div>
                    <span className="font-medium">
                      {formatTime(latestSession.startTime)}
                      {latestSession.endTime && ` - ${formatTime(latestSession.endTime)}`}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                      <Navigation className="h-4 w-4 text-gray-500" />
                      <span className="text-sm text-gray-500">Distance</span>
                    </div>
                    <span className="font-medium">
                      {latestSession.distance ? `${latestSession.distance} mi` : 'In progress'}
                    </span>
                  </div>
                  
                  {latestSession.averageSpeed && (
                    <div className="flex justify-between items-center">
                      <div className="flex items-center gap-2">
                        <Gauge className="h-4 w-4 text-gray-500" />
                        <span className="text-sm text-gray-500">Avg. Speed</span>
                      </div>
                      <span className="font-medium">
                        {formatSpeed(latestSession.averageSpeed)}
                      </span>
                    </div>
                  )}
                  
                  {latestSession.events.length > 0 && (
                    <div className="mt-3 pt-3 border-t">
                      <p className="text-sm text-gray-500 mb-2">Events</p>
                      <div className="space-y-2">
                        {latestSession.events.slice(0, 3).map((event) => (
                          <div key={event.id} className="flex justify-between items-center">
                            <span className="text-sm">{event.details || event.behavior}</span>
                            {getBehaviorBadge(event.behavior, event.severity)}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
        
        {/* Trips Tab */}
        <TabsContent value="trips">
          {sessions.length > 0 ? (
            <div className="space-y-4">
              {sessions.map((session) => (
                <Card key={session.id}>
                  <CardContent className="pt-6">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <p className="font-medium">
                          {formatDate(session.startTime)} • {formatTime(session.startTime)}
                        </p>
                        <p className="text-sm text-gray-500">
                          {session.distance ? `${session.distance} mi` : 'In progress'}
                          {session.averageSpeed && ` • Avg ${formatSpeed(session.averageSpeed)}`}
                        </p>
                      </div>
                      {session.score && (
                        <div className={`text-lg font-bold ${session.score >= 90 ? 'text-green-500' : 
                          session.score >= 70 ? 'text-yellow-500' : 'text-red-500'}`}>
                          {session.score}
                        </div>
                      )}
                    </div>
                    
                    {session.events.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs text-gray-500 mb-1">Events</p>
                        <div className="flex flex-wrap gap-2">
                          {session.events.map((event) => (
                            <div key={event.id}>
                              {getBehaviorBadge(event.behavior, event.severity)}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-10">
              <p className="text-gray-500">No trips recorded yet</p>
            </div>
          )}
        </TabsContent>
        
        {/* Events Tab */}
        <TabsContent value="events">
          {sortedEvents.length > 0 ? (
            <div className="space-y-3">
              {sortedEvents.map((event) => (
                <Card key={event.id}>
                  <CardContent className="pt-6">
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          {getBehaviorBadge(event.behavior, event.severity)}
                          <span className="text-sm text-gray-500">
                            {formatDate(event.timestamp)} • {formatTime(event.timestamp)}
                          </span>
                        </div>
                        <p className="text-sm">{event.details || 'No details available'}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-10">
              <p className="text-gray-500">No events recorded</p>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default VehicleDetail;
