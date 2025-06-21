
import React, { useState, useEffect, useCallback } from 'react';
import { vehicleService } from '@/services/supabaseVehicleService';
import { useToast } from '@/hooks/use-toast';
import { Smartphone, MapPin, Play, Square, Navigation, Map } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { supabase } from '@/integrations/supabase/client';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface MobileGPSTrackerProps {
  onPositionUpdate?: (position: { lat: number; lng: number; speed?: number }) => void;
}

interface GPSPosition {
  latitude: number;
  longitude: number;
  speed: number | null;
  heading: number | null;
  accuracy: number;
  timestamp: number;
}

const MobileGPSTracker: React.FC<MobileGPSTrackerProps> = ({ onPositionUpdate }) => {
  const [isTracking, setIsTracking] = useState(false);
  const [currentPosition, setCurrentPosition] = useState<GPSPosition | null>(null);
  const [selectedVehicleId, setSelectedVehicleId] = useState<string>('');
  const [vehicles, setVehicles] = useState<any[]>([]);
  const [trackingSession, setTrackingSession] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isCreatingVehicle, setIsCreatingVehicle] = useState(false);
  const [newVehicleName, setNewVehicleName] = useState('My Mobile Device');
  
  // Manual coordinates input
  const [manualLat, setManualLat] = useState('');
  const [manualLng, setManualLng] = useState('');
  const [manualSpeed, setManualSpeed] = useState('0');
  const [useManualMode, setUseManualMode] = useState(false);
  
  const { toast } = useToast();
  const watchIdRef = React.useRef<number | null>(null);

  // Load vehicles on component mount
  useEffect(() => {
    loadVehicles();
  }, []);

  const loadVehicles = async () => {
    try {
      const vehicleList = await vehicleService.getVehicles();
      setVehicles(vehicleList);
      
      // Auto-select first vehicle if none selected
      if (vehicleList.length > 0 && !selectedVehicleId) {
        setSelectedVehicleId(vehicleList[0].id);
      }
    } catch (err) {
      console.error('Failed to load vehicles:', err);
      setError('Failed to load vehicles');
    }
  };

  const createMobileVehicle = async () => {
    try {
      setIsCreatingVehicle(true);
      const deviceId = `mobile-${Date.now()}`;
      
      // Get current user
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setError('User not authenticated');
        return;
      }
      
      const vehicle = await vehicleService.createVehicle({
        name: newVehicleName,
        make: 'Mobile',
        model: 'GPS Tracker',
        year: new Date().getFullYear(),
        device_id: deviceId,
        license_plate: `MOB-${deviceId.slice(-4).toUpperCase()}`,
        user_id: user.id
      });
      
      setSelectedVehicleId(vehicle.id);
      await loadVehicles();
      
      toast({
        title: 'Vehicle Created',
        description: `Mobile vehicle "${newVehicleName}" has been created successfully.`,
      });
    } catch (err) {
      console.error('Failed to create vehicle:', err);
      setError('Failed to create mobile vehicle');
    } finally {
      setIsCreatingVehicle(false);
    }
  };

  const updatePositionFromManual = useCallback(async () => {
    const lat = parseFloat(manualLat);
    const lng = parseFloat(manualLng);
    const speed = parseFloat(manualSpeed);
    
    if (isNaN(lat) || isNaN(lng)) {
      setError('Please enter valid latitude and longitude');
      return;
    }
    
    const gpsPosition: GPSPosition = {
      latitude: lat,
      longitude: lng,
      speed: isNaN(speed) ? 0 : speed,
      heading: 0,
      accuracy: 10, // Assume 10m accuracy for manual input
      timestamp: Date.now()
    };

    setCurrentPosition(gpsPosition);
    setError(null);

    // Update position in database
    if (selectedVehicleId) {
      try {
        await vehicleService.updateVehiclePosition(selectedVehicleId, {
          latitude: gpsPosition.latitude,
          longitude: gpsPosition.longitude,
          speed: gpsPosition.speed || 0,
          heading: gpsPosition.heading || 0,
          accuracy: gpsPosition.accuracy,
          altitude: 0
        });

        // Notify parent component
        onPositionUpdate?.(
          { 
            lat: gpsPosition.latitude, 
            lng: gpsPosition.longitude, 
            speed: gpsPosition.speed || undefined 
          }
        );
        
        toast({
          title: 'Position Updated',
          description: 'Manual position has been set successfully.',
        });
      } catch (err) {
        console.error('Failed to update position:', err);
        setError('Failed to update position in database');
      }
    }
  }, [selectedVehicleId, manualLat, manualLng, manualSpeed, onPositionUpdate, toast]);

  const updatePosition = useCallback(async (position: GeolocationPosition) => {
    const gpsPosition: GPSPosition = {
      latitude: position.coords.latitude,
      longitude: position.coords.longitude,
      speed: position.coords.speed,
      heading: position.coords.heading,
      accuracy: position.coords.accuracy,
      timestamp: Date.now()
    };

    setCurrentPosition(gpsPosition);
    setError(null);

    // Update position in database
    if (selectedVehicleId) {
      try {
        await vehicleService.updateVehiclePosition(selectedVehicleId, {
          latitude: gpsPosition.latitude,
          longitude: gpsPosition.longitude,
          speed: gpsPosition.speed || 0,
          heading: gpsPosition.heading || 0,
          accuracy: gpsPosition.accuracy,
          altitude: position.coords.altitude || 0
        });

        // Notify parent component
        onPositionUpdate?.(
          { 
            lat: gpsPosition.latitude, 
            lng: gpsPosition.longitude, 
            speed: gpsPosition.speed || undefined 
          }
        );
      } catch (err) {
        console.error('Failed to update position:', err);
      }
    }
  }, [selectedVehicleId, onPositionUpdate]);

  const handleLocationError = useCallback((error: GeolocationPositionError) => {
    let errorMessage = 'Location access failed';
    
    switch (error.code) {
      case error.PERMISSION_DENIED:
        errorMessage = 'Location access denied. Please allow location permissions or use manual mode.';
        break;
      case error.POSITION_UNAVAILABLE:
        errorMessage = 'Location information unavailable. Try manual mode.';
        break;
      case error.TIMEOUT:
        errorMessage = 'Location request timed out. Try manual mode.';
        break;
    }
    
    setError(errorMessage);
    console.error('GPS Error:', error);
  }, []);

  const startTracking = useCallback(async () => {
    if (!selectedVehicleId) {
      setError('Please select a vehicle first');
      return;
    }

    if (useManualMode) {
      // Start manual tracking session
      try {
        const session = await vehicleService.startDrivingSession(selectedVehicleId);
        setTrackingSession(session.id);
        setIsTracking(true);
        setError(null);
        
        toast({
          title: 'Manual Tracking Started',
          description: 'You can now update your position manually.',
        });
      } catch (err) {
        console.error('Failed to start tracking:', err);
        setError('Failed to start tracking session');
      }
      return;
    }

    if (!navigator.geolocation) {
      setError('Geolocation is not supported by this browser. Use manual mode instead.');
      return;
    }

    try {
      // Start driving session
      const session = await vehicleService.startDrivingSession(selectedVehicleId);
      setTrackingSession(session.id);
      setIsTracking(true);
      setError(null);

      // Start watching position
      watchIdRef.current = navigator.geolocation.watchPosition(
        updatePosition,
        handleLocationError,
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 1000
        }
      );

      toast({
        title: 'GPS Tracking Started',
        description: 'Your location is now being tracked.',
      });
    } catch (err) {
      console.error('Failed to start tracking:', err);
      setError('Failed to start tracking session');
    }
  }, [selectedVehicleId, useManualMode, updatePosition, handleLocationError, toast]);

  const stopTracking = useCallback(async () => {
    if (watchIdRef.current !== null) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }

    setIsTracking(false);
    
    // End driving session
    if (trackingSession) {
      try {
        await vehicleService.endDrivingSession(trackingSession, {
          total_distance: 0,
          max_speed: currentPosition?.speed || 0,
          avg_speed: currentPosition?.speed || 0
        });
        setTrackingSession(null);
      } catch (err) {
        console.error('Failed to end session:', err);
      }
    }

    toast({
      title: 'Tracking Stopped',
      description: 'Location tracking has been stopped.',
    });
  }, [trackingSession, currentPosition, toast]);

  useEffect(() => {
    return () => {
      if (watchIdRef.current !== null) {
        navigator.geolocation.clearWatch(watchIdRef.current);
      }
    };
  }, []);

  const formatSpeed = (speed: number | null) => {
    if (speed === null) return 'N/A';
    return `${Math.round(speed * 2.237)} mph`; // Convert m/s to mph
  };

  const formatCoordinate = (coord: number) => {
    return coord.toFixed(6);
  };

  const getGoogleMapsLink = () => {
    if (!currentPosition) return '';
    return `https://www.google.com/maps?q=${currentPosition.latitude},${currentPosition.longitude}`;
  };

  return (
    <Card className="w-full max-w-md">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Smartphone className="h-5 w-5" />
          Mobile GPS Tracker
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Vehicle Selection */}
        <div className="space-y-2">
          <Label>Select Vehicle</Label>
          {vehicles.length > 0 ? (
            <select
              value={selectedVehicleId}
              onChange={(e) => setSelectedVehicleId(e.target.value)}
              className="w-full p-2 border rounded-md"
              disabled={isTracking}
            >
              <option value="">Select a vehicle...</option>
              {vehicles.map((vehicle) => (
                <option key={vehicle.id} value={vehicle.id}>
                  {vehicle.name} ({vehicle.make} {vehicle.model})
                </option>
              ))}
            </select>
          ) : (
            <div className="space-y-2">
              <Input
                value={newVehicleName}
                onChange={(e) => setNewVehicleName(e.target.value)}
                placeholder="Enter vehicle name"
                disabled={isCreatingVehicle}
              />
              <Button 
                onClick={createMobileVehicle} 
                disabled={isCreatingVehicle}
                className="w-full"
              >
                Create Mobile Vehicle
              </Button>
            </div>
          )}
        </div>

        {/* Tracking Mode Tabs */}
        <Tabs value={useManualMode ? "manual" : "auto"} onValueChange={(value) => setUseManualMode(value === "manual")}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="auto">Auto GPS</TabsTrigger>
            <TabsTrigger value="manual">Manual</TabsTrigger>
          </TabsList>
          
          <TabsContent value="auto" className="space-y-2">
            <p className="text-sm text-gray-600">Uses device GPS for automatic location tracking</p>
          </TabsContent>
          
          <TabsContent value="manual" className="space-y-2">
            <p className="text-sm text-gray-600">Enter coordinates manually from Google Maps</p>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <Label htmlFor="lat">Latitude</Label>
                <Input
                  id="lat"
                  value={manualLat}
                  onChange={(e) => setManualLat(e.target.value)}
                  placeholder="e.g. 40.7128"
                  disabled={isTracking}
                />
              </div>
              <div>
                <Label htmlFor="lng">Longitude</Label>
                <Input
                  id="lng"
                  value={manualLng}
                  onChange={(e) => setManualLng(e.target.value)}
                  placeholder="e.g. -74.0060"
                  disabled={isTracking}
                />
              </div>
            </div>
            <div>
              <Label htmlFor="speed">Speed (mph)</Label>
              <Input
                id="speed"
                value={manualSpeed}
                onChange={(e) => setManualSpeed(e.target.value)}
                placeholder="e.g. 25"
                disabled={isTracking}
              />
            </div>
            {useManualMode && isTracking && (
              <Button 
                onClick={updatePositionFromManual} 
                className="w-full"
                variant="outline"
              >
                <MapPin className="mr-2 h-4 w-4" />
                Update Position
              </Button>
            )}
          </TabsContent>
        </Tabs>

        {/* Tracking Controls */}
        <div className="flex gap-2">
          {!isTracking ? (
            <Button 
              onClick={startTracking} 
              disabled={!selectedVehicleId || (useManualMode && (!manualLat || !manualLng))}
              className="flex-1"
            >
              <Play className="mr-2 h-4 w-4" />
              Start Tracking
            </Button>
          ) : (
            <Button 
              onClick={stopTracking} 
              variant="destructive" 
              className="flex-1"
            >
              <Square className="mr-2 h-4 w-4" />
              Stop Tracking
            </Button>
          )}
        </div>

        {/* Status */}
        <div className="flex justify-between items-center">
          <span className="text-sm font-medium">Status:</span>
          <Badge variant={isTracking ? "default" : "secondary"}>
            {isTracking ? `Tracking (${useManualMode ? 'Manual' : 'Auto'})` : 'Stopped'}
          </Badge>
        </div>

        {/* Position Info */}
        {currentPosition && (
          <div className="space-y-2 p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <MapPin className="h-4 w-4" />
                <span className="font-medium">Current Position</span>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={() => window.open(getGoogleMapsLink(), '_blank')}
              >
                <Map className="h-3 w-3 mr-1" />
                View
              </Button>
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-gray-600">Lat:</span>
                <span className="ml-1">{formatCoordinate(currentPosition.latitude)}</span>
              </div>
              <div>
                <span className="text-gray-600">Lng:</span>
                <span className="ml-1">{formatCoordinate(currentPosition.longitude)}</span>
              </div>
              <div>
                <span className="text-gray-600">Speed:</span>
                <span className="ml-1">{formatSpeed(currentPosition.speed)}</span>
              </div>
              <div>
                <span className="text-gray-600">Accuracy:</span>
                <span className="ml-1">{Math.round(currentPosition.accuracy)}m</span>
              </div>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}

        {/* Instructions */}
        <div className="text-xs text-gray-500 space-y-1">
          <p><strong>Auto GPS:</strong></p>
          <p>• Allow location permissions when prompted</p>
          <p>• Keep this tab active for best accuracy</p>
          <p><strong>Manual Mode:</strong></p>
          <p>• Get coordinates from Google Maps on your phone</p>
          <p>• Long press on map → coordinates appear</p>
        </div>
      </CardContent>
    </Card>
  );
};

export default MobileGPSTracker;
