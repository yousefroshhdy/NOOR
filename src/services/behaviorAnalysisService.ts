
import { Vehicle, DrivingBehavior } from './mockData';

// Simulate a behavior analysis algorithm
export const analyzeDrivingBehavior = (vehicle: Vehicle): {
  behavior: DrivingBehavior,
  confidence: number,
  details: string
} => {
  // Check if vehicle is driving
  if (vehicle.status !== 'driving' || !vehicle.lastPosition || !vehicle.lastPosition.speed) {
    return {
      behavior: 'normal',
      confidence: 0.98,
      details: 'Vehicle is not in motion'
    };
  }

  // Factors that might indicate abnormal behavior
  const speed = vehicle.lastPosition.speed;
  const time = new Date().getHours();
  const random = Math.random();
  
  // For demo purposes, we'll use a combination of speed, time of day, and randomness
  // to simulate different behaviors
  
  // Simulate aggressive driving based on speed
  if (speed > 65 && random > 0.7) {
    return {
      behavior: 'aggressive',
      confidence: 0.85 + (random * 0.1),
      details: 'High speed detected with erratic movements'
    };
  }
  
  // Simulate distracted driving based on time of day and randomness
  if ((time > 22 || time < 6) && random > 0.8) {
    return {
      behavior: 'distracted',
      confidence: 0.75 + (random * 0.15),
      details: 'Irregular lane positioning detected'
    };
  }
  
  // Simulate tired driving during night hours
  if ((time > 0 && time < 5) && random > 0.75) {
    return {
      behavior: 'tired',
      confidence: 0.82 + (random * 0.1),
      details: 'Reduced reaction time patterns detected'
    };
  }
  
  // Simulate speeding
  if (speed > 75 && random > 0.6) {
    return {
      behavior: 'speeding',
      confidence: 0.9 + (random * 0.08),
      details: 'Vehicle exceeding speed limit by 20+ mph'
    };
  }
  
  // Default to normal behavior
  return {
    behavior: 'normal',
    confidence: 0.95 + (random * 0.05),
    details: 'No concerning patterns detected'
  };
};

// Simulate endpoint call to /analyze-driving
export const fetchBehaviorAnalysis = async (vehicleId: string, position: any): Promise<{
  vehicleId: string,
  behavior: DrivingBehavior,
  confidence: number,
  details: string
}> => {
  // Simulate network latency
  await new Promise(resolve => setTimeout(resolve, 300));
  
  // Get vehicle data (in a real app, we'd send the position data to the backend)
  const mockVehicles = await import('./mockData').then(m => m.mockVehicles);
  const vehicle = mockVehicles.find(v => v.id === vehicleId);
  
  if (!vehicle) {
    throw new Error('Vehicle not found');
  }
  
  // Analyze behavior
  const analysis = analyzeDrivingBehavior(vehicle);
  
  return {
    vehicleId,
    ...analysis
  };
};
