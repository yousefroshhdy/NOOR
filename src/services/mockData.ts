
import { v4 as uuidv4 } from 'uuid';

// User Types
export interface User {
  id: string;
  name: string;
  email: string;
  avatar?: string;
}

// Vehicle Types
export type VehicleStatus = 'driving' | 'parked' | 'offline';
export type DrivingBehavior = 'normal' | 'aggressive' | 'distracted' | 'tired' | 'speeding';

export interface Position {
  lat: number;
  lng: number;
  speed?: number;
  heading?: number;
  timestamp: number;
}

export interface DrivingEvent {
  id: string;
  vehicleId: string;
  behavior: DrivingBehavior;
  severity: 'low' | 'medium' | 'high';
  timestamp: number;
  position: Position;
  details?: string;
}

export interface Vehicle {
  id: string;
  ownerId: string;
  name: string;
  licensePlate: string;
  make: string;
  model: string;
  year: number;
  color: string;
  status: VehicleStatus;
  lastPosition?: Position;
  fuelLevel?: number;
  drivingScore?: number;
  image?: string;
}

export interface DrivingSession {
  id: string;
  vehicleId: string;
  startTime: number;
  endTime?: number;
  startPosition: Position;
  endPosition?: Position;
  distance?: number;
  averageSpeed?: number;
  maxSpeed?: number;
  events: DrivingEvent[];
  score?: number;
}

// Mock user data
export const currentUser: User = {
  id: 'user-1',
  name: 'Alex Johnson',
  email: 'alex@example.com',
  avatar: 'https://i.pravatar.cc/150?img=11'
};

// Mock vehicle data
export const mockVehicles: Vehicle[] = [
  {
    id: 'v-1',
    ownerId: 'user-1',
    name: 'Work Sedan',
    licensePlate: 'ABC123',
    make: 'Toyota',
    model: 'Camry',
    year: 2021,
    color: 'Silver',
    status: 'driving',
    lastPosition: {
      lat: 37.773972,
      lng: -122.431297,
      speed: 35,
      heading: 90,
      timestamp: Date.now() - 120000
    },
    fuelLevel: 0.7,
    drivingScore: 87,
    image: 'https://images.unsplash.com/photo-1550355291-bbee04a92027?q=80&w=256&h=256&auto=format'
  },
  {
    id: 'v-2',
    ownerId: 'user-1',
    name: 'Family SUV',
    licensePlate: 'XYZ789',
    make: 'Honda',
    model: 'CR-V',
    year: 2020,
    color: 'Blue',
    status: 'parked',
    lastPosition: {
      lat: 37.783972,
      lng: -122.411297,
      speed: 0,
      heading: 0,
      timestamp: Date.now() - 3600000
    },
    fuelLevel: 0.4,
    drivingScore: 92,
    image: 'https://images.unsplash.com/photo-1533473359331-0135ef1b58bf?q=80&w=256&h=256&auto=format'
  },
  {
    id: 'v-3',
    ownerId: 'user-1',
    name: 'Delivery Van',
    licensePlate: 'LMN456',
    make: 'Ford',
    model: 'Transit',
    year: 2019,
    color: 'White',
    status: 'offline',
    lastPosition: {
      lat: 37.753972,
      lng: -122.451297,
      speed: 0,
      heading: 0,
      timestamp: Date.now() - 86400000
    },
    fuelLevel: 0.1,
    drivingScore: 65,
    image: 'https://images.unsplash.com/photo-1566912914645-206ac5bd0e48?q=80&w=256&h=256&auto=format'
  }
];

// Mock driving events data
export const mockDrivingEvents: DrivingEvent[] = [
  {
    id: 'e-1',
    vehicleId: 'v-1',
    behavior: 'aggressive',
    severity: 'medium',
    timestamp: Date.now() - 1800000,
    position: {
      lat: 37.763972,
      lng: -122.421297,
      timestamp: Date.now() - 1800000
    },
    details: 'Harsh acceleration detected'
  },
  {
    id: 'e-2',
    vehicleId: 'v-1',
    behavior: 'speeding',
    severity: 'high',
    timestamp: Date.now() - 3600000,
    position: {
      lat: 37.753972,
      lng: -122.431297,
      timestamp: Date.now() - 3600000
    },
    details: 'Driving 30 mph over speed limit'
  },
  {
    id: 'e-3',
    vehicleId: 'v-2',
    behavior: 'distracted',
    severity: 'low',
    timestamp: Date.now() - 7200000,
    position: {
      lat: 37.783972,
      lng: -122.401297,
      timestamp: Date.now() - 7200000
    },
    details: 'Phone usage detected'
  }
];

// Mock driving sessions
export const mockDrivingSessions: DrivingSession[] = [
  {
    id: 's-1',
    vehicleId: 'v-1',
    startTime: Date.now() - 7200000,
    endTime: Date.now() - 3600000,
    startPosition: {
      lat: 37.753972,
      lng: -122.431297,
      timestamp: Date.now() - 7200000
    },
    endPosition: {
      lat: 37.773972,
      lng: -122.431297,
      timestamp: Date.now() - 3600000
    },
    distance: 12.5,
    averageSpeed: 25,
    maxSpeed: 55,
    events: [mockDrivingEvents[1]],
    score: 75
  },
  {
    id: 's-2',
    vehicleId: 'v-2',
    startTime: Date.now() - 86400000,
    endTime: Date.now() - 82800000,
    startPosition: {
      lat: 37.763972,
      lng: -122.401297,
      timestamp: Date.now() - 86400000
    },
    endPosition: {
      lat: 37.783972,
      lng: -122.411297,
      timestamp: Date.now() - 82800000
    },
    distance: 8.2,
    averageSpeed: 22,
    maxSpeed: 40,
    events: [mockDrivingEvents[2]],
    score: 92
  },
  {
    id: 's-3',
    vehicleId: 'v-1',
    startTime: Date.now() - 3600000,
    startPosition: {
      lat: 37.773972,
      lng: -122.431297,
      timestamp: Date.now() - 3600000
    },
    events: [mockDrivingEvents[0]],
    distance: 5.3,
    averageSpeed: 30,
    maxSpeed: 45
  }
];

// Mock API service functions
export const mockAuth = {
  login: async (email: string, password: string): Promise<{ user: User; token: string }> => {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    if (email === 'demo@example.com' && password === 'password') {
      return { 
        user: currentUser, 
        token: 'mock-jwt-token-' + Math.random().toString(36).substring(2) 
      };
    }
    throw new Error('Invalid credentials');
  },
  
  register: async (name: string, email: string, password: string): Promise<{ user: User; token: string }> => {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return { 
      user: { ...currentUser, name, email }, 
      token: 'mock-jwt-token-' + Math.random().toString(36).substring(2) 
    };
  },
  
  getCurrentUser: async (): Promise<User> => {
    await new Promise(resolve => setTimeout(resolve, 300));
    return currentUser;
  }
};

export const mockVehicleAPI = {
  getVehicles: async (): Promise<Vehicle[]> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    return mockVehicles;
  },
  
  getVehicleById: async (id: string): Promise<Vehicle> => {
    await new Promise(resolve => setTimeout(resolve, 300));
    const vehicle = mockVehicles.find(v => v.id === id);
    if (!vehicle) throw new Error('Vehicle not found');
    return vehicle;
  },
  
  updateVehicleStatus: async (id: string, status: VehicleStatus): Promise<Vehicle> => {
    await new Promise(resolve => setTimeout(resolve, 400));
    const vehicle = mockVehicles.find(v => v.id === id);
    if (!vehicle) throw new Error('Vehicle not found');
    
    vehicle.status = status;
    return vehicle;
  }
};

export const mockDrivingAPI = {
  getEventsByVehicleId: async (vehicleId: string): Promise<DrivingEvent[]> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    return mockDrivingEvents.filter(e => e.vehicleId === vehicleId);
  },
  
  getSessionsByVehicleId: async (vehicleId: string): Promise<DrivingSession[]> => {
    await new Promise(resolve => setTimeout(resolve, 600));
    return mockDrivingSessions.filter(s => s.vehicleId === vehicleId);
  },
  
  getCurrentSession: async (vehicleId: string): Promise<DrivingSession | null> => {
    await new Promise(resolve => setTimeout(resolve, 300));
    return mockDrivingSessions.find(s => s.vehicleId === vehicleId && !s.endTime) || null;
  }
};

// Helper functions for generating random updates (for simulation)
export const generateUpdatedPosition = (vehicle: Vehicle): Position => {
  if (!vehicle.lastPosition) {
    return {
      lat: 37.7749 + (Math.random() * 0.05 - 0.025),
      lng: -122.4194 + (Math.random() * 0.05 - 0.025),
      speed: Math.floor(Math.random() * 60),
      heading: Math.floor(Math.random() * 360),
      timestamp: Date.now()
    };
  }
  
  const latChange = (Math.random() * 0.002 - 0.001) * (vehicle.status === 'driving' ? 1 : 0);
  const lngChange = (Math.random() * 0.002 - 0.001) * (vehicle.status === 'driving' ? 1 : 0);
  const speed = vehicle.status === 'driving' ? 
    Math.min(Math.max((vehicle.lastPosition.speed || 0) + (Math.random() * 10 - 5), 0), 80) : 0;
  
  return {
    lat: vehicle.lastPosition.lat + latChange,
    lng: vehicle.lastPosition.lng + lngChange,
    speed,
    heading: vehicle.status === 'driving' ? 
      (vehicle.lastPosition.heading || 0) + (Math.random() * 10 - 5) : 
      (vehicle.lastPosition.heading || 0),
    timestamp: Date.now()
  };
};

export const simulateVehicleUpdate = async (vehicleId: string): Promise<Vehicle> => {
  const vehicle = mockVehicles.find(v => v.id === vehicleId);
  if (!vehicle) throw new Error('Vehicle not found');
  
  vehicle.lastPosition = generateUpdatedPosition(vehicle);
  
  // Small chance to generate a driving event
  if (vehicle.status === 'driving' && Math.random() < 0.1) {
    const behaviors: DrivingBehavior[] = ['aggressive', 'distracted', 'tired', 'speeding', 'normal'];
    const severities: ('low' | 'medium' | 'high')[] = ['low', 'medium', 'high'];
    
    const newEvent: DrivingEvent = {
      id: 'e-' + uuidv4(),
      vehicleId: vehicle.id,
      behavior: behaviors[Math.floor(Math.random() * behaviors.length)],
      severity: severities[Math.floor(Math.random() * severities.length)],
      timestamp: Date.now(),
      position: { ...vehicle.lastPosition },
      details: 'Simulated event'
    };
    
    mockDrivingEvents.push(newEvent);
    
    // Add to current session if exists
    const currentSession = mockDrivingSessions.find(s => s.vehicleId === vehicleId && !s.endTime);
    if (currentSession) {
      currentSession.events.push(newEvent);
      
      // Recalculate score based on new event
      if (newEvent.behavior !== 'normal') {
        currentSession.score = currentSession.score ? 
          Math.max(currentSession.score - (newEvent.severity === 'high' ? 15 : newEvent.severity === 'medium' ? 10 : 5), 0) : 
          85;
      }
    }
  }
  
  return vehicle;
};

// Websocket simulation
export type DataUpdateListener = (data: any) => void;
let listeners: { type: string; callback: DataUpdateListener }[] = [];

export const mockWebSocket = {
  subscribe: (type: string, callback: DataUpdateListener) => {
    listeners.push({ type, callback });
    return () => {
      listeners = listeners.filter(listener => listener.callback !== callback);
    };
  },
  
  publishUpdate: (type: string, data: any) => {
    listeners
      .filter(listener => listener.type === type)
      .forEach(listener => listener.callback(data));
  }
};

// Call this periodically to simulate updates
export const simulateAllVehicleUpdates = async () => {
  for (const vehicle of mockVehicles) {
    if (vehicle.status !== 'offline') {
      const updated = await simulateVehicleUpdate(vehicle.id);
      mockWebSocket.publishUpdate('vehicle-update', updated);
    }
  }
};
