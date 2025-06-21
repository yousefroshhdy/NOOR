
import { 
  mockVehicleAPI, 
  mockDrivingAPI, 
  Vehicle, 
  DrivingEvent, 
  DrivingSession, 
  VehicleStatus,
  mockWebSocket,
  simulateAllVehicleUpdates
} from './mockData';

// Initialize periodic update simulation
let updateInterval: number | null = null;

// Vehicle data functions
export const getVehicles = async (): Promise<Vehicle[]> => {
  try {
    return await mockVehicleAPI.getVehicles();
  } catch (error) {
    console.error('Failed to fetch vehicles:', error);
    return [];
  }
};

export const getVehicleById = async (id: string): Promise<Vehicle | null> => {
  try {
    return await mockVehicleAPI.getVehicleById(id);
  } catch (error) {
    console.error(`Failed to fetch vehicle ${id}:`, error);
    return null;
  }
};

export const updateVehicleStatus = async (id: string, status: VehicleStatus): Promise<boolean> => {
  try {
    await mockVehicleAPI.updateVehicleStatus(id, status);
    return true;
  } catch (error) {
    console.error(`Failed to update vehicle ${id} status:`, error);
    return false;
  }
};

// Driving data functions
export const getVehicleEvents = async (vehicleId: string): Promise<DrivingEvent[]> => {
  try {
    return await mockDrivingAPI.getEventsByVehicleId(vehicleId);
  } catch (error) {
    console.error(`Failed to fetch events for vehicle ${vehicleId}:`, error);
    return [];
  }
};

export const getVehicleSessions = async (vehicleId: string): Promise<DrivingSession[]> => {
  try {
    return await mockDrivingAPI.getSessionsByVehicleId(vehicleId);
  } catch (error) {
    console.error(`Failed to fetch sessions for vehicle ${vehicleId}:`, error);
    return [];
  }
};

export const getCurrentSession = async (vehicleId: string): Promise<DrivingSession | null> => {
  try {
    return await mockDrivingAPI.getCurrentSession(vehicleId);
  } catch (error) {
    console.error(`Failed to fetch current session for vehicle ${vehicleId}:`, error);
    return null;
  }
};

// Realtime updates
export const subscribeToVehicleUpdates = (
  callback: (vehicle: Vehicle) => void
): () => void => {
  // Start simulation if not already running
  if (!updateInterval) {
    updateInterval = window.setInterval(() => {
      simulateAllVehicleUpdates();
    }, 5000) as unknown as number;
  }
  
  return mockWebSocket.subscribe('vehicle-update', callback);
};

export const subscribeToVehicleEvents = (
  vehicleId: string,
  callback: (event: DrivingEvent) => void
): () => void => {
  return mockWebSocket.subscribe(`vehicle-${vehicleId}-event`, callback);
};

// Cleanup function
export const cleanup = (): void => {
  if (updateInterval) {
    clearInterval(updateInterval);
    updateInterval = null;
  }
};
